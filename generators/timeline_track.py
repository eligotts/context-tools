"""timeline_track — stateful narrative world.

A timeline of events that mutate the state of named objects (owner,
location, intact-or-destroyed). The model walks the timeline, maintains
a state dict that gets *overwritten* as events come in, and answers a
snapshot query at some time T.

Designed to force *overwrite* of the scratchpad: state has fixed schema
but values mutate. Append play (logging every event) loses against
``max_context_chars``; edit play (a single state dict that overwrites
in place) wins regardless of timeline length.

State schema
------------
::

    {
        "events": [
            {"type": "CREATE",   "object": "x1y", "owner": "Alice", "location": "tavern", "time": 0},
            {"type": "TRANSFER", "object": "x1y", "from": "Alice",  "to": "Bob", "time": 5},
            {"type": "MOVE",     "object": "x1y", "from": "tavern", "to": "forest", "time": 12},
            {"type": "DESTROY",  "object": "x1y", "time": 30},
            ...
        ],
        "objects":   ["x1y", "z3w", ...],
        "actors":    ["Alice", "Bob", ...],
        "locations": ["tavern", "forest", ...],
        "_question_meta": {                    # stripped before sandbox sees it
            "template": "owner_at_time" | "count_owned_at_time",
            "object":   "x1y" | None,
            "actor":    "Alice" | None,
            "time":     int,
        },
    }

Tools the model sees
--------------------
- ``read_event(i: int) -> dict``           — single event at index i
- ``read_events(start, end) -> list[dict]`` — events in [start, end), capped at 5
- pre-seeded: ``N_EVENTS``, ``OBJECTS``, ``ACTORS``, ``LOCATIONS``

"At time T" means "after the first T events have been applied" —
state at time T reflects events with index ``< T``.
"""

from __future__ import annotations

import json
import random
import string
from typing import Any

from .base import QueryTemplate, WorldGenerator


# ---------------------------------------------------------------------------
# Universe (fixed pools we sample from)
# ---------------------------------------------------------------------------

_FIRST_NAMES = [
    "Alice", "Bob", "Cara", "Dan", "Evie", "Finn", "Gus", "Hana",
    "Ivan", "Joy", "Kai", "Lia", "Mira", "Nico", "Omar", "Pia",
]

_LOCATION_POOL = [
    "tavern", "forest", "harbor", "market", "mountain", "library",
    "tower", "cellar", "garden", "river",
]


TIMELINE_TRACK_TEMPLATES = [
    QueryTemplate(
        name="owner_at_time",
        description=(
            "Who is the owner of object X at time T (or 'none' if destroyed/uncreated)?"
        ),
        min_turns=3,
        max_turns=15,
        answer_type="str",
    ),
    QueryTemplate(
        name="count_owned_at_time",
        description=(
            "How many non-destroyed objects does actor A currently own at time T?"
        ),
        min_turns=4,
        max_turns=20,
        answer_type="int",
    ),
    QueryTemplate(
        name="most_active_actor",
        description=(
            "Which actor receives the most TRANSFER actions before time T?"
        ),
        min_turns=5,
        max_turns=25,
        answer_type="str",
    ),
    QueryTemplate(
        name="owned_transfer_top_k_at_time",
        description=(
            "Among objects currently owned by an actor at time T, rank top-K by prior transfer count."
        ),
        min_turns=8,
        max_turns=30,
        answer_type="json",
    ),
    QueryTemplate(
        name="checkpoint_actor_audit",
        description=(
            "For several checkpoints, combine per-interval transfer counts with the live ownership snapshot."
        ),
        min_turns=8,
        max_turns=35,
        answer_type="json",
    ),
]


# ---------------------------------------------------------------------------
# Pure helpers (also used by the solver)
# ---------------------------------------------------------------------------


def replay_to(events: list[dict], T: int, objects: list[str]) -> dict[str, dict]:
    """Replay events[:T] and return per-object state.

    Each entry: {"owner": str|None, "location": str|None, "exists": bool}.
    Objects never seen in events are returned with their initial all-None
    inactive state.
    """
    state: dict[str, dict] = {
        o: {"owner": None, "location": None, "exists": False} for o in objects
    }
    n = max(0, min(T, len(events)))
    for i in range(n):
        e = events[i]
        obj = e.get("object")
        if obj not in state:
            continue
        t = e.get("type")
        if t == "CREATE":
            state[obj] = {
                "owner": e.get("owner"),
                "location": e.get("location"),
                "exists": True,
            }
        elif t == "TRANSFER":
            if state[obj]["exists"]:
                state[obj]["owner"] = e.get("to")
        elif t == "MOVE":
            if state[obj]["exists"]:
                state[obj]["location"] = e.get("to")
        elif t == "DESTROY":
            state[obj]["exists"] = False
    return state


def transfer_receiver_counts(events: list[dict], T: int, actors: list[str]) -> dict[str, int]:
    """Count TRANSFER events by recipient actor before T."""
    counts = {a: 0 for a in actors}
    for e in events[: max(0, min(T, len(events)))]:
        if e.get("type") == "TRANSFER":
            to_actor = e.get("to")
            if to_actor in counts:
                counts[to_actor] += 1
    return counts


def transfer_object_counts(events: list[dict], T: int, objects: list[str]) -> dict[str, int]:
    """Count prior TRANSFER events per object before T."""
    counts = {o: 0 for o in objects}
    for e in events[: max(0, min(T, len(events)))]:
        if e.get("type") == "TRANSFER":
            obj = e.get("object")
            if obj in counts:
                counts[obj] += 1
    return counts


def receiver_lead_changes(events: list[dict], T: int, actors: list[str]) -> int:
    """Count changes to the current recipient-count leader before T."""
    counts = {a: 0 for a in actors}
    leader = sorted(actors)[0] if actors else ""
    changes = 0
    for e in events[: max(0, min(T, len(events)))]:
        if e.get("type") != "TRANSFER":
            continue
        to_actor = e.get("to")
        if to_actor not in counts:
            continue
        counts[to_actor] += 1
        new_leader = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        if new_leader != leader:
            changes += 1
            leader = new_leader
    return changes


def checkpoint_actor_audit(
    events: list[dict],
    checkpoints: list[int],
    objects: list[str],
    actors: list[str],
) -> list[list[Any]]:
    """Return checkpoint audit rows.

    For each checkpoint C, count TRANSFER recipients in the interval since the
    previous checkpoint, choose the leading recipient actor (tie by actor name),
    then report how many live objects that same actor owns after applying
    events[:C]. This deliberately mixes state that persists across checkpoints
    with interval state that must be reset exactly once per checkpoint.
    """
    snap: dict[str, dict] = {
        o: {"owner": None, "location": None, "exists": False}
        for o in objects
    }
    rows: list[list[Any]] = []
    cursor = 0
    n_events = len(events)
    actor_set = set(actors)

    for checkpoint in checkpoints:
        C = max(0, min(int(checkpoint), n_events))
        interval_counts = {a: 0 for a in actors}
        for e in events[cursor:C]:
            obj = e.get("object")
            if obj not in snap:
                continue
            typ = e.get("type")
            if typ == "CREATE":
                snap[obj] = {
                    "owner": e.get("owner"),
                    "location": e.get("location"),
                    "exists": True,
                }
            elif typ == "TRANSFER":
                to_actor = e.get("to")
                if snap[obj]["exists"]:
                    snap[obj]["owner"] = to_actor
                if to_actor in actor_set:
                    interval_counts[to_actor] += 1
            elif typ == "MOVE":
                if snap[obj]["exists"]:
                    snap[obj]["location"] = e.get("to")
            elif typ == "DESTROY":
                snap[obj]["exists"] = False

        actor, received = sorted(
            interval_counts.items(), key=lambda kv: (-kv[1], kv[0])
        )[0]
        owned_count = sum(
            1
            for o in objects
            if snap[o]["exists"] and snap[o]["owner"] == actor
        )
        rows.append([C, actor, received, owned_count])
        cursor = C

    return rows


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _short_id(rng: random.Random) -> str:
    return (
        rng.choice(string.ascii_lowercase)
        + str(rng.randint(0, 9))
        + rng.choice(string.ascii_lowercase)
    )


# ---------------------------------------------------------------------------
# Flavor text for events. Each event gets a `narrative` prose blob that's
# never used by the solver — pure noise that bloats raw read_event() /
# read_events() returns. Forces extraction of the structured fields
# (type/object/owner/from/to) instead of journaling raw event dicts.
# ---------------------------------------------------------------------------

_EVENT_NARRATIVES = [
    "The action was witnessed and confirmed by the standing oversight committee.",
    "Routine procedural verification was completed prior to recording the event.",
    "Documentation was logged in accordance with the standard chain-of-custody protocol.",
    "All parties acknowledged the entry into the regional record system.",
    "Subsequent administrative steps proceeded without notable incident.",
    "The transition was processed during a regularly scheduled audit window.",
    "Notarial signatures were collected and filed for retrospective review.",
    "Surveillance footage corroborates the timing of the recorded event.",
    "A supplemental note was attached referencing the relevant policy clause.",
    "The change was reflected in both primary and backup ledger systems.",
    "Cross-references were updated to maintain narrative continuity.",
    "Transient discrepancies in reporting were resolved within the same cycle.",
    "Background checks were completed for all parties involved in the action.",
    "Compliance flags were reviewed and cleared by the duty officer on shift.",
    "An entry was made in the regional sequence log for traceability.",
    "Standard formality requirements were met before the event was finalized.",
]


def _make_narrative(rng: random.Random) -> str:
    """Return a 2-3 sentence prose blob (~100-200 chars) attached to each
    event. Constraint-irrelevant — pure observational noise."""
    n = rng.randint(2, 3)
    return " ".join(rng.sample(_EVENT_NARRATIVES, n))


def _generate_timeline(
    n_events: int,
    n_objects: int,
    n_actors: int,
    n_locations: int,
    rng: random.Random,
) -> tuple[list[dict], list[str], list[str], list[str]]:
    """Produce a lifecycle-respecting event timeline.

    Returns (events, objects, actors, locations).
    """
    objects: list[str] = []
    seen: set[str] = set()
    while len(objects) < n_objects:
        oid = _short_id(rng)
        if oid in seen:
            continue
        seen.add(oid)
        objects.append(oid)

    actors = list(rng.sample(_FIRST_NAMES, min(n_actors, len(_FIRST_NAMES))))
    locations = list(rng.sample(_LOCATION_POOL, min(n_locations, len(_LOCATION_POOL))))

    obj_state: dict[str, dict] = {}  # oid -> {owner, location, exists}
    events: list[dict] = []

    for t in range(n_events):
        active = [o for o in objects if obj_state.get(o, {}).get("exists")]
        pending = [o for o in objects if o not in obj_state]

        if not obj_state:
            # First event has to introduce something
            evt_type = "CREATE"
        elif not active:
            if pending:
                evt_type = "CREATE"
            else:
                # World has shut down; stop emitting
                break
        else:
            choices: list[str] = []
            if pending:
                choices.extend(["CREATE"] * 2)
            choices.extend(["TRANSFER"] * 4)
            choices.extend(["MOVE"] * 4)
            if t > n_events // 4 and len(active) > 1:
                choices.append("DESTROY")
            evt_type = rng.choice(choices)

        if evt_type == "CREATE":
            obj = pending[0] if pending else rng.choice(objects)
            owner = rng.choice(actors)
            location = rng.choice(locations)
            evt: dict = {
                "type": "CREATE",
                "object": obj,
                "owner": owner,
                "location": location,
                "time": t,
            }
            obj_state[obj] = {"owner": owner, "location": location, "exists": True}

        elif evt_type == "TRANSFER":
            obj = rng.choice(active)
            old = obj_state[obj]["owner"]
            cands = [a for a in actors if a != old]
            new = rng.choice(cands) if cands else old
            evt = {
                "type": "TRANSFER",
                "object": obj,
                "from": old,
                "to": new,
                "time": t,
            }
            obj_state[obj]["owner"] = new

        elif evt_type == "MOVE":
            obj = rng.choice(active)
            old = obj_state[obj]["location"]
            cands = [l for l in locations if l != old]
            new = rng.choice(cands) if cands else old
            evt = {
                "type": "MOVE",
                "object": obj,
                "from": old,
                "to": new,
                "time": t,
            }
            obj_state[obj]["location"] = new

        else:  # DESTROY
            obj = rng.choice(active)
            evt = {"type": "DESTROY", "object": obj, "time": t}
            obj_state[obj]["exists"] = False

        evt["narrative"] = _make_narrative(rng)
        events.append(evt)

    return events, objects, actors, locations


# ---------------------------------------------------------------------------
# Difficulty mapping
# ---------------------------------------------------------------------------


def _difficulty_to_timeline_params(
    depth: int, breadth: int
) -> tuple[int, int, int, int]:
    """Fallback for depth/breadth → (n_events, n_objects, n_actors, n_locations)
    when ``difficulty`` is not supplied."""
    n_events = max(40, 20 * depth)              # 40 .. ~100+
    n_objects = max(6, min(12, breadth + 2))    # 6 .. 12
    n_actors = max(4, min(8, breadth))          # 4 .. 8
    n_locations = max(3, min(7, breadth - 1))   # 3 .. 7
    return n_events, n_objects, n_actors, n_locations


# Difficulty axis: scales the state-dict width × per-key churn — i.e. the
# total number of overwrites the model has to perform across the whole
# scratchpad. d=1..3 are calibrated for a 15-turn training cap: with
# read_events capped at 5 and 5 calls/turn, snapshot tasks usually require
# ~5/7/10 scan turns rather than ending after only a few chunks.
DIFFICULTY_AXIS: dict[int, dict] = {
    1: {"n_objects":  8, "events_per_object": 12, "n_actors": 5, "n_locations": 4},
    2: {"n_objects": 10, "events_per_object": 15, "n_actors": 6, "n_locations": 5},
    3: {"n_objects": 12, "events_per_object": 18, "n_actors": 7, "n_locations": 6},
    4: {"n_objects": 14, "events_per_object": 14, "n_actors": 8, "n_locations": 7},
    5: {"n_objects": 16, "events_per_object": 16, "n_actors": 8, "n_locations": 7},
}


# ---------------------------------------------------------------------------
# WorldGenerator
# ---------------------------------------------------------------------------


class TimelineTrackWorld(WorldGenerator):
    """Stateful narrative: replay events to answer a snapshot query."""

    world_type = "timeline_track"

    def get_query_templates(self) -> list[QueryTemplate]:
        return TIMELINE_TRACK_TEMPLATES

    def get_system_prompt(self) -> str:
        return _TIMELINE_PROMPT_PRELUDE

    def generate_state(
        self,
        depth: int,
        breadth: int,
        rng: Any,
        *,
        difficulty: int | None = None,
    ) -> dict:
        if difficulty is not None:
            d = max(1, min(5, int(difficulty)))
            axis = DIFFICULTY_AXIS[d]
            n_objects = axis["n_objects"]
            events_per_object = axis["events_per_object"]
            n_actors = axis["n_actors"]
            n_locations = axis["n_locations"]
            n_events = n_objects * events_per_object
        else:
            n_events, n_objects, n_actors, n_locations = (
                _difficulty_to_timeline_params(depth, breadth)
            )
        events, objects, actors, locations = _generate_timeline(
            n_events=n_events,
            n_objects=n_objects,
            n_actors=n_actors,
            n_locations=n_locations,
            rng=rng,
        )
        return {
            "events": events,
            "objects": objects,
            "actors": actors,
            "locations": locations,
            "depth": depth,
            "breadth": breadth,
        }

    def generate_query_rich(
        self,
        template: QueryTemplate,
        state: dict,
        target_turns: int,
        rng: Any,
    ) -> dict | None:
        events: list[dict] = state["events"]
        objects: list[str] = state["objects"]
        actors: list[str] = state["actors"]
        if not events or not objects:
            return None

        # Pick T in the latter portion of the timeline so enough state has
        # accumulated. Range: [n_events // 3, n_events].
        n = len(events)
        T_lo = max(2, n // 3)
        T_hi = n

        if template.name == "owner_at_time":
            return self._gen_owner_at_time(
                events, objects, T_lo, T_hi, state, rng
            )
        if template.name == "count_owned_at_time":
            return self._gen_count_owned_at_time(
                events, objects, actors, T_lo, T_hi, state, rng
            )
        if template.name == "most_active_actor":
            return self._gen_most_active_actor(
                events, actors, T_lo, T_hi, state, rng
            )
        if template.name == "owned_transfer_top_k_at_time":
            return self._gen_owned_transfer_top_k_at_time(
                events, objects, actors, T_lo, T_hi, state, rng
            )
        if template.name == "checkpoint_actor_audit":
            return self._gen_checkpoint_actor_audit(
                events, objects, actors, state, rng
            )
        return None

    # -- per-template helpers ------------------------------------------------

    def _gen_owner_at_time(
        self,
        events: list[dict],
        objects: list[str],
        T_lo: int,
        T_hi: int,
        state: dict,
        rng: random.Random,
    ) -> dict | None:
        # Try several (object, T) pairs; prefer objects with multiple events
        # affecting them so the answer is a real overwrite, not a one-shot
        # CREATE lookup.
        events_by_obj: dict[str, list[int]] = {}
        for i, e in enumerate(events):
            o = e.get("object")
            if o is not None:
                events_by_obj.setdefault(o, []).append(i)
        candidates = sorted(
            events_by_obj.items(),
            key=lambda kv: -len(kv[1]),
        )
        if not candidates:
            return None

        for obj, idxs in candidates[: max(5, len(candidates) // 2)]:
            for _ in range(20):
                T = rng.randint(T_lo, T_hi)
                snap = replay_to(events, T, objects)
                rec = snap[obj]
                if rec["exists"]:
                    answer = str(rec["owner"]) if rec["owner"] else "none"
                else:
                    # Skip uncreated/destroyed snapshots ~half the time so we
                    # do see "none" answers but most are real names.
                    if rng.random() < 0.5:
                        continue
                    answer = "none"
                state["_question_meta"] = {
                    "template": "owner_at_time",
                    "object": obj,
                    "actor": None,
                    "time": T,
                }
                query_text = (
                    f"At time T={T}, who is the owner of object '{obj}'? "
                    f"\"At time T\" means after the first T events have been "
                    f"applied (events with index < T). Answer with the actor "
                    f"name, or 'none' if the object is destroyed or has not "
                    f"yet been created at time T. Use read_event(i) or "
                    f"read_events(start, end) (capped at 5/call) to walk the "
                    f"timeline. N_EVENTS, OBJECTS, ACTORS, LOCATIONS are "
                    f"pre-seeded. Submit via submit_answer(answer)."
                )
                return {
                    "query_text": query_text,
                    "expected_answer": answer,
                    "actual_depth": state.get("depth", 0),
                    "actual_breadth": state.get("breadth", 0),
                    "target_entities": [obj],
                    "parameter_refs": [],
                    "rules": [],
                }
        return None

    def _gen_count_owned_at_time(
        self,
        events: list[dict],
        objects: list[str],
        actors: list[str],
        T_lo: int,
        T_hi: int,
        state: dict,
        rng: random.Random,
    ) -> dict | None:
        # Try until we get an actor + T pair where the count is >= 1 and
        # there's been enough event activity that several objects are alive.
        for _ in range(60):
            T = rng.randint(T_lo, T_hi)
            snap = replay_to(events, T, objects)
            alive = [o for o in objects if snap[o]["exists"]]
            if len(alive) < 2:
                continue
            actor = rng.choice(actors)
            count = sum(1 for o in alive if snap[o]["owner"] == actor)
            if count == 0:
                # Half the time accept zero so the model can't always count
                # on the answer being non-zero.
                if rng.random() > 0.3:
                    continue
            state["_question_meta"] = {
                "template": "count_owned_at_time",
                "object": None,
                "actor": actor,
                "time": T,
            }
            query_text = (
                f"At time T={T}, how many non-destroyed objects does actor "
                f"'{actor}' currently own? \"At time T\" means after the "
                f"first T events have been applied (events with index < T). "
                f"Use read_event(i) or read_events(start, end) (capped at "
                f"5/call) to walk the timeline. N_EVENTS, OBJECTS, ACTORS, "
                f"LOCATIONS are pre-seeded. Submit the integer count via "
                f"submit_answer(n)."
            )
            return {
                "query_text": query_text,
                "expected_answer": str(count),
                "actual_depth": state.get("depth", 0),
                "actual_breadth": state.get("breadth", 0),
                "target_entities": [],
                "parameter_refs": [],
                "rules": [],
            }
        return None

    def _gen_most_active_actor(
        self,
        events: list[dict],
        actors: list[str],
        T_lo: int,
        T_hi: int,
        state: dict,
        rng: random.Random,
    ) -> dict | None:
        # Require enough churn that a single scalar leader is unreliable unless
        # the model also keeps the per-actor counts in REPL state.
        for _ in range(80):
            T = rng.randint(T_lo, T_hi)
            counts = transfer_receiver_counts(events, T, actors)
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if not ranked or ranked[0][1] < 4:
                continue
            margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else ranked[0][1]
            changes = receiver_lead_changes(events, T, actors)
            if margin > 3 or changes < 3:
                continue
            answer = ranked[0][0]
            state["_question_meta"] = {
                "template": "most_active_actor",
                "object": None,
                "actor": None,
                "time": T,
                "lead_changes": changes,
                "counts": counts,
            }
            query_text = (
                f"Before time T={T} (events with index < {T}), which actor "
                f"received the most TRANSFER actions? Count a TRANSFER for "
                f"the actor named in its `to` field. If actors tie, choose "
                f"the alphabetically earliest actor. Use read_event(i) or "
                f"read_events(start, end) (capped at 5/call) to scan the "
                f"timeline. ACTORS and N_EVENTS are pre-seeded. Submit the "
                f"actor name via submit_answer(answer)."
            )
            return {
                "query_text": query_text,
                "expected_answer": answer,
                "actual_depth": state.get("depth", 0),
                "actual_breadth": state.get("breadth", 0),
                "target_entities": [],
                "parameter_refs": [],
                "rules": [],
            }
        return None

    def _gen_owned_transfer_top_k_at_time(
        self,
        events: list[dict],
        objects: list[str],
        actors: list[str],
        T_lo: int,
        T_hi: int,
        state: dict,
        rng: random.Random,
    ) -> dict | None:
        # This task requires two live state structures: current object owners
        # and per-object transfer counts. The final answer is compact but the
        # intermediate state is wider than a scalar.
        k = 3 if len(objects) <= 10 else 4
        for _ in range(120):
            T = rng.randint(T_lo, T_hi)
            snap = replay_to(events, T, objects)
            transfer_counts = transfer_object_counts(events, T, objects)
            viable: list[tuple[str, list[str]]] = []
            for actor in actors:
                owned = [
                    o for o in objects
                    if snap[o]["exists"] and snap[o]["owner"] == actor
                ]
                if len(owned) >= k + 1:
                    viable.append((actor, owned))
            if not viable:
                continue
            actor, owned = rng.choice(viable)
            ranked = sorted(
                ((o, transfer_counts[o]) for o in owned),
                key=lambda item: (-item[1], item[0]),
            )
            if ranked[k - 1][1] == 0:
                continue
            # Prefer nontrivial boards where at least one non-answer object has
            # a close count, so the model cannot safely remember only the leader.
            if len(ranked) > k and ranked[k - 1][1] - ranked[k][1] > 2:
                continue
            answer = [[o, c] for o, c in ranked[:k]]
            state["_question_meta"] = {
                "template": "owned_transfer_top_k_at_time",
                "object": None,
                "actor": actor,
                "time": T,
                "k": k,
                "transfer_counts": transfer_counts,
            }
            query_text = (
                f"At time T={T}, consider the non-destroyed objects currently "
                f"owned by actor '{actor}'. Among those objects, find the top {k} "
                f"by number of prior TRANSFER events involving that object before "
                f"T. Rank by transfer count descending, breaking ties by object id "
                f"ascending. Submit JSON as [[object_id, count], ...]. Use "
                f"read_event(i) or read_events(start, end) (capped at 5/call) to "
                f"track current owners/existence and transfer counts. OBJECTS, "
                f"ACTORS, and N_EVENTS are pre-seeded."
            )
            return {
                "query_text": query_text,
                "expected_answer": json.dumps(answer, separators=(",", ":")),
                "actual_depth": state.get("depth", 0),
                "actual_breadth": state.get("breadth", 0),
                "target_entities": [actor] + [o for o, _ in answer],
                "parameter_refs": [],
                "rules": [],
            }
        return None

    def _gen_checkpoint_actor_audit(
        self,
        events: list[dict],
        objects: list[str],
        actors: list[str],
        state: dict,
        rng: random.Random,
    ) -> dict | None:
        # This task is explicitly about checkpoint discipline:
        #
        # * object ownership/existence persists across all checkpoints;
        # * recipient counts are local to the interval since the previous
        #   checkpoint and must be reset at the right time;
        # * the result list grows compactly while the scan cursor advances.
        #
        # Reprocessing a chunk, restoring a stale cursor, or forgetting to reset
        # interval counts gives plausible but wrong rows.
        n = len(events)
        if n < 30 or not actors:
            return None

        target_n = 3 if n < 90 else 4 if n < 140 else 5
        min_gap = max(8, n // (target_n * 3))

        for _ in range(180):
            checkpoints: list[int] = []
            prev = 0
            for i in range(target_n):
                remaining = target_n - i - 1
                lo = max(prev + min_gap, int((i + 1) * n / (target_n + 2)) - min_gap)
                hi = min(
                    n - remaining * min_gap,
                    int((i + 2) * n / (target_n + 2)) + min_gap,
                )
                if lo >= hi:
                    break
                C = rng.randint(lo, hi)
                checkpoints.append(C)
                prev = C
            if len(checkpoints) != target_n:
                continue
            checkpoints = sorted(set(checkpoints))
            if len(checkpoints) != target_n:
                continue

            answer = checkpoint_actor_audit(events, checkpoints, objects, actors)
            if any(row[2] <= 0 for row in answer):
                continue
            if len({row[1] for row in answer}) < min(3, target_n):
                continue
            if sum(1 for row in answer if row[3] > 0) < target_n - 1:
                continue

            state["_question_meta"] = {
                "template": "checkpoint_actor_audit",
                "object": None,
                "actor": None,
                "time": checkpoints[-1],
                "checkpoints": checkpoints,
                "answer": answer,
            }
            query_text = (
                f"Process the timeline at checkpoints {checkpoints}. For each "
                f"checkpoint C, look only at events in the interval since the "
                f"previous checkpoint (starting from 0 for the first interval) "
                f"and find the actor who received the most TRANSFER actions in "
                f"that interval. Count a TRANSFER for the actor in its `to` "
                f"field; break ties by alphabetically earliest actor name. Also "
                f"report how many non-destroyed objects that same actor owns at "
                f"checkpoint C, after events with index < C have been applied. "
                f"Submit JSON as [[checkpoint, actor, interval_transfer_count, "
                f"owned_count_at_checkpoint], ...] in checkpoint order. You need "
                f"to carry object ownership/existence forward across checkpoints "
                f"but reset interval transfer counts after each checkpoint. Use "
                f"read_event(i) or read_events(start, end) (capped at 5/call). "
                f"OBJECTS, ACTORS, LOCATIONS, and N_EVENTS are pre-seeded."
            )
            return {
                "query_text": query_text,
                "expected_answer": json.dumps(answer, separators=(",", ":")),
                "actual_depth": state.get("depth", 0),
                "actual_breadth": state.get("breadth", 0),
                "target_entities": [row[1] for row in answer],
                "parameter_refs": [],
                "rules": [],
            }
        return None

    def generate_query(
        self,
        template: QueryTemplate,
        state: dict,
        target_turns: int,
        rng: Any,
    ) -> tuple[str, Any, int, int]:
        rich = self.generate_query_rich(template, state, target_turns, rng)
        if rich is None:
            return ("", "0", state.get("depth", 0), state.get("breadth", 0))
        return (
            rich["query_text"],
            rich["expected_answer"],
            rich["actual_depth"],
            rich["actual_breadth"],
        )


_TIMELINE_PROMPT_PRELUDE = (
    "timeline_track world: replay events with read_event / read_events to "
    "compute the state of objects at a query time T."
)
