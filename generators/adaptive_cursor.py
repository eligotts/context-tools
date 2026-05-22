"""adaptive_cursor — ordinary REPL observation tasks.

This family is built without a synthetic tool-call budget. The model receives
one opaque observation handle and calls the ordinary Python function
``observe(handle)`` to get a cursor page string. In context-rewrite mode the
model only sees what it writes to ``context_window``, so the data pressure is
on deciding what returned page content to keep, compact, overwrite, or discard.

The concrete task is a cursor-led ledger audit. Pages contain mutating object
events plus occasional checkpoints. The answer requires carrying ownership
state forward while resetting interval transfer counts at checkpoints.
"""

from __future__ import annotations

import json
import random
from typing import Any

from .base import QueryTemplate, TrainingExample, WorldGenerator
from .natural_ids import NaturalIdBank


ACTORS = [
    "Alice", "Bob", "Cara", "Evie", "Finn", "Gus", "Hana", "Ivan",
    "Joy", "Kai", "Lia", "Mira", "Nico", "Omar", "Pia",
]

LOCATIONS = [
    "archive", "cellar", "garden", "harbor", "library", "market",
    "river", "tower", "vault",
]

TAB_WORDS = [
    "cedar", "amber", "indigo", "silver", "copper", "violet", "maple",
    "saffron", "onyx", "pearl", "willow", "basalt",
]


ADAPTIVE_CURSOR_TEMPLATES = [
    QueryTemplate(
        name="cursor_checkpoint_audit",
        description=(
            "Follow opaque observation cursors and audit checkpoint transfer "
            "leaders plus live ownership counts."
        ),
        min_turns=4,
        max_turns=15,
        answer_type="json",
    ),
]


def _object_ids(id_bank: NaturalIdBank, n: int) -> list[str]:
    return sorted(id_bank.fresh(words=2) for _ in range(n))


def _new_handle(id_bank: NaturalIdBank) -> str:
    return id_bank.fresh(words=2)


class AdaptiveCursorWorld(WorldGenerator):
    world_type = "adaptive_cursor"

    def get_query_templates(self) -> list[QueryTemplate]:
        return list(ADAPTIVE_CURSOR_TEMPLATES)

    def get_system_prompt(self) -> str:
        return (
            "adaptive_cursor world: call observe(handle) to return exactly one "
            "cursor page string. Maintain compact state and submit "
            "the final JSON audit rows."
        )

    def generate_state(
        self,
        depth: int,
        breadth: int,
        rng: random.Random,
        *,
        difficulty: int | None = None,
    ) -> dict:
        difficulty = max(0, min(4, difficulty if difficulty is not None else 1))
        if difficulty == 0:
            n_pages, entries_per_page, n_objects, n_actors, n_checkpoints = 3, 1, 2, 2, 1
            destroy_roll = 1.01
        elif difficulty == 1:
            n_pages, entries_per_page, n_objects, n_actors, n_checkpoints = 4, 1, 3, 3, 2
            destroy_roll = 1.01
        elif difficulty == 2:
            # d2-lite: same multi-checkpoint continuation pressure as harder
            # tasks, but fewer ledger updates and rare lifecycle churn.
            n_pages, entries_per_page, n_objects, n_actors, n_checkpoints = 5, 1, 3, 3, 2
            destroy_roll = 0.97
        elif difficulty == 3:
            # d2-hard: previous d2 shape.
            n_pages, entries_per_page, n_objects, n_actors, n_checkpoints = 6, 2, 4, 3, 2
            destroy_roll = 0.88
        else:
            # d3: previous hardest shape.
            n_pages, entries_per_page, n_objects, n_actors, n_checkpoints = 8, 2, 5, 4, 3
            destroy_roll = 0.88

        actors = sorted(rng.sample(ACTORS, n_actors))
        locations = sorted(rng.sample(LOCATIONS, min(len(LOCATIONS), max(4, n_actors))))
        id_bank = NaturalIdBank(rng)
        objects = _object_ids(id_bank, n_objects)
        start_handle = id_bank.fresh(words=2)
        handles = [start_handle] + [_new_handle(id_bank) for _ in range(n_pages - 1)]
        checkpoint_pages = sorted(rng.sample(range(1, n_pages + 1), n_checkpoints))
        checkpoint_pages[-1] = n_pages
        checkpoint_pages = sorted(set(checkpoint_pages))

        pages_events: list[list[dict[str, Any]]] = []
        owners: dict[str, str | None] = {o: None for o in objects}
        exists: dict[str, bool] = {o: False for o in objects}
        event_index = 0

        for page_idx in range(n_pages):
            page_events: list[dict[str, Any]] = []
            for _ in range(entries_per_page):
                live = [o for o in objects if exists[o]]
                inactive = [o for o in objects if not exists[o]]
                force_create = not live or (inactive and rng.random() < 0.28)
                if force_create:
                    obj = rng.choice(inactive)
                    actor = rng.choice(actors)
                    loc = rng.choice(locations)
                    e = {
                        "idx": event_index,
                        "type": "CREATE",
                        "object": obj,
                        "owner": actor,
                        "location": loc,
                    }
                    owners[obj] = actor
                    exists[obj] = True
                else:
                    obj = rng.choice(live)
                    roll = rng.random()
                    if roll < 0.68:
                        old = owners[obj] or rng.choice(actors)
                        choices = [a for a in actors if a != old] or actors
                        new = rng.choice(choices)
                        e = {
                            "idx": event_index,
                            "type": "TRANSFER",
                            "object": obj,
                            "from": old,
                            "to": new,
                        }
                        owners[obj] = new
                    elif roll < destroy_roll:
                        e = {
                            "idx": event_index,
                            "type": "MOVE",
                            "object": obj,
                            "to": rng.choice(locations),
                        }
                    else:
                        e = {
                            "idx": event_index,
                            "type": "DESTROY",
                            "object": obj,
                        }
                        exists[obj] = False
                        owners[obj] = None
                page_events.append(e)
                event_index += 1
            pages_events.append(page_events)

        answer_rows, states_after, checkpoint_rows = self._simulate(
            pages_events,
            checkpoint_pages,
            objects,
            actors,
        )
        # Regenerate if any interval has no transfer; zero-transfer intervals
        # create ambiguity and weak pressure.
        if any(row[2] == 0 for row in answer_rows):
            return self.generate_state(depth, breadth, rng, difficulty=difficulty)

        pages: dict[str, dict[str, str]] = {}
        for i, handle in enumerate(handles):
            next_handle = handles[i + 1] if i + 1 < len(handles) else None
            shadow_start = None
            if next_handle is not None:
                shadow_start = self._add_shadow_chain(
                    pages=pages,
                    start_from_index=i + 1,
                    n_pages=n_pages,
                    checkpoint_pages=checkpoint_pages,
                    actors=actors,
                    objects=objects,
                    rng=rng,
                    id_bank=id_bank,
                )
            text = self._render_page(
                page_number=i + 1,
                n_pages=n_pages,
                handle=handle,
                events=pages_events[i],
                checkpoint=(i + 1) if (i + 1) in checkpoint_pages else None,
                next_handle=next_handle,
                shadow_handle=shadow_start,
                state_after=states_after[i],
                checkpoint_row=checkpoint_rows.get(i + 1),
                actors=actors,
                rng=rng,
            )
            pages[handle] = {"text": text}

        return {
            "pages": pages,
            "start_handle": start_handle,
            "objects": objects,
            "actors": actors,
            "locations": locations,
            "_checkpoint_pages": checkpoint_pages,
            "_pages_events": pages_events,
            "_answer": answer_rows,
            "_difficulty": difficulty,
            "_actual_pages": n_pages,
        }

    def _simulate(
        self,
        pages_events: list[list[dict[str, Any]]],
        checkpoint_pages: list[int],
        objects: list[str],
        actors: list[str],
    ) -> tuple[list[list[Any]], list[dict[str, Any]], dict[int, list[Any]]]:
        owners: dict[str, str | None] = {o: None for o in objects}
        exists: dict[str, bool] = {o: False for o in objects}
        interval_counts: dict[str, int] = {a: 0 for a in actors}
        rows: list[list[Any]] = []
        states_after: list[dict[str, Any]] = []
        checkpoint_rows: dict[int, list[Any]] = {}

        for page_number, events in enumerate(pages_events, start=1):
            for e in events:
                obj = e["object"]
                if e["type"] == "CREATE":
                    owners[obj] = e["owner"]
                    exists[obj] = True
                elif e["type"] == "TRANSFER" and exists.get(obj, False):
                    owners[obj] = e["to"]
                    interval_counts[e["to"]] = interval_counts.get(e["to"], 0) + 1
                elif e["type"] == "DESTROY":
                    exists[obj] = False
                    owners[obj] = None
                elif e["type"] == "MOVE":
                    pass

            if page_number in checkpoint_pages:
                max_count = max(interval_counts.values()) if interval_counts else 0
                winners = [a for a, c in interval_counts.items() if c == max_count]
                actor = sorted(winners)[0]
                owned_count = sum(
                    1 for obj in objects
                    if exists.get(obj, False) and owners.get(obj) == actor
                )
                rows.append([f"CP{page_number}", actor, max_count, owned_count])
                checkpoint_rows[page_number] = rows[-1]
                interval_counts = {a: 0 for a in actors}
            states_after.append({
                "owners": dict(owners),
                "exists": dict(exists),
                "interval_counts": dict(interval_counts),
            })
        return rows, states_after, checkpoint_rows

    def _render_page(
        self,
        *,
        page_number: int,
        n_pages: int,
        handle: str,
        events: list[dict[str, Any]],
        checkpoint: int | None,
        next_handle: str | None,
        shadow_handle: str | None,
        state_after: dict[str, Any],
        checkpoint_row: list[Any] | None,
        actors: list[str],
        rng: random.Random,
    ) -> str:
        lines = [
            f"Slip {page_number}/{n_pages}; handle {handle}.",
        ]
        for e in events:
            lines.append(f"- {self._event_sentence(e, rng)}")
        if checkpoint is not None:
            lines.append(f"Checkpoint CP{checkpoint} closes here.")
        if next_handle is None:
            lines.append("Terminal slip: no next tab.")
        else:
            route_text, route_actor = self._route_instruction(
                checkpoint_row=checkpoint_row,
                state_after=state_after,
                rng=rng,
            )
            choices = self._route_choices(
                route_actor=route_actor,
                next_handle=next_handle,
                shadow_handle=shadow_handle or next_handle,
                actors=actors,
                rng=rng,
            )
            lines.append(route_text)
            lines.append("Tabs: " + "; ".join(choices) + ".")
        return "\n".join(lines)

    def _event_sentence(self, e: dict[str, Any], rng: random.Random) -> str:
        obj = e["object"]
        if e["type"] == "CREATE":
            return rng.choice([
                f"{e['owner']} opened live file `{obj}`.",
                f"New live file `{obj}` belongs to {e['owner']}.",
                f"{e['owner']} became the first holder of `{obj}`.",
            ])
        if e["type"] == "TRANSFER":
            return rng.choice([
                f"Custody of `{obj}` passed from {e['from']} to {e['to']}.",
                f"{e['to']} received `{obj}` from {e['from']}.",
                f"{e['from']} handed `{obj}` to {e['to']}.",
            ])
        if e["type"] == "MOVE":
            return rng.choice([
                f"`{obj}` moved to {e['to']}; holder stayed same.",
                f"`{obj}` was re-shelved at {e['to']}; owner unchanged.",
                f"Location for `{obj}` changed to {e['to']}; no custody change.",
            ])
        return rng.choice([
            f"`{obj}` was voided; it is not live.",
            f"`{obj}` was closed; nobody holds it.",
            f"`{obj}` left the live ledger.",
        ])

    def _route_instruction(
        self,
        *,
        checkpoint_row: list[Any] | None,
        state_after: dict[str, Any],
        rng: random.Random,
    ) -> tuple[str, str]:
        if checkpoint_row is not None:
            actor = str(checkpoint_row[1])
            return (
                "Route: use this checkpoint's winning actor.",
                actor,
            )
        live = [
            obj for obj, exists in state_after["exists"].items()
            if exists and state_after["owners"].get(obj)
        ]
        if live:
            obj = rng.choice(sorted(live))
            actor = str(state_after["owners"][obj])
            return (
                f"Route: use current holder of `{obj}`.",
                actor,
            )
        counts = state_after["interval_counts"]
        max_count = max(counts.values()) if counts else 0
        actor = sorted(a for a, c in counts.items() if c == max_count)[0]
        return (
            "Route: no live files; use current interval transfer leader.",
            actor,
        )

    def _route_choices(
        self,
        *,
        route_actor: str,
        next_handle: str,
        shadow_handle: str,
        actors: list[str],
        rng: random.Random,
    ) -> list[str]:
        decoys = [a for a in actors if a != route_actor]
        decoy_actor = rng.choice(decoys) if decoys else route_actor
        pairs = [
            (route_actor, next_handle),
            (decoy_actor, shadow_handle),
        ]
        rng.shuffle(pairs)
        tabs = rng.sample(TAB_WORDS, 2)
        return [
            f"{actor}/{tab}->{handle}"
            for tab, (actor, handle) in zip(tabs, pairs)
        ]

    def _add_shadow_chain(
        self,
        *,
        pages: dict[str, dict[str, str]],
        start_from_index: int,
        n_pages: int,
        checkpoint_pages: list[int],
        actors: list[str],
        objects: list[str],
        rng: random.Random,
        id_bank: NaturalIdBank,
    ) -> str:
        levels: list[tuple[str, str]] = [
            (_new_handle(id_bank), _new_handle(id_bank))
            for _ in range(start_from_index, n_pages)
        ]
        for offset, page_index in enumerate(range(start_from_index, n_pages)):
            page_number = page_index + 1
            next_pair = levels[offset + 1] if offset + 1 < len(levels) else None
            for variant, handle in enumerate(levels[offset]):
                pages[handle] = {
                    "text": self._render_shadow_page(
                        page_number=page_number,
                        n_pages=n_pages,
                        handle=handle,
                        checkpoint=page_number if page_number in checkpoint_pages else None,
                        next_pair=next_pair,
                        actors=actors,
                        objects=objects,
                        rng=rng,
                    )
                }
        return levels[0][0]

    def _render_shadow_page(
        self,
        *,
        page_number: int,
        n_pages: int,
        handle: str,
        checkpoint: int | None,
        next_pair: tuple[str, str] | None,
        actors: list[str],
        objects: list[str],
        rng: random.Random,
    ) -> str:
        actor_a, actor_b = rng.sample(actors, 2)
        obj = rng.choice(objects)
        lines = [
            f"Slip {page_number}/{n_pages}; handle {handle}.",
            f"- {actor_a} reviewed `{obj}`; no holder change.",
            f"- {actor_b} copied a side receipt for `{obj}`.",
        ]
        if checkpoint is not None:
            lines.append(
                f"Checkpoint CP{checkpoint} closes here on this branch."
            )
        if next_pair is None:
            lines.append("Terminal slip: this branch has no next tab.")
        else:
            choices = [
                f"{actor_a}/{rng.choice(TAB_WORDS)}->{next_pair[0]}",
                f"{actor_b}/{rng.choice(TAB_WORDS)}->{next_pair[1]}",
            ]
            rng.shuffle(choices)
            lines.append(
                f"Route: use who copied `{obj}` on this branch."
            )
            lines.append("Tabs: " + "; ".join(choices) + ".")
        return "\n".join(lines)

    def generate_query(
        self,
        template: QueryTemplate,
        state: dict,
        target_turns: int,
        rng: random.Random,
    ) -> tuple[str, Any, int, int]:
        rich = self.generate_query_rich(template, state, target_turns, rng)
        return (
            rich["query_text"],
            rich["expected_answer"],
            rich["actual_depth"],
            rich["actual_breadth"],
        )

    def generate_query_rich(
        self,
        template: QueryTemplate,
        state: dict,
        target_turns: int,
        rng: random.Random,
    ) -> dict:
        answer = state["_answer"]
        n_pages = int(state.get("_actual_pages", len(state["pages"])))
        cps = list(state["_checkpoint_pages"])
        query = (
            "Follow the opaque cursor ledger. The Python variable START_HANDLE "
            "is already seeded; call observe(START_HANDLE) without quotes to "
            "return the first page string. Each nonterminal page gives a route "
            "rule and two candidate tabs; choose the tab by interpreting the "
            "page and your current ledger state, then call observe(handle) on "
            "that tab's handle. If you need to see a page or state next turn, "
            "write it to context_window yourself. Process every ledger entry in "
            "order, keeping context_window compact by overwriting stale raw "
            "pages with durable state. At each checkpoint "
            f"{[f'CP{x}' for x in cps]}, report [checkpoint, actor, "
            "interval_transfer_count, owned_count_at_checkpoint], where the "
            "checkpoint field is the exact string id with the CP prefix, such "
            "as \"CP1\" or \"CP10\"; do not submit a bare number like 1 or 10. "
            "The final answer must be a JSON-style list of rows like "
            "[[\"CP1\", \"Alice\", 2, 1], ...]. The "
            "actor is the alphabetically earliest actor tied for the most "
            "TRANSFER receipts since the previous checkpoint. "
            "interval_transfer_count is that winning actor's receipt count, "
            "not the total number of transfers in the interval. "
            "owned_count_at_checkpoint is the number of live objects owned by "
            "that same reported actor at the checkpoint, not the total number "
            "of live objects. TRANSFER receipts are counted for the receiver "
            "(the actor the object moves to), while created/opened/first-holder "
            "objects are not transfer receipts. Destroyed, closed, voided, or "
            "left-ledger objects are not live and count for nobody unless a "
            "later page creates/opens them again. Ownership and existence "
            "carry forward across pages; interval transfer counts reset after "
            "each checkpoint. Submit JSON rows in checkpoint order after the "
            "terminal page."
        )
        return {
            "query_text": query,
            "expected_answer": json.dumps(answer, separators=(",", ":")),
            "actual_depth": n_pages,
            "actual_breadth": len(state["objects"]),
            "target_entities": [row[1] for row in answer],
            "parameter_refs": [],
            "rules": [],
        }


def make_example(seed: int, difficulty: int) -> TrainingExample:
    rng = random.Random(seed)
    gen = AdaptiveCursorWorld()
    template = gen.get_query_templates()[0]
    state = gen.generate_state(depth=3, breadth=4, rng=rng, difficulty=difficulty)
    rich = gen.generate_query_rich(template, state, target_turns=12, rng=rng)
    clean_state = {k: v for k, v in state.items() if not k.startswith("_")}
    return TrainingExample(
        example_id=rng.randint(0, 2**31 - 1),
        world_type=gen.world_type,
        system_prompt=gen.get_system_prompt(),
        user_query=rich["query_text"],
        state=clean_state,
        optimal_turns=int(state.get("_actual_pages", len(clean_state["pages"]))) + 1,
        expected_answer=rich["expected_answer"],
        answer_type=template.answer_type,
        difficulty=difficulty,
        depth=rich["actual_depth"],
        breadth=rich["actual_breadth"],
        query_template=template.name,
        target_entities=list(rich.get("target_entities", [])),
        parameter_refs=[],
        rules=[],
    )
