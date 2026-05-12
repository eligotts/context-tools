"""ReferenceSolver — the optimal policy, run against each generated episode.

Purpose
-------
1. **Verifiability.** After ``generate_example`` picks a state + computes the
   expected answer, we need an *independent* derivation of that answer that
   uses only the real tool signatures. If the solver's answer matches, the
   example is provably solvable-from-tools. If they diverge, the generator
   has a bug.
2. **Authoritative ``optimal_turns``.** Each solver run counts the minimum
   number of parallel-turn rounds needed (under the env's
   ``tool_call_budget_per_turn`` cap, default 5). Replaces the old
   ``optimal_turns = actual_depth + 1`` heuristic with a measured number.

Scope
-----
Active families: ``rule_hunt``, ``corpus_dive``, ``timeline_track``,
``detective``, ``maze_walk``. All five have solvers below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .base import Question


# =============================================================================
# Result types
# =============================================================================


@dataclass
class ToolCall:
    tool: str
    args: dict
    result: Any


@dataclass
class SolverTurn:
    """One parallel batch of tool calls."""

    calls: list[ToolCall] = field(default_factory=list)
    # Tool-result keys (tool_name, arg-tuple) this turn depends on *being
    # available in memory* — i.e. they were fetched earlier and we still need
    # them. Used to build the retention schedule.
    uses_prior: list[tuple[str, tuple]] = field(default_factory=list)


@dataclass
class SolverResult:
    answer: str
    turns: list[SolverTurn]

    @property
    def turn_count(self) -> int:
        """Tool-call turns + 1 final answer turn."""
        return len(self.turns) + 1

    @property
    def total_tool_calls(self) -> int:
        return sum(len(t.calls) for t in self.turns)

    def retention_schedule(self) -> list[set[tuple[str, tuple]]]:
        """Per turn, the set of prior tool-result keys still needed."""
        return [set(t.uses_prior) for t in self.turns]


class SolverError(Exception):
    """Raised when a template is not solvable-from-tools or inputs are malformed."""


# =============================================================================
# Solver dispatch
# =============================================================================


class ReferenceSolver:
    """Run the optimal tool-use policy for a question against a world state."""

    def __init__(self, world_type: str):
        self.world_type = world_type

    def solve(
        self,
        state: dict,
        question: Question,
        rules: list[dict] | None = None,
    ) -> SolverResult:
        world = self.world_type
        tpl = question.template_name
        handler_name = f"_solve_{world}__{tpl}"
        handler: Callable | None = getattr(self, handler_name, None)
        if handler is None:
            raise SolverError(
                f"No solver for world={world!r} template={tpl!r}."
            )
        try:
            return handler(state, question, rules=rules or [])
        except TypeError:
            return handler(state, question)

    # -------------------------------------------------------------------------
    # rule_hunt
    # -------------------------------------------------------------------------

    def _solve_rule_hunt__find_rule(
        self, state: dict, q: Question
    ) -> SolverResult:
        """rule_hunt has no canonical optimal probe sequence — only a hidden
        rule to triangulate. The solver echoes the rule (so the
        ``expected_answer`` match in ``dataset.py`` succeeds) and reports a
        rough turn-count lower bound based on rule complexity.
        """
        import json as _json

        rule = state.get("rule")
        if rule is None:
            raise SolverError("rule_hunt: state missing 'rule' field")
        answer = _json.dumps(rule, sort_keys=True)
        n_atoms = int(state.get("n_atoms", 1) or 1)
        n_turns = max(3, 2 + n_atoms)
        turns = [SolverTurn() for _ in range(n_turns)]
        return SolverResult(answer=answer, turns=turns)

    # -------------------------------------------------------------------------
    # corpus_dive
    # -------------------------------------------------------------------------

    def _solve_corpus_dive__count_under_path(
        self, state: dict, q: Question
    ) -> SolverResult:
        return self._solve_corpus_dive_aggregate(state, q, kind="count")

    def _solve_corpus_dive__sum_values_under_path(
        self, state: dict, q: Question
    ) -> SolverResult:
        return self._solve_corpus_dive_aggregate(state, q, kind="sum_value")

    def _solve_corpus_dive_aggregate(
        self, state: dict, q: Question, kind: str
    ) -> SolverResult:
        """Walk the indicated subtree, evaluate predicate per node, aggregate.

        Predicate + subtree path are stashed by the generator under
        ``state["_question_meta"]``. One ``list_keys`` + one ``read_node``
        per node visited (the model can't know a node is a leaf without
        ``list_keys``, so we count both).
        """
        from .corpus_dive import eval_predicate, subtree_paths

        meta = state.get("_question_meta") or {}
        pred = meta.get("predicate")
        sub_path = meta.get("subtree_path")
        tree = state.get("tree") or {}
        if not pred or not sub_path or not tree:
            raise SolverError(
                "corpus_dive: missing _question_meta (predicate / subtree_path)"
            )

        sub_paths = subtree_paths(tree, sub_path)
        all_calls: list[ToolCall] = []
        running = 0
        for p in sub_paths:
            node = tree[p]
            attrs = node.get("attrs", {})
            children = list(node.get("children", []))
            all_calls.append(ToolCall("list_keys", {"path": p}, children))
            all_calls.append(ToolCall("read_node", {"path": p}, dict(attrs)))
            if eval_predicate(pred, attrs):
                if kind == "count":
                    running += 1
                elif kind == "sum_value":
                    running += int(attrs.get("value", 0))
        return SolverResult(
            answer=str(running),
            turns=_chunk_calls_into_turns(all_calls),
        )

    # -------------------------------------------------------------------------
    # timeline_track
    # -------------------------------------------------------------------------

    def _solve_timeline_track__owner_at_time(
        self, state: dict, q: Question
    ) -> SolverResult:
        return self._solve_timeline_track_snapshot(state, q, kind="owner")

    def _solve_timeline_track__count_owned_at_time(
        self, state: dict, q: Question
    ) -> SolverResult:
        return self._solve_timeline_track_snapshot(state, q, kind="count")

    def _solve_timeline_track__most_active_actor(
        self, state: dict, q: Question
    ) -> SolverResult:
        return self._solve_timeline_track_snapshot(state, q, kind="most_active")

    def _solve_timeline_track__owned_transfer_top_k_at_time(
        self, state: dict, q: Question
    ) -> SolverResult:
        return self._solve_timeline_track_snapshot(state, q, kind="owned_transfer_top_k")

    def _solve_timeline_track__checkpoint_actor_audit(
        self, state: dict, q: Question
    ) -> SolverResult:
        return self._solve_timeline_track_snapshot(state, q, kind="checkpoint_actor_audit")

    def _solve_timeline_track_snapshot(
        self, state: dict, q: Question, kind: str
    ) -> SolverResult:
        """Replay ``events[:T]`` in chunks of 5 (mirrors the env's
        ``read_events`` per-call cap) and build the optimal call list."""
        meta = state.get("_question_meta") or {}
        T = int(meta.get("time", 0) or 0)
        target_obj = meta.get("object")
        target_actor = meta.get("actor")
        events: list[dict] = state.get("events") or []
        objects: list[str] = state.get("objects") or []

        snap = {
            o: {"owner": None, "location": None, "exists": False}
            for o in objects
        }
        receiver_counts = {a: 0 for a in state.get("actors") or []}
        transfer_counts = {o: 0 for o in objects}
        all_calls: list[ToolCall] = []
        chunk = 5
        T_eff = max(0, min(T, len(events)))
        for start in range(0, T_eff, chunk):
            end = min(start + chunk, T_eff)
            block = list(events[start:end])
            all_calls.append(
                ToolCall("read_events", {"start": start, "end": end}, block)
            )
            for e in block:
                o = e.get("object")
                if o not in snap:
                    continue
                t = e.get("type")
                if t == "CREATE":
                    snap[o] = {
                        "owner": e.get("owner"),
                        "location": e.get("location"),
                        "exists": True,
                    }
                elif t == "TRANSFER" and snap[o]["exists"]:
                    snap[o]["owner"] = e.get("to")
                    if e.get("to") in receiver_counts:
                        receiver_counts[e.get("to")] += 1
                    transfer_counts[o] += 1
                elif t == "MOVE" and snap[o]["exists"]:
                    snap[o]["location"] = e.get("to")
                elif t == "DESTROY":
                    snap[o]["exists"] = False
                elif t == "TRANSFER":
                    if e.get("to") in receiver_counts:
                        receiver_counts[e.get("to")] += 1
                    if o in transfer_counts:
                        transfer_counts[o] += 1

        if kind == "owner":
            if target_obj is None or target_obj not in snap:
                raise SolverError("owner_at_time: missing target object")
            rec = snap[target_obj]
            answer = (
                str(rec["owner"])
                if rec["exists"] and rec["owner"]
                else "none"
            )
        elif kind == "count":
            if target_actor is None:
                raise SolverError("count_owned_at_time: missing target actor")
            answer = str(
                sum(
                    1 for o in objects
                    if snap[o]["exists"] and snap[o]["owner"] == target_actor
                )
            )
        elif kind == "most_active":
            if not receiver_counts:
                raise SolverError("most_active_actor: missing actors")
            answer = sorted(receiver_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        elif kind == "owned_transfer_top_k":
            import json as _json

            if target_actor is None:
                raise SolverError("owned_transfer_top_k: missing target actor")
            k = int(meta.get("k", 3) or 3)
            owned = [
                o for o in objects
                if snap[o]["exists"] and snap[o]["owner"] == target_actor
            ]
            ranked = sorted(
                ((o, transfer_counts[o]) for o in owned),
                key=lambda item: (-item[1], item[0]),
            )[:k]
            answer = _json.dumps([[o, c] for o, c in ranked], separators=(",", ":"))
        elif kind == "checkpoint_actor_audit":
            import json as _json

            from .timeline_track import checkpoint_actor_audit

            checkpoints = list(meta.get("checkpoints") or [])
            if not checkpoints:
                raise SolverError("checkpoint_actor_audit: missing checkpoints")
            answer = _json.dumps(
                checkpoint_actor_audit(
                    events=events,
                    checkpoints=checkpoints,
                    objects=objects,
                    actors=list(state.get("actors") or []),
                ),
                separators=(",", ":"),
            )
        else:
            raise SolverError(f"timeline_track: unknown kind {kind!r}")

        return SolverResult(
            answer=answer, turns=_chunk_calls_into_turns(all_calls)
        )

    # -------------------------------------------------------------------------
    # detective
    # -------------------------------------------------------------------------

    def _solve_detective__find_unique(
        self, state: dict, q: Question
    ) -> SolverResult:
        """Compute the unique satisfying entity via constraint evaluation.

        Optimal cost: one ``get_entity`` per entity. With ``N`` entities and
        budget 5/turn, that's ``ceil(N/5)`` turns. The actual winning play
        does this in code with a kernel-side ``candidates`` set; the solver
        only records the call count.
        """
        from .corpus_dive import eval_predicate

        constraints = state.get("constraints") or []
        entities = state.get("entities") or {}
        if not constraints or not entities:
            raise SolverError("detective: missing constraints or entities")

        satisfying = [
            eid for eid, attrs in entities.items()
            if all(eval_predicate(c, attrs) for c in constraints)
        ]
        if len(satisfying) != 1:
            raise SolverError(
                f"detective: expected exactly 1 satisfying entity, got {len(satisfying)}"
            )
        answer = satisfying[0]

        all_calls: list[ToolCall] = [
            ToolCall("get_entity", {"entity_id": eid}, dict(attrs))
            for eid, attrs in entities.items()
        ]
        return SolverResult(
            answer=str(answer),
            turns=_chunk_calls_into_turns(all_calls),
        )

    def _solve_detective__count_matching(
        self, state: dict, q: Question
    ) -> SolverResult:
        """Count entities satisfying all constraints.

        Forces a full scan — there is no early-exit because every entity
        affects the final count. With ``N`` entities and budget 5/turn this
        is ``ceil(N/5)`` turns of get_entity calls plus 1 submission turn.
        """
        from .corpus_dive import eval_predicate

        constraints = state.get("constraints") or []
        entities = state.get("entities") or {}
        if not constraints or not entities:
            raise SolverError("detective: missing constraints or entities")

        satisfying = [
            eid for eid, attrs in entities.items()
            if all(eval_predicate(c, attrs) for c in constraints)
        ]
        answer = str(len(satisfying))

        all_calls: list[ToolCall] = [
            ToolCall("get_entity", {"entity_id": eid}, dict(attrs))
            for eid, attrs in entities.items()
        ]
        return SolverResult(
            answer=answer,
            turns=_chunk_calls_into_turns(all_calls),
        )

    def _solve_detective__top_k_by_region(
        self, state: dict, q: Question
    ) -> SolverResult:
        """Solve the multi-bucket top-K leaderboard task with a full scan."""
        import json as _json

        from .corpus_dive import eval_predicate

        constraints = state.get("constraints") or []
        entities = state.get("entities") or {}
        meta = state.get("_top_k_meta") or {}
        regions = list(meta.get("target_regions") or [])
        k = int(meta.get("k", 0) or 0)
        if not constraints or not entities or not regions or k <= 0:
            raise SolverError("detective top_k: missing constraints/entities/meta")

        answer: list[list] = []
        for region in sorted(regions):
            eligible = [
                (int(attrs["price"]), eid)
                for eid, attrs in entities.items()
                if attrs.get("region") == region
                and all(eval_predicate(c, attrs) for c in constraints)
            ]
            ranked = sorted(eligible, key=lambda x: (-x[0], x[1]))[:k]
            answer.append([region, [[eid, price] for price, eid in ranked]])

        all_calls: list[ToolCall] = [
            ToolCall("get_entity", {"entity_id": eid}, dict(attrs))
            for eid, attrs in entities.items()
        ]
        return SolverResult(
            answer=_json.dumps(answer, separators=(",", ":")),
            turns=_chunk_calls_into_turns(all_calls),
        )

    # -------------------------------------------------------------------------
    # maze_walk
    # -------------------------------------------------------------------------

    def _solve_maze_walk__find_goal_secret(
        self, state: dict, q: Question
    ) -> SolverResult:
        """Optimal play: BFS from ``start`` until the goal is found.

        At minimum the model needs ``L`` ``move()`` calls (``L`` =
        shortest-path length). A pure BFS that finds the goal at depth
        ``L`` will visit every node at distance ``< L`` plus a fraction
        of those at distance ``L`` — bounded by the L-ball around start.
        """
        from collections import deque

        graph = state.get("graph") or {}
        start = state.get("start")
        goal = state.get("goal")
        secrets = state.get("node_secrets") or {}
        if not graph or start is None or goal is None or goal not in secrets:
            raise SolverError("maze_walk: missing graph/start/goal/secret")

        answer = secrets[goal]

        dist = {start: 0}
        order = [start]
        frontier = deque([start])
        while frontier:
            n = frontier.popleft()
            if n == goal:
                break
            for nbr in graph.get(n, []):
                if nbr not in dist:
                    dist[nbr] = dist[n] + 1
                    order.append(nbr)
                    frontier.append(nbr)

        all_calls: list[ToolCall] = [
            ToolCall(
                "move",
                {"target": n},
                {"at": n, "is_goal": (n == goal), "secret": secrets[n]},
            )
            for n in order[1:]  # skip start (free; pre-seeded position)
        ]
        return SolverResult(
            answer=str(answer),
            turns=_chunk_calls_into_turns(all_calls),
        )


# =============================================================================
# Helper: chunk a flat call list into per-turn batches of <= budget calls
# =============================================================================


def _chunk_calls_into_turns(
    calls: list[ToolCall], budget: int = 5
) -> list[SolverTurn]:
    """``len(turns) == ceil(len(calls) / budget)``. Default budget mirrors
    the env's per-turn ``tool_call_budget_per_turn``."""
    if not calls:
        return []
    chunks: list[SolverTurn] = []
    for i in range(0, len(calls), budget):
        chunks.append(SolverTurn(calls=list(calls[i : i + budget])))
    return chunks
