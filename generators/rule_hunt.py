"""rule_hunt — hypothesis-refinement world.

The hidden rule is a parse tree (boolean function) over entity attributes.
The model probes entities (tools: get_entity, test) to triangulate the
rule, then submits the parse tree. Reward is functional equivalence on a
held-out test set the model never sees.

Designed to force *editing* of the scratchpad: the model that wins
maintains a single ``current_hypothesis`` slot in ``context_window`` and
*rewrites* it as evidence narrows — appending every (id, attrs, label)
triple suffocates the context.

State schema
------------
::

    {
        "entities": {                 # probe set, exposed via tools
            "e1k": {"color": "red", "size": 42, ...},
            ...
        },
        "labels": {                   # used by `test()`
            "e1k": True, ...
        },
        "rule": {...},                # ground-truth parse tree (stripped at runtime)
        "held_out": [                 # rubric-only; stripped before sandbox sees it
            {"attrs": {...}, "label": True}, ...
        ],
        "attribute_schema": {...},    # exposed as ATTRIBUTE_SCHEMA
        "n_atoms": int,
    }

The env's ``setup_state`` strips ``rule`` and ``held_out`` from
``world_state`` before it is shipped to the sandbox, so the model cannot
read them via ``_world_state``. They survive on ``state`` for the rubric.
"""

from __future__ import annotations

import json
import random
import string
from typing import Any

from .base import QueryTemplate, WorldGenerator


# ---------------------------------------------------------------------------
# Schema (fixed across all rule_hunt examples for v1)
# ---------------------------------------------------------------------------

DEFAULT_SCHEMA: dict = {
    "color": {
        "type": "categorical",
        "values": ["red", "blue", "green", "yellow", "purple"],
    },
    "size": {"type": "numeric", "range": [1, 100]},
    "weight": {"type": "numeric", "range": [1, 50]},
    "category": {
        "type": "categorical",
        "values": ["alpha", "beta", "gamma", "delta"],
    },
    "tier": {"type": "numeric", "range": [1, 5]},
    "tags": {
        "type": "list",
        "values": ["industrial", "organic", "fragile", "valuable", "common", "rare"],
    },
}


RULE_HUNT_TEMPLATES = [
    QueryTemplate(
        name="find_rule",
        description="Find the hidden rule that classifies entities.",
        min_turns=3,
        max_turns=15,
        answer_type="str",
    ),
]


# ---------------------------------------------------------------------------
# DSL evaluator (pure — used by generator, solver, and rubric)
# ---------------------------------------------------------------------------


def eval_rule(rule: dict, entity: dict) -> bool:
    """Evaluate `rule` against `entity` (a dict of attr -> value).

    Raises ValueError on malformed rules. Returns False for atoms whose
    attribute is missing or whose value type doesn't match the operator.
    """
    if not isinstance(rule, dict):
        raise ValueError(f"Rule must be a dict, got {type(rule).__name__}")
    op = rule.get("op")
    if op == "AND":
        return all(eval_rule(a, entity) for a in rule.get("args", []))
    if op == "OR":
        return any(eval_rule(a, entity) for a in rule.get("args", []))
    if op == "NOT":
        args = rule.get("args", [])
        return not eval_rule(args[0], entity) if args else False
    # Atom
    attr = rule.get("attr")
    val = rule.get("value")
    actual = entity.get(attr)
    if op == "==":
        return actual == val
    if op == "!=":
        return actual != val
    if op == "<":
        return actual is not None and actual < val
    if op == ">":
        return actual is not None and actual > val
    if op == "<=":
        return actual is not None and actual <= val
    if op == ">=":
        return actual is not None and actual >= val
    if op == "contains":
        return isinstance(actual, list) and val in actual
    raise ValueError(f"Unknown rule op: {op!r}")


def rule_atom_count(rule: dict) -> int:
    op = rule.get("op")
    if op in ("AND", "OR"):
        return sum(rule_atom_count(a) for a in rule.get("args", []))
    if op == "NOT":
        args = rule.get("args", [])
        return rule_atom_count(args[0]) if args else 0
    return 1


# ---------------------------------------------------------------------------
# Atom + rule sampling
# ---------------------------------------------------------------------------


def _sample_atom(schema: dict, rng: random.Random) -> dict:
    """Sample a single atomic predicate."""
    attr = rng.choice(list(schema.keys()))
    spec = schema[attr]
    t = spec["type"]
    if t == "categorical":
        op = rng.choice(["==", "!="])
        value = rng.choice(spec["values"])
        return {"op": op, "attr": attr, "value": value}
    if t == "numeric":
        lo, hi = spec["range"]
        # Threshold inside (lo, hi) so the predicate splits the population
        if hi - lo <= 2:
            value = rng.randint(lo, hi)
        else:
            value = rng.randint(lo + 1, hi - 1)
        op = rng.choice(["<", ">", "<=", ">="])
        return {"op": op, "attr": attr, "value": value}
    if t == "list":
        return {"op": "contains", "attr": attr, "value": rng.choice(spec["values"])}
    raise ValueError(f"Unknown attr type: {t!r}")


def _sample_rule(schema: dict, n_atoms: int, rng: random.Random) -> dict:
    """Sample a rule with roughly `n_atoms` atomic predicates."""
    if n_atoms <= 1:
        return _sample_atom(schema, rng)
    if n_atoms == 2:
        op = rng.choice(["AND", "OR"])
        atoms = [_sample_atom(schema, rng) for _ in range(2)]
        # Force distinct attrs so the two atoms aren't trivially redundant
        retries = 0
        while atoms[0]["attr"] == atoms[1]["attr"] and retries < 10:
            atoms[1] = _sample_atom(schema, rng)
            retries += 1
        return {"op": op, "args": atoms}
    if n_atoms == 3:
        op = rng.choice(["AND", "OR"])
        atoms: list[dict] = []
        used: set[str] = set()
        retries = 0
        while len(atoms) < 3 and retries < 30:
            a = _sample_atom(schema, rng)
            retries += 1
            if a["attr"] in used:
                continue
            atoms.append(a)
            used.add(a["attr"])
        if len(atoms) < 3:  # ran out of attrs to use; fill with any
            while len(atoms) < 3:
                atoms.append(_sample_atom(schema, rng))
        return {"op": op, "args": atoms}
    # n_atoms >= 4: nested combinator
    outer = rng.choice(["AND", "OR"])
    half = n_atoms // 2
    return {
        "op": outer,
        "args": [
            _sample_rule(schema, half, rng),
            _sample_rule(schema, n_atoms - half, rng),
        ],
    }


# ---------------------------------------------------------------------------
# Entity sampling
# ---------------------------------------------------------------------------


def _sample_entity_attrs(schema: dict, rng: random.Random) -> dict:
    e: dict = {}
    for attr, spec in schema.items():
        t = spec["type"]
        if t == "categorical":
            e[attr] = rng.choice(spec["values"])
        elif t == "numeric":
            lo, hi = spec["range"]
            e[attr] = rng.randint(lo, hi)
        elif t == "list":
            n_tags = rng.randint(0, 3)
            e[attr] = rng.sample(spec["values"], n_tags) if n_tags else []
    return e


def _generate_entity_id(rng: random.Random) -> str:
    """3-char ID matching the project style (e.g. 'e1k')."""
    return (
        rng.choice(string.ascii_lowercase)
        + str(rng.randint(0, 9))
        + rng.choice(string.ascii_lowercase)
    )


# ---------------------------------------------------------------------------
# WorldGenerator
# ---------------------------------------------------------------------------


# Difficulty axis: maps the global difficulty (1–5) to the family's primary
# scratchpad-pressure knob — number of atoms in the hidden rule. More atoms
# = more hypothesis revisions before convergence.
DIFFICULTY_AXIS: dict[int, dict] = {
    1: {"n_atoms": 1},
    2: {"n_atoms": 2},
    3: {"n_atoms": 3},
    4: {"n_atoms": 4},
    5: {"n_atoms": 5},
}


def _depth_to_n_atoms(depth: int) -> int:
    """Fallback depth → n_atoms mapping when ``difficulty`` is not provided."""
    if depth <= 1:
        return 1
    if depth <= 2:
        return 2
    if depth <= 3:
        return 3
    if depth <= 5:
        return 4
    return 5


_PROBE_POOL_SIZE = 80
_HELD_OUT_SIZE = 20


class RuleHuntWorld(WorldGenerator):
    """Hypothesis-refinement world: probe entities, find the hidden rule."""

    world_type = "rule_hunt"

    def get_query_templates(self) -> list[QueryTemplate]:
        return RULE_HUNT_TEMPLATES

    def get_system_prompt(self) -> str:
        return _RULE_HUNT_SYSTEM_PROMPT_PRELUDE

    def generate_state(
        self,
        depth: int,
        breadth: int,  # unused; probe-pool size is fixed for now
        rng: Any,
        *,
        difficulty: int | None = None,
    ) -> dict:
        # Difficulty (1–5) is the primary axis; falls back to depth-mapping
        # for legacy callers that don't pass it.
        if difficulty is not None:
            d = max(1, min(5, int(difficulty)))
            n_atoms = DIFFICULTY_AXIS[d]["n_atoms"]
        else:
            n_atoms = _depth_to_n_atoms(depth)
        schema = DEFAULT_SCHEMA

        # Try multiple rules; reject ones that are too unbalanced.
        for _attempt in range(80):
            rule = _sample_rule(schema, n_atoms, rng)

            pool: dict[str, dict] = {}
            target_pool = (_PROBE_POOL_SIZE + _HELD_OUT_SIZE) * 4
            tries = 0
            while len(pool) < target_pool and tries < target_pool * 4:
                tries += 1
                eid = _generate_entity_id(rng)
                if eid in pool:
                    continue
                pool[eid] = _sample_entity_attrs(schema, rng)

            labeled = {eid: eval_rule(rule, attrs) for eid, attrs in pool.items()}
            pos_ids = [eid for eid, lab in labeled.items() if lab]
            neg_ids = [eid for eid, lab in labeled.items() if not lab]

            min_per_class = max(
                _PROBE_POOL_SIZE // 4 + _HELD_OUT_SIZE // 2,
                _HELD_OUT_SIZE,
            )
            if len(pos_ids) < min_per_class or len(neg_ids) < min_per_class:
                continue

            # Build balanced probe + held-out splits.
            rng.shuffle(pos_ids)
            rng.shuffle(neg_ids)
            n_probe_pos = _PROBE_POOL_SIZE // 2
            n_probe_neg = _PROBE_POOL_SIZE - n_probe_pos
            n_held_pos = _HELD_OUT_SIZE // 2
            n_held_neg = _HELD_OUT_SIZE - n_held_pos
            if (
                len(pos_ids) < n_probe_pos + n_held_pos
                or len(neg_ids) < n_probe_neg + n_held_neg
            ):
                continue

            probe_ids = pos_ids[:n_probe_pos] + neg_ids[:n_probe_neg]
            held_ids = (
                pos_ids[n_probe_pos : n_probe_pos + n_held_pos]
                + neg_ids[n_probe_neg : n_probe_neg + n_held_neg]
            )
            rng.shuffle(probe_ids)
            rng.shuffle(held_ids)

            entities = {eid: pool[eid] for eid in probe_ids}
            labels = {eid: labeled[eid] for eid in probe_ids}
            held_out = [
                {"attrs": pool[eid], "label": labeled[eid]} for eid in held_ids
            ]

            return {
                "entities": entities,
                "labels": labels,
                "rule": rule,
                "held_out": held_out,
                "attribute_schema": schema,
                "n_atoms": rule_atom_count(rule),
            }

        raise RuntimeError(
            "rule_hunt: failed to sample a sufficiently balanced rule "
            f"after retries (n_atoms={n_atoms})"
        )

    def generate_query(
        self,
        template: QueryTemplate,
        state: dict,
        target_turns: int,
        rng: Any,
    ) -> tuple[str, Any, int, int]:
        rule = state["rule"]
        query_text = (
            "Find the hidden rule that classifies these entities. Use "
            "get_entity(id) to inspect attributes and test(id) to read the "
            "rule's label. ENTITY_IDS and ATTRIBUTE_SCHEMA are pre-seeded in "
            "the kernel. Submit the parse tree via submit_answer(rule_dict). "
            "Reward is 1 iff your rule produces the same labels as the "
            "hidden rule on a held-out test set."
        )
        # Use a stable JSON serialization for the expected_answer string.
        expected_answer = json.dumps(rule, sort_keys=True)
        actual_depth = state.get("n_atoms", rule_atom_count(rule))
        actual_breadth = len(state.get("entities", {}))
        return query_text, expected_answer, actual_depth, actual_breadth

    def generate_query_rich(
        self,
        template: QueryTemplate,
        state: dict,
        target_turns: int,
        rng: Any,
    ) -> dict:
        q, ans, d, b = self.generate_query(template, state, target_turns, rng)
        return {
            "query_text": q,
            "expected_answer": ans,
            "actual_depth": d,
            "actual_breadth": b,
            "target_entities": [],
            "parameter_refs": [],
            "rules": [],
        }


# Shown by `WorldGenerator.get_system_prompt()`; the env may use its own
# system prompt template instead, but this is here for parity with the
# other generators' interfaces.
_RULE_HUNT_SYSTEM_PROMPT_PRELUDE = (
    "rule_hunt world: probe entities to find the hidden boolean rule. "
    "Tools: get_entity(id), test(id). Submit a parse tree via "
    "submit_answer(rule_dict)."
)
