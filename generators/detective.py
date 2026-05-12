"""detective — set-shrinking / candidate-elimination world.

A pool of ~25–65 entities (each with attributes) and 3–7 conjunctive
constraints. Exactly one entity satisfies all constraints; the rest
violate at least one. The model must find the unique answer.

Designed to force a *shrinking* candidate set in the scratchpad. The
scratchpad-winning play maintains a single ``candidates: set[str]``
entry that gets *rewritten* (overwritten with set difference) as each
constraint is applied — distinct from corpus_dive (matches list grows),
rule_hunt (theory edits in place), and timeline_track (state values
mutate but cardinality is fixed). Append-style play loses by inflating
context with every entity check.

State schema
------------
::

    {
        "entities": {            # ~25–65 entities, exposed via tools
            "e1k": {"color": "red", "kind": "alpha", ...},
            ...
        },
        "constraints": [         # exposed as ``CONSTRAINTS`` in kernel
            {"op": "==", "attr": "color", "value": "red"},
            {"op": ">", "attr": "size", "value": 50},
            ...
        ],
        "attribute_schema": {...},
        "n_entities": int,
        "n_constraints": int,
        "_answer_id": str,       # generator-internal; stripped before sandbox
    }

The constraints themselves are NOT secret — they are the question. They
are exposed both as natural-language text (in ``query_text``) and as
structured Python data (``CONSTRAINTS`` pre-seeded in the kernel) so the
model can iterate over them programmatically.
"""

from __future__ import annotations

import json
import random
import string
from typing import Any

from .base import QueryTemplate, WorldGenerator
from .corpus_dive import eval_predicate  # single-atom predicate eval; reused


# ---------------------------------------------------------------------------
# Schema (shared style with rule_hunt / corpus_dive)
# ---------------------------------------------------------------------------

DEFAULT_SCHEMA: dict = {
    "color": {
        "type": "categorical",
        "values": ["red", "blue", "green", "yellow", "purple"],
    },
    "kind": {
        "type": "categorical",
        "values": ["alpha", "beta", "gamma", "delta"],
    },
    "region": {
        "type": "categorical",
        "values": ["north", "south", "east", "west", "central"],
    },
    "status": {
        "type": "categorical",
        "values": ["active", "dormant", "archived"],
    },
    "size": {"type": "numeric", "range": [1, 100]},
    "weight": {"type": "numeric", "range": [1, 50]},
    "tier": {"type": "numeric", "range": [1, 5]},
    "price": {"type": "numeric", "range": [10, 500]},
    "tags": {
        "type": "list",
        "values": [
            "industrial", "organic", "fragile", "valuable",
            "common", "rare", "imported", "perishable",
        ],
    },
}


DETECTIVE_TEMPLATES = [
    QueryTemplate(
        name="find_unique",
        description="Find the unique entity satisfying all conjunctive constraints.",
        min_turns=3,
        max_turns=20,
        answer_type="str",
    ),
    QueryTemplate(
        name="count_matching",
        description=(
            "Count how many entities satisfy all conjunctive constraints. "
            "Forces a full scan — no short-circuit on first match."
        ),
        min_turns=4,
        max_turns=25,
        answer_type="int",
    ),
    QueryTemplate(
        name="top_k_by_region",
        description=(
            "For several regions at once, find the top-K matching entities by price."
        ),
        min_turns=8,
        max_turns=30,
        answer_type="json",
    ),
]


# ---------------------------------------------------------------------------
# Difficulty axis: scales the candidate-pool size and constraint count, the
# two knobs that govern how much shrinking the model has to do.
#
# Top-K counts are calibrated for a 15-turn training cap. Under the env's
# default ``tool_call_budget_per_turn=5``, the reference full-scan policy needs
# ~8/9/10 turns at d=1/2/3 respectively, leaving enough room for learned
# overhead under a 15-turn cap while still preventing one-shot solutions.
# ---------------------------------------------------------------------------

DIFFICULTY_AXIS: dict[int, dict] = {
    1: {
        "n_entities": 50,
        "count_n_entities": 25,
        "n_constraints": 3,
        "topk_n_entities": 35,
        "topk_regions": 3,
        "topk_k": 3,
    },
    2: {
        "n_entities": 70,
        "count_n_entities": 40,
        "n_constraints": 4,
        "topk_n_entities": 40,
        "topk_regions": 4,
        "topk_k": 3,
    },
    3: {
        "n_entities": 90,
        "count_n_entities": 55,
        "n_constraints": 5,
        "topk_n_entities": 45,
        "topk_regions": 5,
        "topk_k": 3,
    },
    4: {
        "n_entities": 110,
        "count_n_entities": 70,
        "n_constraints": 6,
        "topk_n_entities": 120,
        "topk_regions": 5,
        "topk_k": 3,
    },
    5: {
        "n_entities": 130,
        "count_n_entities": 85,
        "n_constraints": 7,
        "topk_n_entities": 140,
        "topk_regions": 5,
        "topk_k": 4,
    },
}
# count_matching uses ``count_n_entities`` (a smaller pool) — the forcing
# function for multi-turn is ``n_entities / tool_call_budget_per_turn``,
# so 25 entities at budget=5 still requires ≥5 scan turns + accumulation,
# but the smaller world makes it tractable for Qwen to learn the
# checkpoint-progress-before-budget-hit pattern (the actual learning goal).

# count_matching needs a non-trivial number of *answer-satisfying* entities
# (so the answer isn't 0 or 1, which would let the model short-circuit).
# Range tuned to be informative (model has to count) without trivial.
_COUNT_MATCHING_MIN = 3
_COUNT_MATCHING_MAX = 12


def _difficulty_to_params(depth: int, breadth: int) -> tuple[int, int]:
    """Fallback for legacy callers that don't pass ``difficulty``."""
    n_constraints = max(3, min(7, depth + 1))
    n_entities = max(20, min(60, breadth * 6 + 20))
    return n_entities, n_constraints


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _short_id(rng: random.Random) -> str:
    return (
        rng.choice(string.ascii_lowercase)
        + str(rng.randint(0, 9))
        + rng.choice(string.ascii_lowercase)
    )


# ---------------------------------------------------------------------------
# Flavor text: each entity gets a `description` field with prose that is
# never used by any constraint. Purpose: bloat the size of get_entity()
# returns (and to a lesser extent the schema view) so that naive
# context_window.append(get_entity(eid)) blows the cap quickly. Forces the
# model to extract the constraint-relevant fields and discard the prose
# rather than journaling raw observations.
# ---------------------------------------------------------------------------

_DESCRIPTION_PHRASES = [
    "Acquired through a regional supplier during the third quarter audit.",
    "Catalogued under the standard inventory taxonomy with reference codes.",
    "Subject to routine inspection cycles and periodic revaluation.",
    "Stored in a controlled-atmosphere facility with limited access.",
    "Assigned a custodian during the most recent distribution audit.",
    "Documented in the regional manifest with appropriate provenance notes.",
    "Marked for internal review pending final certification approval.",
    "Originally produced at one of three approved manufacturing partners.",
    "Tracked through several intermediate handlers before final placement.",
    "Categorized using both legacy and modern classification systems.",
    "Pending reassignment based on the upcoming logistical review.",
    "Inventoried via the latest revision of the cataloging standard.",
    "Subject to disposition decisions by the regional oversight committee.",
    "Reviewed by two independent inspectors during the last verification cycle.",
    "Maintained in accordance with current regulatory guidelines and standards.",
    "Issued a unique identifier upon entry into the active registry system.",
    "Bound by the terms of the standard custody and chain-of-handling protocol.",
    "Listed in the public-facing index but withheld from external distribution.",
    "Retained in storage for evaluation by the next quarterly review board.",
    "Filed under the appropriate regulatory category with cross-referenced tags.",
    "Subjected to non-destructive analysis during the most recent intake batch.",
    "Logged into the central registry with a timestamped intake record.",
]


def _make_description(rng: random.Random) -> str:
    """Return a 2-4 sentence prose blob (~150-280 chars). Not used by any
    constraint — pure noise that bloats raw observations."""
    n = rng.randint(2, 4)
    sentences = rng.sample(_DESCRIPTION_PHRASES, n)
    return " ".join(sentences)


_OPS_PER_TYPE = {
    "categorical": ["==", "!="],
    "numeric": ["<", ">", "<=", ">="],
    "list": ["contains"],
}


def _sample_value(spec: dict, rng: random.Random) -> Any:
    t = spec["type"]
    if t == "categorical":
        return rng.choice(spec["values"])
    if t == "numeric":
        lo, hi = spec["range"]
        return rng.randint(lo, hi)
    if t == "list":
        n = rng.randint(0, 3)
        return rng.sample(spec["values"], n) if n else []
    raise ValueError(f"Unknown attr type: {t!r}")


def _sample_constraint(
    schema: dict, used_attrs: set[str], rng: random.Random
) -> dict:
    avail = [a for a in schema if a not in used_attrs] or list(schema)
    attr = rng.choice(avail)
    spec = schema[attr]
    t = spec["type"]
    op = rng.choice(_OPS_PER_TYPE[t])
    if t == "categorical":
        value = rng.choice(spec["values"])
    elif t == "numeric":
        lo, hi = spec["range"]
        # threshold inside (lo,hi) so the predicate splits the population
        if hi - lo <= 2:
            value = rng.randint(lo, hi)
        else:
            value = rng.randint(lo + 1, hi - 1)
    else:  # list
        value = rng.choice(spec["values"])
    return {"op": op, "attr": attr, "value": value}


def _sample_constraint_from_attrs(
    schema: dict, allowed_attrs: list[str], used_attrs: set[str], rng: random.Random
) -> dict:
    """Sample a constraint from an explicit attr subset."""
    avail = [a for a in allowed_attrs if a not in used_attrs] or list(allowed_attrs)
    attr = rng.choice(avail)
    spec = schema[attr]
    t = spec["type"]
    op = rng.choice(_OPS_PER_TYPE[t])
    if t == "categorical":
        value = rng.choice(spec["values"])
    elif t == "numeric":
        lo, hi = spec["range"]
        value = rng.randint(lo + 1, hi - 1) if hi - lo > 2 else rng.randint(lo, hi)
    else:
        value = rng.choice(spec["values"])
    return {"op": op, "attr": attr, "value": value}


def _force_satisfy(
    constraint: dict, spec: dict, rng: random.Random
) -> Any:
    """Return a value that satisfies the constraint."""
    op, val = constraint["op"], constraint["value"]
    t = spec["type"]
    if t == "categorical":
        if op == "==":
            return val
        choices = [v for v in spec["values"] if v != val]
        return rng.choice(choices) if choices else val
    if t == "numeric":
        lo, hi = spec["range"]
        if op == "<":
            return rng.randint(lo, max(lo, val - 1))
        if op == "<=":
            return rng.randint(lo, min(hi, val))
        if op == ">":
            return rng.randint(min(hi, val + 1), hi)
        if op == ">=":
            return rng.randint(max(lo, val), hi)
    if t == "list":
        if op == "contains":
            base = [v for v in spec["values"] if v != val]
            extra_n = rng.randint(0, 2)
            extras = rng.sample(base, extra_n) if extra_n else []
            return [val] + extras
    return _sample_value(spec, rng)


def _force_violate(
    constraint: dict, spec: dict, rng: random.Random
) -> Any:
    """Return a value that violates the constraint."""
    op, val = constraint["op"], constraint["value"]
    t = spec["type"]
    if t == "categorical":
        if op == "==":
            choices = [v for v in spec["values"] if v != val]
            return rng.choice(choices) if choices else val
        return val  # ``!=`` violated by equality
    if t == "numeric":
        lo, hi = spec["range"]
        if op == "<":
            return rng.randint(min(hi, val), hi)
        if op == "<=":
            return rng.randint(min(hi, val + 1), hi) if val < hi else hi
        if op == ">":
            return rng.randint(lo, max(lo, val))
        if op == ">=":
            return rng.randint(lo, max(lo, val - 1)) if val > lo else lo
    if t == "list":
        if op == "contains":
            base = [v for v in spec["values"] if v != val]
            n = rng.randint(0, 3)
            return rng.sample(base, n) if n else []
    return _sample_value(spec, rng)


def _sample_satisfying_entity(
    constraints: list[dict], schema: dict, rng: random.Random
) -> dict:
    """Sample an entity satisfying every constraint."""
    attrs: dict[str, Any] = {}
    constrained_attrs = {c["attr"]: c for c in constraints}
    for attr, spec in schema.items():
        if attr in constrained_attrs:
            attrs[attr] = _force_satisfy(constrained_attrs[attr], spec, rng)
        else:
            attrs[attr] = _sample_value(spec, rng)
    attrs["description"] = _make_description(rng)
    return attrs


def _sample_distractor(
    constraints: list[dict],
    schema: dict,
    rng: random.Random,
) -> dict:
    """Sample an entity that violates 1–2 random constraints (chosen uniformly).

    We do this by starting from a satisfying entity and then flipping the
    chosen attributes to violate. This guarantees the distractor *is*
    distinct from the answer on at least one attribute.
    """
    attrs = _sample_satisfying_entity(constraints, schema, rng)
    n_violations = rng.choice([1, 1, 2])  # 67% of distractors violate exactly 1
    n_violations = min(n_violations, len(constraints))
    violated_idxs = rng.sample(range(len(constraints)), n_violations)
    for idx in violated_idxs:
        c = constraints[idx]
        spec = schema[c["attr"]]
        attrs[c["attr"]] = _force_violate(c, spec, rng)
    # Re-roll the description so distractors don't share text with the
    # answer (otherwise the model could spot the answer by description match).
    attrs["description"] = _make_description(rng)
    return attrs


def _rank_region_top(
    entities: dict[str, dict], constraints: list[dict], target_regions: list[str], k: int
) -> list[list]:
    """Return stable JSON-friendly top-K answer by region."""
    out: list[list] = []
    for region in sorted(target_regions):
        eligible = [
            (int(attrs["price"]), eid)
            for eid, attrs in entities.items()
            if attrs.get("region") == region
            and all(eval_predicate(c, attrs) for c in constraints)
        ]
        ranked = sorted(eligible, key=lambda x: (-x[0], x[1]))[:k]
        out.append([region, [[eid, price] for price, eid in ranked]])
    return out


def _leaderboard_churn(
    entities: dict[str, dict], constraints: list[dict], target_regions: list[str], k: int
) -> int:
    """Count times a qualifying entity changes a full per-region top-K board."""
    boards: dict[str, list[tuple[int, str]]] = {r: [] for r in target_regions}
    churn = 0
    for eid, attrs in entities.items():
        region = attrs.get("region")
        if region not in boards:
            continue
        if not all(eval_predicate(c, attrs) for c in constraints):
            continue
        before = list(boards[region])
        board = before + [(int(attrs["price"]), eid)]
        board = sorted(board, key=lambda x: (-x[0], x[1]))[:k]
        if len(before) >= k and board != before:
            churn += 1
        boards[region] = board
    return churn


def _generate_topk_state(
    schema: dict,
    n_entities: int,
    n_constraints: int,
    n_regions: int,
    k: int,
    rng: random.Random,
) -> dict | None:
    """Generate a multi-bucket top-K state with enough leaderboard churn.

    The working set is deliberately wider than a scalar: solvers need a
    per-region top-K board plus a scan cursor. Raw append logs overflow quickly;
    a compact manifest can point to REPL vars holding the full boards.
    """
    allowed_constraint_attrs = [
        a for a in schema.keys() if a not in {"region", "price"}
    ]
    target_regions = rng.sample(schema["region"]["values"], n_regions)
    min_eligible_per_region = k + 5

    for _attempt in range(100):
        used: set[str] = set()
        constraints: list[dict] = []
        for _ in range(n_constraints):
            c = _sample_constraint_from_attrs(
                schema, allowed_constraint_attrs, used, rng
            )
            constraints.append(c)
            used.add(c["attr"])

        attrs_by_id: list[dict] = []
        for region in target_regions:
            # Many eligible entities per bucket, with prices spread enough
            # to produce replacements while scanning.
            for _ in range(min_eligible_per_region + rng.randint(0, 3)):
                attrs = _sample_satisfying_entity(constraints, schema, rng)
                attrs["region"] = region
                attrs["price"] = rng.randint(25, 999)
                attrs_by_id.append(attrs)

        while len(attrs_by_id) < n_entities:
            if rng.random() < 0.75:
                attrs = _sample_distractor(constraints, schema, rng)
                attrs["region"] = rng.choice(schema["region"]["values"])
            else:
                attrs = _sample_satisfying_entity(constraints, schema, rng)
                non_target = [
                    r for r in schema["region"]["values"] if r not in target_regions
                ]
                attrs["region"] = rng.choice(non_target or schema["region"]["values"])
            attrs["price"] = rng.randint(25, 999)
            attrs_by_id.append(attrs)

        ids: set[str] = set()
        pairs: list[tuple[str, dict]] = []
        while len(pairs) < n_entities:
            eid = _short_id(rng)
            if eid in ids:
                continue
            ids.add(eid)
            pairs.append((eid, attrs_by_id[len(pairs)]))

        rng.shuffle(pairs)
        entities = dict(pairs)
        answer = _rank_region_top(entities, constraints, target_regions, k)
        if any(len(region_answer[1]) < k for region_answer in answer):
            continue
        churn = _leaderboard_churn(entities, constraints, target_regions, k)
        if churn < max(4, n_regions * k):
            continue

        return {
            "entities": entities,
            "constraints": constraints,
            "attribute_schema": schema,
            "n_entities": n_entities,
            "n_constraints": len(constraints),
            "_top_k_meta": {
                "target_regions": sorted(target_regions),
                "k": k,
                "rank_attr": "price",
                "answer": answer,
                "leaderboard_churn": churn,
            },
        }
    return None


# ---------------------------------------------------------------------------
# WorldGenerator
# ---------------------------------------------------------------------------


class DetectiveWorld(WorldGenerator):
    """Detective: find the unique entity satisfying all conjunctive constraints."""

    world_type = "detective"

    def get_query_templates(self) -> list[QueryTemplate]:
        return DETECTIVE_TEMPLATES

    def get_system_prompt(self) -> str:
        return _DETECTIVE_PROMPT_PRELUDE

    def generate_state(
        self,
        depth: int,
        breadth: int,
        rng: Any,
        *,
        difficulty: int | None = None,
    ) -> dict:
        # Pick the target state family up front. 1 = find_unique-eligible
        # state; K in [_COUNT_MATCHING_MIN, _COUNT_MATCHING_MAX] = count_matching-
        # eligible state; top_k states have a multi-bucket leaderboard target.
        # dataset.py's template picker rejects mismatches, so this keeps the
        # templates supplied without passing template names into generate_state.
        mode = rng.choices(
            ["find_unique", "count_matching", "top_k_by_region"],
            weights=[0.35, 0.25, 0.40],
            k=1,
        )[0]
        is_count_matching = mode == "count_matching"

        if difficulty is not None:
            d = max(1, min(5, int(difficulty)))
            axis = DIFFICULTY_AXIS[d]
            if mode == "top_k_by_region":
                state = _generate_topk_state(
                    schema=DEFAULT_SCHEMA,
                    n_entities=axis["topk_n_entities"],
                    n_constraints=max(2, min(4, axis["n_constraints"] - 1)),
                    n_regions=axis["topk_regions"],
                    k=axis["topk_k"],
                    rng=rng,
                )
                if state is not None:
                    return state
                mode = "find_unique"
                is_count_matching = False
            # count_matching uses a smaller pool — see note next to DIFFICULTY_AXIS.
            n_entities = (
                axis.get("count_n_entities", axis["n_entities"])
                if is_count_matching else axis["n_entities"]
            )
            n_constraints = axis["n_constraints"]
        else:
            n_entities, n_constraints = _difficulty_to_params(depth, breadth)

        schema = DEFAULT_SCHEMA

        if is_count_matching:
            target_count = rng.randint(_COUNT_MATCHING_MIN, _COUNT_MATCHING_MAX)
        else:
            target_count = 1
        target_count = min(target_count, n_entities - 1)

        for _attempt in range(100):
            # 1. Sample K constraints over distinct attrs (each attr used at
            #    most once → no contradictory or redundant constraints on a
            #    single attribute).
            used: set[str] = set()
            constraints: list[dict] = []
            for _ in range(n_constraints):
                if len(used) >= len(schema):
                    break
                c = _sample_constraint(schema, used, rng)
                constraints.append(c)
                used.add(c["attr"])
            if len(constraints) < n_constraints:
                continue

            # 2. Build target_count satisfying entities. Slot 0 is the
            #    designated answer entity (find_unique uses it).
            sat_attrs_list = []
            for _ in range(target_count):
                a = _sample_satisfying_entity(constraints, schema, rng)
                if not all(eval_predicate(c, a) for c in constraints):
                    break
                sat_attrs_list.append(a)
            if len(sat_attrs_list) < target_count:
                continue

            # 3. Build (n_entities - target_count) distractors that violate at
            #    least one constraint.
            n_distractors = n_entities - target_count
            distractor_attrs_list = []
            distractor_attempts = 0
            while (
                len(distractor_attrs_list) < n_distractors
                and distractor_attempts < n_distractors * 5
            ):
                distractor_attempts += 1
                attrs = _sample_distractor(constraints, schema, rng)
                if all(eval_predicate(c, attrs) for c in constraints):
                    bad_idx = rng.randrange(len(constraints))
                    bad_c = constraints[bad_idx]
                    attrs[bad_c["attr"]] = _force_violate(
                        bad_c, schema[bad_c["attr"]], rng
                    )
                    if all(eval_predicate(c, attrs) for c in constraints):
                        continue
                distractor_attrs_list.append(attrs)
            if len(distractor_attrs_list) < n_distractors:
                continue

            # 4. Assign IDs and shuffle entity order so the answer (and the
            #    set of satisfying entities) is uniformly distributed across
            #    ENTITY_IDS positions. Python dicts preserve insertion order
            #    and the kernel exposes ENTITY_IDS = list(entities.keys()) —
            #    if we always inserted ``ans_id`` first the model could (and
            #    did) learn to just submit ENTITY_IDS[0]. See run
            #    ekb80e1p4imbzxy30vs1qlsi step-30 analysis: 46/46 short-
            #    correct find_unique rollouts had answer at position 0.
            ids: set[str] = set()
            ans_id = _short_id(rng)
            ids.add(ans_id)
            other_ids: list[str] = []
            id_attempts = 0
            while len(other_ids) < n_entities - 1 and id_attempts < n_entities * 5:
                id_attempts += 1
                eid = _short_id(rng)
                if eid in ids:
                    continue
                ids.add(eid)
                other_ids.append(eid)
            if len(other_ids) < n_entities - 1:
                continue

            other_attrs = sat_attrs_list[1:] + distractor_attrs_list
            rng.shuffle(other_attrs)

            all_pairs = [(ans_id, sat_attrs_list[0])] + list(
                zip(other_ids, other_attrs)
            )
            rng.shuffle(all_pairs)
            entities: dict[str, dict] = dict(all_pairs)

            # 5. Verify exact satisfying count.
            satisfying = [
                eid for eid, a in entities.items()
                if all(eval_predicate(c, a) for c in constraints)
            ]
            if len(satisfying) != target_count:
                continue
            if ans_id not in satisfying:
                continue

            # 6. Each constraint must be *binding* — i.e. removing it changes
            #    the satisfying count. Prevents redundant constraints.
            binding = True
            for skip_idx in range(len(constraints)):
                others = [c for i, c in enumerate(constraints) if i != skip_idx]
                n_left = sum(
                    1 for a in entities.values()
                    if all(eval_predicate(c, a) for c in others)
                )
                if n_left == target_count:
                    binding = False
                    break
            if not binding:
                continue

            return {
                "entities": entities,
                "constraints": constraints,
                "attribute_schema": schema,
                "n_entities": n_entities,
                "n_constraints": len(constraints),
                "_answer_id": ans_id if target_count == 1 else None,
                "_satisfying_count": target_count,
                "_satisfying_ids": sorted(satisfying),
            }

        raise RuntimeError(
            f"detective: failed to generate (n_entities={n_entities}, "
            f"n_constraints={n_constraints}, target_count={target_count}) "
            "after 100 attempts"
        )

    def generate_query(
        self,
        template: QueryTemplate,
        state: dict,
        target_turns: int,
        rng: Any,
    ) -> tuple[str, Any, int, int]:
        rich = self.generate_query_rich(template, state, target_turns, rng)
        if rich is None:
            raise RuntimeError(
                f"detective: state does not match template={template.name}"
            )
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
        rng: Any,
    ) -> dict | None:
        constraints = state["constraints"]
        cstr_lines = "\n".join(
            f"  - `{c['attr']}` {c['op']} {c['value']!r}"
            for c in constraints
        )
        sat_count = state.get("_satisfying_count")

        if template.name == "find_unique":
            if sat_count != 1:
                return None
            ans = state["_answer_id"]
            query_text = (
                f"Find the unique entity (among {state['n_entities']}) that "
                f"satisfies ALL of the following {len(constraints)} "
                f"constraints:\n{cstr_lines}\n\n"
                f"Use get_entity(id) to inspect a full attribute dict, or "
                f"query_attribute(id, attr_name) to read a single attribute. "
                f"ENTITY_IDS, ATTRIBUTE_SCHEMA, and CONSTRAINTS are pre-seeded "
                f"in the kernel. Submit the unique entity id via "
                f"submit_answer(\"<id>\")."
            )
            return {
                "query_text": query_text,
                "expected_answer": ans,
                "actual_depth": state["n_constraints"],
                "actual_breadth": state["n_entities"],
                "target_entities": [ans],
                "parameter_refs": [],
                "rules": [],
            }

        if template.name == "count_matching":
            if sat_count is None or not (
                _COUNT_MATCHING_MIN <= sat_count <= _COUNT_MATCHING_MAX
            ):
                return None
            query_text = (
                f"Among the {state['n_entities']} entities, count how many "
                f"satisfy ALL of the following {len(constraints)} "
                f"constraints:\n{cstr_lines}\n\n"
                f"Use get_entity(id) to inspect a full attribute dict, or "
                f"query_attribute(id, attr_name) to read a single attribute. "
                f"ENTITY_IDS, ATTRIBUTE_SCHEMA, and CONSTRAINTS are pre-seeded "
                f"in the kernel. Submit the count (an integer) via "
                f"submit_answer(<count>)."
            )
            return {
                "query_text": query_text,
                "expected_answer": sat_count,
                "actual_depth": state["n_constraints"],
                "actual_breadth": state["n_entities"],
                "target_entities": list(state.get("_satisfying_ids") or []),
                "parameter_refs": [],
                "rules": [],
            }

        if template.name == "top_k_by_region":
            meta = state.get("_top_k_meta") or {}
            if not meta:
                return None
            target_regions = list(meta["target_regions"])
            k = int(meta["k"])
            answer = meta["answer"]
            cstr_lines = "\n".join(
                f"  - `{c['attr']}` {c['op']} {c['value']!r}"
                for c in constraints
            )
            query_text = (
                f"For each target region in {target_regions!r}, find the top {k} "
                f"entities by highest `price` among the {state['n_entities']} "
                f"entities that satisfy ALL of these {len(constraints)} "
                f"constraints:\n{cstr_lines}\n\n"
                f"Rank within each region by price descending, breaking ties by "
                f"entity id ascending. Submit JSON as a list of "
                f"[region, [[entity_id, price], ...]] pairs sorted by region, "
                f"for example [[\"east\", [[\"a1b\", 400]]]]. Use get_entity(id) "
                f"or query_attribute(id, attr_name). ENTITY_IDS, ATTRIBUTE_SCHEMA, "
                f"and CONSTRAINTS are pre-seeded."
            )
            return {
                "query_text": query_text,
                "expected_answer": json.dumps(answer, separators=(",", ":")),
                "actual_depth": len(constraints),
                "actual_breadth": state["n_entities"],
                "target_entities": [
                    eid for _region, rows in answer for eid, _price in rows
                ],
                "parameter_refs": [],
                "rules": [],
            }

        return None


_DETECTIVE_PROMPT_PRELUDE = (
    "detective world: a pool of entities + a list of conjunctive constraints. "
    "Find the unique entity satisfying all of them via get_entity / "
    "query_attribute. Submit the entity id."
)
