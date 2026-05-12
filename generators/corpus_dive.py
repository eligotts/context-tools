"""corpus_dive — long-document exploration world.

Forces *pruning* of the scratchpad. The world is a tree with ~80–150 nodes
that the model can only inspect through paginated tools (one call per
node). Most nodes are irrelevant to the question; the right move is to
maintain a small running aggregate (matches list / running count / sum)
in ``context_window`` and discard everything else.

State schema
------------
::

    {
        "tree": {                       # flat path -> node mapping
            "root":              {"attrs": {...}, "children": ["a1b", ...]},
            "root.a1b":          {"attrs": {...}, "children": [...]},
            ...
            "root.a1b.f7q.x3w":  {"attrs": {...}, "children": []},  # leaf
        },
        "root_path": "root",
        "_question_meta": {             # populated by generate_query_rich;
            "predicate": {              # stripped before the sandbox sees it
                "attr": "color", "op": "==", "value": "red"
            },
            "subtree_path": "root.a1b",
            "template": "count_under_path",
        },
    }

Tools the model sees
--------------------
- ``list_keys(path) -> list[str]`` — children *names* (relative, e.g. ``["x3w","f7q"]``)
- ``read_node(path) -> dict``     — own attrs only (children's data NOT included)
- ``ROOT_PATH``                   — pre-seeded kernel variable (path of root)

Path handling: children are returned by short relative names. To descend,
the model builds ``f"{base}.{child}"``.
"""

from __future__ import annotations

import random
import string
from typing import Any

from .base import QueryTemplate, WorldGenerator


# ---------------------------------------------------------------------------
# Schema (shared style with rule_hunt)
# ---------------------------------------------------------------------------

DEFAULT_NODE_ATTR_SCHEMA: dict = {
    # categorical
    "color": ["red", "blue", "green", "yellow", "purple"],
    "kind": ["alpha", "beta", "gamma", "delta"],
    # numeric (lo, hi inclusive)
    "size": (1, 100),
    "value": (1, 50),
    # list (small subset of available tags per node)
    "tags": ["industrial", "organic", "fragile", "valuable", "common", "rare"],
}


CORPUS_DIVE_TEMPLATES = [
    QueryTemplate(
        name="count_under_path",
        description="Count nodes under a subtree path matching a predicate.",
        min_turns=4,
        max_turns=20,
        answer_type="int",
    ),
    QueryTemplate(
        name="sum_values_under_path",
        description="Sum `value` of nodes under a subtree path matching a predicate.",
        min_turns=4,
        max_turns=20,
        answer_type="int",
    ),
]


# ---------------------------------------------------------------------------
# Predicate evaluation (also used by the solver and any future debug tools)
# ---------------------------------------------------------------------------

# op-set per attr: which operators we'll sample for that attr
_OPS_BY_ATTR = {
    "color": ["==", "!="],
    "kind": ["==", "!="],
    "size": ["<", ">", "<=", ">="],
    "value": ["<", ">", "<=", ">="],
    "tags": ["contains"],
}


def eval_predicate(pred: dict, attrs: dict) -> bool:
    """Evaluate a single-conjunct predicate against a node's attrs."""
    a, op, v = pred.get("attr"), pred.get("op"), pred.get("value")
    actual = attrs.get(a)
    if op == "==":
        return actual == v
    if op == "!=":
        return actual != v
    if op == "<":
        return actual is not None and actual < v
    if op == ">":
        return actual is not None and actual > v
    if op == "<=":
        return actual is not None and actual <= v
    if op == ">=":
        return actual is not None and actual >= v
    if op == "contains":
        return isinstance(actual, list) and v in actual
    raise ValueError(f"Unknown predicate op: {op!r}")


def _format_predicate_for_query(pred: dict) -> str:
    a, op, v = pred["attr"], pred["op"], pred["value"]
    if op == "contains":
        return f"`{a}` contains '{v}'"
    if isinstance(v, str):
        return f"`{a}` {op} '{v}'"
    return f"`{a}` {op} {v}"


def subtree_paths(tree: dict, root: str) -> list[str]:
    """All paths in the subtree rooted at `root` (DFS order)."""
    out: list[str] = []
    stack: list[str] = [root]
    while stack:
        p = stack.pop()
        out.append(p)
        node = tree.get(p, {})
        for c in node.get("children", []):
            stack.append(f"{p}.{c}")
    return out


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _sample_node_attrs(rng: random.Random) -> dict:
    return {
        "color": rng.choice(DEFAULT_NODE_ATTR_SCHEMA["color"]),
        "kind": rng.choice(DEFAULT_NODE_ATTR_SCHEMA["kind"]),
        "size": rng.randint(*DEFAULT_NODE_ATTR_SCHEMA["size"]),
        "value": rng.randint(*DEFAULT_NODE_ATTR_SCHEMA["value"]),
        "tags": (
            rng.sample(DEFAULT_NODE_ATTR_SCHEMA["tags"], rng.randint(0, 3))
        ),
    }


def _short_id(rng: random.Random) -> str:
    """3-char id like 'a1b' — matches the project style."""
    return (
        rng.choice(string.ascii_lowercase)
        + str(rng.randint(0, 9))
        + rng.choice(string.ascii_lowercase)
    )


def _build_tree(
    target_depth: int, target_branching: int, rng: random.Random
) -> tuple[dict, str]:
    """Build a tree as a flat path -> node dict."""
    tree: dict[str, dict] = {}
    root_path = "root"

    def build(path: str, depth_remaining: int) -> None:
        attrs = _sample_node_attrs(rng)
        if depth_remaining <= 0:
            tree[path] = {"attrs": attrs, "children": []}
            return
        # Variable branching for diversity
        n = rng.randint(
            max(2, target_branching - 1),
            target_branching + 1,
        )
        children: list[str] = []
        used: set[str] = set()
        guard = 0
        while len(children) < n and guard < n * 20:
            guard += 1
            cid = _short_id(rng)
            if cid in used:
                continue
            used.add(cid)
            children.append(cid)
        tree[path] = {"attrs": attrs, "children": children}
        for c in children:
            build(f"{path}.{c}", depth_remaining - 1)

    build(root_path, target_depth)
    return tree, root_path


def _sample_predicate(
    rng: random.Random, exclude_attr: str | None = None
) -> dict:
    """Sample a single-conjunct predicate."""
    attrs = list(_OPS_BY_ATTR.keys())
    if exclude_attr:
        attrs = [a for a in attrs if a != exclude_attr]
    attr = rng.choice(attrs)
    op = rng.choice(_OPS_BY_ATTR[attr])
    if attr in ("color", "kind"):
        value: Any = rng.choice(DEFAULT_NODE_ATTR_SCHEMA[attr])
    elif attr in ("size", "value"):
        lo, hi = DEFAULT_NODE_ATTR_SCHEMA[attr]
        value = rng.randint(lo + 1, hi - 1)
    else:  # tags
        value = rng.choice(DEFAULT_NODE_ATTR_SCHEMA[attr])
    return {"attr": attr, "op": op, "value": value}


# ---------------------------------------------------------------------------
# Difficulty mapping
# ---------------------------------------------------------------------------


def _difficulty_to_tree_params(depth: int, breadth: int) -> tuple[int, int]:
    """Fallback mapping for depth/breadth → (tree_depth, tree_branching)
    when ``difficulty`` is not supplied."""
    target_depth = max(2, min(4, depth))
    target_branching = max(3, min(5, breadth // 2 + 2))
    return target_depth, target_branching


# Difficulty axis: scales the subtree-of-interest size, which is the
# primary pruning-pressure knob. Bigger subtree (with the same small
# matching set) = more nodes to scan-and-discard. ``tree_depth`` and
# ``tree_branching`` co-scale so a subtree of the target size can be
# found inside the generated tree.
DIFFICULTY_AXIS: dict[int, dict] = {
    1: {"tree_depth": 3, "tree_branching": 3, "subtree_size_target": 8},
    2: {"tree_depth": 3, "tree_branching": 3, "subtree_size_target": 15},
    3: {"tree_depth": 3, "tree_branching": 4, "subtree_size_target": 25},
    4: {"tree_depth": 4, "tree_branching": 4, "subtree_size_target": 45},
    5: {"tree_depth": 4, "tree_branching": 5, "subtree_size_target": 80},
}


def _subtree_bounds_for_target(target: int) -> tuple[int, int]:
    """Acceptable subtree size window around a target (±50%, with a floor)."""
    lo = max(4, int(target * 0.6))
    hi = max(lo + 4, int(target * 1.6))
    return lo, hi


def _match_bounds_for_subtree(sub_size: int) -> tuple[int, int]:
    """How many predicate-matching nodes the subtree should contain.

    Match count stays small (3–10) regardless of subtree size, so as
    subtree grows the noise ratio (sub_size / matches) grows with it —
    that's the actual pruning pressure.
    """
    lo = 3
    hi = max(lo + 1, min(10, sub_size // 4))
    return lo, hi


# ---------------------------------------------------------------------------
# WorldGenerator
# ---------------------------------------------------------------------------


class CorpusDiveWorld(WorldGenerator):
    """Long-document exploration: walk a JSON tree, aggregate over a subtree."""

    world_type = "corpus_dive"

    def get_query_templates(self) -> list[QueryTemplate]:
        return CORPUS_DIVE_TEMPLATES

    def get_system_prompt(self) -> str:
        return _CORPUS_DIVE_PROMPT_PRELUDE

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
            target_depth = axis["tree_depth"]
            target_branching = axis["tree_branching"]
            subtree_size_target = axis["subtree_size_target"]
        else:
            target_depth, target_branching = _difficulty_to_tree_params(
                depth, breadth
            )
            subtree_size_target = 25  # neutral default for legacy callers
        tree, root_path = _build_tree(target_depth, target_branching, rng)
        return {
            "tree": tree,
            "root_path": root_path,
            "depth": target_depth,
            "branching": target_branching,
            # Generator-internal: read by generate_query_rich, then stripped
            # before the sandbox sees world_state.
            "_subtree_size_target": subtree_size_target,
        }

    def generate_query_rich(
        self,
        template: QueryTemplate,
        state: dict,
        target_turns: int,
        rng: Any,
    ) -> dict | None:
        tree: dict = state["tree"]
        root_path: str = state["root_path"]
        target_size = int(state.get("_subtree_size_target", 25))
        sub_lo, sub_hi = _subtree_bounds_for_target(target_size)

        # Find subtree paths whose subtree size is in the desired window
        # around the difficulty-driven target.
        candidates: list[tuple[str, int]] = []
        for p in tree.keys():
            if p == root_path:
                continue
            sz = len(subtree_paths(tree, p))
            if sub_lo <= sz <= sub_hi:
                candidates.append((p, sz))
        if not candidates:
            # Fallback: looser bound — accept anything >= half the floor
            for p in tree.keys():
                if p == root_path:
                    continue
                sz = len(subtree_paths(tree, p))
                if sz >= max(4, sub_lo // 2):
                    candidates.append((p, sz))
        if not candidates:
            return None

        rng.shuffle(candidates)

        # For sum_values_under_path, exclude `value` from predicate so the
        # filter and the aggregate are over different attrs.
        exclude = (
            "value" if template.name == "sum_values_under_path" else None
        )

        for sub_path, sub_size in candidates[:30]:
            sub_paths = subtree_paths(tree, sub_path)
            match_lo, match_hi = _match_bounds_for_subtree(sub_size)
            # Try several predicates per subtree before giving up
            chosen_pred: dict | None = None
            chosen_matches: list[str] = []
            for _ in range(60):
                pred = _sample_predicate(rng, exclude_attr=exclude)
                matches = [
                    p for p in sub_paths
                    if eval_predicate(pred, tree[p]["attrs"])
                ]
                if match_lo <= len(matches) <= match_hi:
                    chosen_pred = pred
                    chosen_matches = matches
                    break
            if chosen_pred is None:
                continue

            if template.name == "count_under_path":
                answer = len(chosen_matches)
                query_text = (
                    f"Under path '{sub_path}' (including itself), count the "
                    f"nodes where {_format_predicate_for_query(chosen_pred)}. "
                    f"Use list_keys(path) to find children and read_node(path) "
                    f"to inspect attributes. ROOT_PATH is pre-seeded. Submit "
                    f"the integer count via submit_answer(n)."
                )
            elif template.name == "sum_values_under_path":
                answer = sum(
                    int(tree[p]["attrs"].get("value", 0))
                    for p in chosen_matches
                )
                query_text = (
                    f"Under path '{sub_path}' (including itself), sum the "
                    f"`value` attribute of nodes where "
                    f"{_format_predicate_for_query(chosen_pred)}. Use "
                    f"list_keys(path) and read_node(path) to traverse. "
                    f"ROOT_PATH is pre-seeded. Submit the integer sum via "
                    f"submit_answer(n)."
                )
            else:
                continue

            # Stash predicate + subtree path in state for the solver.
            # context_tools setup_state strips this before the sandbox sees
            # world_state.
            state["_question_meta"] = {
                "predicate": chosen_pred,
                "subtree_path": sub_path,
                "template": template.name,
            }

            return {
                "query_text": query_text,
                "expected_answer": str(answer),
                "actual_depth": state["depth"],
                "actual_breadth": state["branching"],
                "target_entities": [sub_path],
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
            return ("", "0", state.get("depth", 0), state.get("branching", 0))
        return (
            rich["query_text"],
            rich["expected_answer"],
            rich["actual_depth"],
            rich["actual_breadth"],
        )


_CORPUS_DIVE_PROMPT_PRELUDE = (
    "corpus_dive world: walk a JSON tree using list_keys(path) and "
    "read_node(path). Aggregate over a subtree and submit an integer."
)
