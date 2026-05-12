"""maze_walk — graph navigation with backtracking.

The model navigates a connected graph from a known start node to find
a hidden goal node. Each ``move()`` reveals the new node's neighbors
and a per-node ``secret`` string; the goal node's secret is the answer.

Designed to force *push/pop* discipline in the scratchpad. The state
that wins this task has *oscillating cardinality* — the model maintains
a current path that GROWS on advance and SHRINKS on backtrack — which
no other family in this set requires. Append-only play fails because
there's no way to represent an "undo" once the model has committed to a
path that turned out to be a dead end.

State schema
------------
::

    {
        "graph": {                  # adjacency dict (relative neighbor names)
            "n001": ["n002", "n005"],
            "n002": ["n001", "n003"],
            ...
        },
        "node_secrets": {           # opaque per-node strings
            "n001": "a3k9p",
            ...
        },
        "start": "n001",
        "goal": "n042",
        "_shortest_path_length": int,
        "n_nodes": int,
        "branching": int,
        "_question_meta": {...},    # generator-internal
    }

Note: the ``_world_state`` is in-process visible to the model's code,
so a sufficiently-adversarial model could read ``_world_state['goal']``
and ``_world_state['node_secrets'][goal]`` directly. The intended use
is non-adversarial training; the discipline being trained is path
maintenance, not infosec.
"""

from __future__ import annotations

import collections
import random
import string
from typing import Any

from .base import QueryTemplate, WorldGenerator


MAZE_WALK_TEMPLATES = [
    QueryTemplate(
        name="find_goal_secret",
        description="Navigate to the goal node and submit its secret string.",
        min_turns=3,
        max_turns=30,
        answer_type="str",
    ),
]


# ---------------------------------------------------------------------------
# Difficulty axis: bigger graph + longer shortest-path target = more
# backtracking pressure (more chances to commit to a dead-end branch).
# ---------------------------------------------------------------------------

DIFFICULTY_AXIS: dict[int, dict] = {
    1: {"n_nodes": 25,  "branching": 2, "shortest_path_target": 4},
    2: {"n_nodes": 45,  "branching": 2, "shortest_path_target": 6},
    3: {"n_nodes": 70,  "branching": 2, "shortest_path_target": 8},
    4: {"n_nodes": 100, "branching": 3, "shortest_path_target": 10},
    5: {"n_nodes": 140, "branching": 3, "shortest_path_target": 12},
}


def _difficulty_to_params(depth: int, breadth: int) -> tuple[int, int, int]:
    """Fallback for legacy callers that don't pass ``difficulty``."""
    n_nodes = max(20, min(140, 10 * depth + 10 * breadth))
    branching = max(2, min(4, breadth // 2 + 1))
    target_path = max(4, min(18, depth * 3))
    return n_nodes, branching, target_path


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def _short_id(rng: random.Random, used: set[str]) -> str:
    """3-char node id like 'a1b' (collision-checked against ``used``)."""
    while True:
        cid = (
            rng.choice(string.ascii_lowercase)
            + str(rng.randint(0, 9))
            + rng.choice(string.ascii_lowercase)
        )
        if cid not in used:
            return cid


def _random_secret(rng: random.Random) -> str:
    """Opaque 6-char per-node secret. Goal's secret is the answer."""
    alpha = string.ascii_lowercase + string.digits
    return "".join(rng.choice(alpha) for _ in range(6))


def _build_graph(
    n_nodes: int, branching: int, rng: random.Random
) -> tuple[dict[str, list[str]], list[str]]:
    """Build a connected graph: spanning tree + extra edges to reach
    target average degree ≈ ``branching``.

    Returns (adjacency_dict, node_list).
    """
    nodes: list[str] = []
    used: set[str] = set()
    for _ in range(n_nodes):
        nid = _short_id(rng, used)
        used.add(nid)
        nodes.append(nid)

    adj: dict[str, set[str]] = {n: set() for n in nodes}

    # Spanning tree, biased toward chain-like (longer-diameter) shapes.
    # Each new node attaches to one of the LAST ``recent_window`` nodes
    # placed (with a small chance of a "shortcut" to any earlier node).
    # Pure-uniform parent selection produces ~sqrt(N)-depth trees, which
    # is too shallow for our target shortest-path lengths at high N.
    recent_window = 4
    for i in range(1, n_nodes):
        if rng.random() < 0.85:
            lo = max(0, i - recent_window)
            parent_idx = rng.randint(lo, i - 1)
        else:
            parent_idx = rng.randint(0, i - 1)
        a, b = nodes[i], nodes[parent_idx]
        adj[a].add(b)
        adj[b].add(a)

    # Add extra edges until the average degree ~= branching. A graph
    # with ``E`` edges and ``N`` nodes has avg degree ``2E/N``. Only
    # connect nodes that are CLOSE in the chain order (within
    # ``shortcut_horizon``) — this gives the graph some local branching
    # (dead ends, alternate paths) without adding long-range shortcuts
    # that would collapse the diameter.
    shortcut_horizon = 5
    target_edges = (n_nodes * branching) // 2
    current_edges = sum(len(s) for s in adj.values()) // 2
    safety = 0
    while current_edges < target_edges and safety < target_edges * 12:
        safety += 1
        i = rng.randrange(n_nodes)
        offset = rng.randint(1, shortcut_horizon)
        j = i + offset if rng.random() < 0.5 else i - offset
        if j < 0 or j >= n_nodes:
            continue
        a, b = nodes[i], nodes[j]
        if a == b or b in adj[a]:
            continue
        adj[a].add(b)
        adj[b].add(a)
        current_edges += 1

    graph = {n: sorted(adj[n]) for n in nodes}
    return graph, nodes


def _bfs_distances(graph: dict[str, list[str]], start: str) -> dict[str, int]:
    """BFS distance from ``start`` to all reachable nodes."""
    dist: dict[str, int] = {start: 0}
    frontier = collections.deque([start])
    while frontier:
        node = frontier.popleft()
        for nbr in graph.get(node, []):
            if nbr not in dist:
                dist[nbr] = dist[node] + 1
                frontier.append(nbr)
    return dist


# ---------------------------------------------------------------------------
# WorldGenerator
# ---------------------------------------------------------------------------


class MazeWalkWorld(WorldGenerator):
    """Maze walk: navigate a graph to find the goal node's secret."""

    world_type = "maze_walk"

    def get_query_templates(self) -> list[QueryTemplate]:
        return MAZE_WALK_TEMPLATES

    def get_system_prompt(self) -> str:
        return _MAZE_WALK_PROMPT_PRELUDE

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
            n_nodes = axis["n_nodes"]
            branching = axis["branching"]
            target_path = axis["shortest_path_target"]
        else:
            n_nodes, branching, target_path = _difficulty_to_params(
                depth, breadth
            )

        # Try a few graphs until one produces a start/goal pair at the
        # target shortest-path distance. With reasonable parameters this
        # almost always succeeds on the first attempt.
        for _attempt in range(40):
            graph, nodes = _build_graph(n_nodes, branching, rng)
            start = rng.choice(nodes)
            dists = _bfs_distances(graph, start)

            # Look for nodes at exactly the target distance.
            cands = [n for n, d in dists.items() if d == target_path]
            if not cands:
                # Accept the closest distance ≥ target - 1 as fallback.
                for fallback_d in (target_path - 1, target_path - 2):
                    if fallback_d <= 0:
                        break
                    cands = [n for n, d in dists.items() if d == fallback_d]
                    if cands:
                        target_path = fallback_d
                        break
            if not cands:
                continue

            goal = rng.choice(cands)
            secrets = {n: _random_secret(rng) for n in nodes}

            # Ensure all secrets are unique (so the model can't accidentally
            # match a non-goal node's secret).
            if len(set(secrets.values())) != len(secrets):
                continue

            return {
                "graph": graph,
                "node_secrets": secrets,
                "start": start,
                "goal": goal,
                "n_nodes": n_nodes,
                "branching": branching,
                "_shortest_path_length": dists[goal],
            }

        raise RuntimeError(
            f"maze_walk: failed to generate a graph with start/goal at "
            f"distance {target_path} after 40 attempts"
        )

    def generate_query(
        self,
        template: QueryTemplate,
        state: dict,
        target_turns: int,
        rng: Any,
    ) -> tuple[str, Any, int, int]:
        secrets = state["node_secrets"]
        goal = state["goal"]
        ans = secrets[goal]
        sp = state["_shortest_path_length"]
        n = state["n_nodes"]
        query_text = (
            f"You are at node '{state['start']}' in a connected graph of "
            f"{n} nodes. Navigate to the goal node and submit the goal's "
            f"`secret` string.\n\n"
            f"Use look() to inspect the current node (you'll see its "
            f"secret, neighbors, and an `is_goal` flag) and move(target) "
            f"to step to a neighbor. Both tools cost 1 from the per-turn "
            f"budget. START and N_NODES are pre-seeded in the kernel. "
            f"The goal is reachable in at most {sp + 4} hops; submit "
            f"the goal node's secret via submit_answer(\"<secret>\")."
        )
        return query_text, ans, sp, n

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
            "target_entities": [state["start"], state["goal"]],
            "parameter_refs": [],
            "rules": [],
        }


_MAZE_WALK_PROMPT_PRELUDE = (
    "maze_walk world: navigate a graph from START to the hidden goal via "
    "look() and move(target). Submit the goal node's `secret` string."
)
