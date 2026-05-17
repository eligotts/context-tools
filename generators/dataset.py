"""Dataset generation and export utilities for context-management families.
"""

import json
import random
from collections import Counter

from .base import (
    Question,
    QueryTemplate,
    TrainingExample,
    WorldGenerator,
)
from .rule_hunt import RuleHuntWorld
from .corpus_dive import CorpusDiveWorld
from .timeline_track import TimelineTrackWorld
from .detective import DetectiveWorld
from .maze_walk import MazeWalkWorld
from .adaptive_cursor import AdaptiveCursorWorld
from .corpus_trail import CorpusTrailWorld
from .solver import ReferenceSolver, SolverError


# Per-family weight dicts. Pass any of these (or your own combination) as
# ``world_weights`` to ``generate_dataset``.
RULE_HUNT_WORLD_WEIGHTS = {"rule_hunt": 1.0}
CORPUS_DIVE_WORLD_WEIGHTS = {"corpus_dive": 1.0}
TIMELINE_TRACK_WORLD_WEIGHTS = {"timeline_track": 1.0}
DETECTIVE_WORLD_WEIGHTS = {"detective": 1.0}
MAZE_WALK_WORLD_WEIGHTS = {"maze_walk": 1.0}
ADAPTIVE_CURSOR_WORLD_WEIGHTS = {"adaptive_cursor": 1.0}
CORPUS_TRAIL_WORLD_WEIGHTS = {"corpus_trail": 1.0}

# Equal-weight mix of all 5 families.
CONTEXT_MGMT_WORLD_WEIGHTS = {
    "rule_hunt": 1.0 / 5,
    "corpus_dive": 1.0 / 5,
    "timeline_track": 1.0 / 5,
    "detective": 1.0 / 5,
    "maze_walk": 1.0 / 5,
}

# Default if no weights are passed: the 5-way equal mix.
DEFAULT_WORLD_WEIGHTS = CONTEXT_MGMT_WORLD_WEIGHTS

# Default turn-target distribution (weighted toward harder, deeper rollouts).
DEFAULT_TURN_DISTRIBUTION = {
    3: 0.30,
    4: 0.35,
    5: 0.35,
}


def _create_world_generator(world_type: str) -> WorldGenerator:
    """Create a world generator by type."""
    if world_type == "rule_hunt":
        return RuleHuntWorld()
    if world_type == "corpus_dive":
        return CorpusDiveWorld()
    if world_type == "timeline_track":
        return TimelineTrackWorld()
    if world_type == "detective":
        return DetectiveWorld()
    if world_type == "maze_walk":
        return MazeWalkWorld()
    if world_type == "adaptive_cursor":
        return AdaptiveCursorWorld()
    if world_type == "corpus_trail":
        return CorpusTrailWorld()
    raise ValueError(f"Unknown world type: {world_type}")


def _sample_from_distribution(
    distribution: dict[int, float], rng: random.Random
) -> int:
    values = list(distribution.keys())
    weights = list(distribution.values())
    return rng.choices(values, weights=weights, k=1)[0]


def _sample_world_type(
    world_weights: dict[str, float], rng: random.Random
) -> str:
    types = list(world_weights.keys())
    weights = list(world_weights.values())
    return rng.choices(types, weights=weights, k=1)[0]


def _select_template_for_turns(
    templates: list[QueryTemplate],
    target_turns: int,
    rng: random.Random,
) -> QueryTemplate | None:
    compatible = [
        t for t in templates if t.min_turns <= target_turns <= t.max_turns
    ]
    if compatible:
        return rng.choice(compatible)
    return rng.choice(templates) if templates else None


def _difficulty_to_params(difficulty: int) -> tuple[int, int]:
    """Convert global difficulty (1–5) to legacy ``depth``/``breadth``.

    The new families ignore these and use the explicit ``difficulty`` kwarg
    (passed alongside in ``generate_state``) — the depth/breadth values are
    only here so generators that still consult them have something sensible
    to read.
    """
    depth_map = {1: 2, 2: 3, 3: 3, 4: 4, 5: 4}
    breadth_map = {1: 3, 2: 4, 3: 5, 4: 6, 5: 8}
    return depth_map.get(difficulty, 4), breadth_map.get(difficulty, 6)


def generate_example(
    world_type: str,
    target_turns: int,
    difficulty: int,
    rng: random.Random,
) -> TrainingExample | None:
    """Generate one training example.

    Returns None if generation fails (e.g. solver mismatch, world too small
    for the requested shape, balance check fails).
    """
    generator = _create_world_generator(world_type)
    templates = generator.get_query_templates()

    template = _select_template_for_turns(templates, target_turns, rng)
    if not template:
        return None

    depth, breadth = _difficulty_to_params(difficulty)
    depth = max(depth, target_turns - 1)

    state = generator.generate_state(
        depth=depth, breadth=breadth, rng=rng, difficulty=difficulty,
    )

    try:
        rich = generator.generate_query_rich(
            template=template,
            state=state,
            target_turns=target_turns,
            rng=rng,
        )
    except Exception:
        return None
    if rich is None:
        return None
    query = rich["query_text"]
    expected_answer = rich["expected_answer"]
    actual_depth = rich["actual_depth"]
    actual_breadth = rich["actual_breadth"]
    target_entities = list(rich.get("target_entities", []))
    parameter_refs = list(rich.get("parameter_refs", []))
    rules = list(rich.get("rules", []))

    # Strip helper keys that some generators stash for internal use.
    clean_state = {
        k: v for k, v in state.items()
        if not k.endswith("_by_depth") and not k.endswith("_by_level")
    }

    example_id = rng.randint(0, 2**31 - 1)

    # Solver verifiability check.
    solver_question = Question(
        template_name=template.name,
        query_text=query,
        expected_answer=str(expected_answer),
        answer_type=template.answer_type,
        target_entities=target_entities,
        parameter_refs=parameter_refs,
    )
    try:
        solver_result = ReferenceSolver(world_type).solve(
            clean_state, solver_question, rules=rules
        )
    except SolverError:
        return None
    if solver_result.answer != str(expected_answer):
        return None
    optimal_turns = len(solver_result.turns)

    return TrainingExample(
        example_id=example_id,
        world_type=world_type,
        system_prompt=generator.get_system_prompt(),
        user_query=query,
        state=clean_state,
        optimal_turns=optimal_turns,
        expected_answer=expected_answer,
        answer_type=template.answer_type,
        difficulty=difficulty,
        depth=actual_depth,
        breadth=actual_breadth,
        query_template=template.name,
        target_entities=target_entities,
        parameter_refs=parameter_refs,
        rules=rules,
    )


def generate_dataset(
    num_examples: int,
    overall_difficulty: int = 3,
    world_weights: dict[str, float] | None = None,
    turn_distribution: dict[int, float] | None = None,
    seed: int = 42,
) -> list[TrainingExample]:
    """Generate a dataset of single-answer training examples.

    ``overall_difficulty`` is jittered ±1 per example (clamped to 1–5).
    """
    rng = random.Random(seed)
    if world_weights is None:
        world_weights = DEFAULT_WORLD_WEIGHTS
    if turn_distribution is None:
        turn_distribution = DEFAULT_TURN_DISTRIBUTION

    examples: list[TrainingExample] = []
    used_ids: set[int] = set()
    attempts = 0
    max_attempts = num_examples * 3

    while len(examples) < num_examples and attempts < max_attempts:
        attempts += 1
        world_type = _sample_world_type(world_weights, rng)
        target_turns = _sample_from_distribution(turn_distribution, rng)
        difficulty = max(1, min(5, overall_difficulty + rng.randint(-1, 1)))

        example = generate_example(
            world_type=world_type,
            target_turns=target_turns,
            difficulty=difficulty,
            rng=rng,
        )
        if example is None:
            continue
        while example.example_id in used_ids:
            example.example_id = rng.randint(0, 2**31 - 1)
        used_ids.add(example.example_id)
        examples.append(example)

    return examples


def save_dataset(examples: list[TrainingExample], path: str) -> None:
    """Save examples in TrainingEpisode JSONL form (one row = one episode)."""
    with open(path, "w") as f:
        for example in examples:
            f.write(example.to_jsonl() + "\n")


def save_metadata(examples: list[TrainingExample], path: str) -> None:
    """Save dataset metadata as JSON (world/turns/difficulty/template dists)."""
    metadata = {
        "total_examples": len(examples),
        "world_distribution": dict(Counter(e.world_type for e in examples)),
        "optimal_turns_distribution": {
            str(k): v
            for k, v in sorted(Counter(e.optimal_turns for e in examples).items())
        },
        "difficulty_distribution": {
            str(k): v
            for k, v in sorted(Counter(e.difficulty for e in examples).items())
        },
        "query_template_distribution": dict(
            Counter(e.query_template for e in examples)
        ),
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_dataset(path: str) -> list[TrainingExample]:
    """Load examples from a JSONL file produced by ``save_dataset``."""
    examples = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                examples.append(TrainingExample(**data))
    return examples


def export_for_verifiers(examples: list[TrainingExample], path: str) -> None:
    """Export dataset in verifiers-compatible row format
    (``prompt``/``answer``/``info``)."""
    with open(path, "w") as f:
        for example in examples:
            row = example.to_dataset_row()
            f.write(json.dumps(row) + "\n")
