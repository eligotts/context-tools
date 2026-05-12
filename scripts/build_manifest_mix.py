#!/usr/bin/env python3
"""Build a high-pressure manifest-management data mix.

This mix focuses on tasks where the REPL can hold rich durable state, but the
model still needs a compact visible manifest in ``context_window`` to resume:

* detective/top_k_by_region
* timeline_track/owned_transfer_top_k_at_time
* timeline_track/checkpoint_actor_audit

The generator uses the same world classes and ReferenceSolver as the regular
pipeline, but targets specific templates directly so the data mix is stable.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_ENV_ROOT = Path(__file__).resolve().parent.parent
if str(_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENV_ROOT))

from generators.base import Question, TrainingExample  # noqa: E402
from generators.dataset import export_for_verifiers, save_metadata  # noqa: E402
from generators.detective import DetectiveWorld  # noqa: E402
from generators.rule_hunt import RuleHuntWorld  # noqa: E402
from generators.solver import ReferenceSolver, SolverError  # noqa: E402
from generators.timeline_track import TimelineTrackWorld  # noqa: E402


HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "my_data"


SPEC = [
    ("detective", "top_k_by_region", 1, 120, 31_101),
    ("detective", "top_k_by_region", 2, 260, 31_202),
    ("detective", "top_k_by_region", 3, 240, 31_303),
    ("timeline_track", "owned_transfer_top_k_at_time", 1, 60, 33_101),
    ("timeline_track", "owned_transfer_top_k_at_time", 2, 120, 33_202),
    ("timeline_track", "owned_transfer_top_k_at_time", 3, 180, 33_303),
    ("timeline_track", "checkpoint_actor_audit", 1, 70, 35_101),
    ("timeline_track", "checkpoint_actor_audit", 2, 120, 35_202),
    ("timeline_track", "checkpoint_actor_audit", 3, 150, 35_303),
]

EVAL_SPEC = [
    ("detective", "top_k_by_region", 1, 8, 41_101),
    ("detective", "top_k_by_region", 2, 12, 41_202),
    ("detective", "top_k_by_region", 3, 12, 41_303),
    ("timeline_track", "owned_transfer_top_k_at_time", 1, 4, 43_101),
    ("timeline_track", "owned_transfer_top_k_at_time", 2, 6, 43_202),
    ("timeline_track", "owned_transfer_top_k_at_time", 3, 8, 43_303),
    ("timeline_track", "checkpoint_actor_audit", 1, 6, 45_101),
    ("timeline_track", "checkpoint_actor_audit", 2, 8, 45_202),
    ("timeline_track", "checkpoint_actor_audit", 3, 8, 45_303),
]


def _generator(world_type: str):
    if world_type == "rule_hunt":
        return RuleHuntWorld()
    if world_type == "detective":
        return DetectiveWorld()
    if world_type == "timeline_track":
        return TimelineTrackWorld()
    raise ValueError(world_type)


def _template(gen: Any, template_name: str):
    for template in gen.get_query_templates():
        if template.name == template_name:
            return template
    raise ValueError(f"{gen.world_type}: unknown template {template_name}")


def build_bucket(
    world_type: str,
    template_name: str,
    difficulty: int,
    n: int,
    seed: int,
) -> list[TrainingExample]:
    rng = random.Random(seed)
    gen = _generator(world_type)
    template = _template(gen, template_name)
    solver = ReferenceSolver(world_type)
    out: list[TrainingExample] = []
    attempts = 0
    max_attempts = max(500, n * 80)

    while len(out) < n and attempts < max_attempts:
        attempts += 1
        state = gen.generate_state(
            depth=max(3, difficulty + 1),
            breadth=max(4, difficulty + 3),
            rng=rng,
            difficulty=difficulty,
        )
        rich = gen.generate_query_rich(
            template=template,
            state=state,
            target_turns=10,
            rng=rng,
        )
        if rich is None:
            continue

        clean_state = {
            k: v for k, v in state.items()
            if not k.endswith("_by_depth") and not k.endswith("_by_level")
        }
        q = Question(
            template_name=template.name,
            query_text=rich["query_text"],
            expected_answer=str(rich["expected_answer"]),
            answer_type=template.answer_type,
            target_entities=list(rich.get("target_entities", [])),
            parameter_refs=list(rich.get("parameter_refs", [])),
        )
        try:
            solved = solver.solve(clean_state, q)
        except SolverError:
            continue
        if solved.answer != str(rich["expected_answer"]):
            continue

        out.append(
            TrainingExample(
                example_id=rng.randint(0, 2**31 - 1),
                world_type=world_type,
                system_prompt=gen.get_system_prompt(),
                user_query=rich["query_text"],
                state=clean_state,
                optimal_turns=solved.turn_count,
                expected_answer=rich["expected_answer"],
                answer_type=template.answer_type,
                difficulty=difficulty,
                depth=rich["actual_depth"],
                breadth=rich["actual_breadth"],
                query_template=template.name,
                target_entities=list(rich.get("target_entities", [])),
                parameter_refs=list(rich.get("parameter_refs", [])),
                rules=list(rich.get("rules", [])),
            )
        )

    if len(out) < n:
        raise RuntimeError(
            f"{world_type}/{template_name}/d{difficulty}: got {len(out)}/{n} "
            f"after {attempts} attempts"
        )
    return out


def write_mix(spec: list[tuple[str, str, int, int, int]], out_path: Path) -> list[TrainingExample]:
    rows: list[TrainingExample] = []
    for world, template, difficulty, n, seed in spec:
        bucket = build_bucket(world, template, difficulty, n, seed)
        rows.extend(bucket)
        opt = Counter(e.optimal_turns for e in bucket)
        print(
            f"{world:14s} {template:32s} d={difficulty}: {len(bucket):4d} "
            f"opt=[{min(opt)}..{max(opt)}]"
        )

    rng = random.Random(20260507)
    rng.shuffle(rows)
    export_for_verifiers(rows, str(out_path))
    save_metadata(rows, str(out_path.with_suffix(".metadata.json")))
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train_manifest_mix.jsonl"
    eval_path = OUT_DIR / "eval_manifest_mix.jsonl"
    print("=== train_manifest_mix ===")
    train = write_mix(SPEC, train_path)
    print("=== eval_manifest_mix ===")
    eval_rows = write_mix(EVAL_SPEC, eval_path)
    print(
        json.dumps(
            {
                "train": str(train_path.relative_to(HERE)),
                "train_rows": len(train),
                "eval": str(eval_path.relative_to(HERE)),
                "eval_rows": len(eval_rows),
                "train_templates": Counter(e.query_template for e in train),
                "eval_templates": Counter(e.query_template for e in eval_rows),
            },
            indent=2,
            default=dict,
        )
    )


if __name__ == "__main__":
    main()
