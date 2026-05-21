#!/usr/bin/env python3
"""Build the default mixed context-management train/eval datasets."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ENV_ROOT = Path(__file__).resolve().parent.parent
if str(_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENV_ROOT))

from generators.adaptive_cursor import make_example as make_adaptive_cursor  # noqa: E402
from generators.corpus_trail import make_example as make_corpus_trail  # noqa: E402
from generators.dataset import export_for_verifiers, save_metadata  # noqa: E402
from scripts.build_corpus_trail import validate as validate_corpus_trail  # noqa: E402


HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "my_data"
TRAIN_SIZE = 8000
EVAL_SIZE = 800

# From-scratch curriculum: adaptive cursor is the on-ramp and dominant signal,
# while corpus trail introduces realistic search/synthesis pressure once the
# model starts earning variance under zero-gradient filtering.
FAMILY_MIX = [("adaptive_cursor", 0.60), ("corpus_trail", 0.40)]
ADAPTIVE_CURSOR_DIFFICULTY_MIX = [
    (0, 0.12),
    (1, 0.23),
    (2, 0.35),
    (3, 0.22),
    (4, 0.08),
]
CORPUS_TRAIL_DIFFICULTY_MIX = [
    (0, 0.12),
    (1, 0.30),
    (2, 0.40),
    (3, 0.14),
    (4, 0.04),
]


def _counts(n: int, mix: list[tuple[int | str, float]]) -> dict[int | str, int]:
    counts = {key: int(n * weight) for key, weight in mix}
    remainder = n - sum(counts.values())
    for key, _ in sorted(mix, key=lambda item: item[1], reverse=True):
        if remainder <= 0:
            break
        counts[key] += 1
        remainder -= 1
    return counts


def _expanded_difficulties(n: int, mix: list[tuple[int, float]]) -> list[int]:
    counts = _counts(n, mix)
    return [difficulty for difficulty, _ in mix for _ in range(int(counts[difficulty]))]


def _make_example(world: str, seed: int, difficulty: int):
    if world == "corpus_trail":
        return make_corpus_trail(seed, difficulty)
    if world == "adaptive_cursor":
        return make_adaptive_cursor(seed, difficulty)
    raise ValueError(f"unknown world: {world}")


def _build_family(
    world: str,
    n: int,
    seed: int,
    difficulty_mix: list[tuple[int, float]],
    used_ids: set[int],
) -> list:
    rng = random.Random(seed)
    difficulties = _expanded_difficulties(n, difficulty_mix)
    rows = []
    while len(rows) < n:
        difficulty = difficulties[len(rows)]
        ex = _make_example(world, rng.randint(0, 2**31 - 1), difficulty)
        while ex.example_id in used_ids:
            ex = _make_example(world, rng.randint(0, 2**31 - 1), difficulty)
        used_ids.add(ex.example_id)
        rows.append(ex)
    return rows


def build(n: int, seed: int) -> list:
    family_counts = _counts(n, FAMILY_MIX)
    used_ids: set[int] = set()
    rows = []
    rows.extend(
        _build_family(
            "corpus_trail",
            int(family_counts["corpus_trail"]),
            seed + 11,
            CORPUS_TRAIL_DIFFICULTY_MIX,
            used_ids,
        )
    )
    rows.extend(
        _build_family(
            "adaptive_cursor",
            int(family_counts["adaptive_cursor"]),
            seed + 29,
            ADAPTIVE_CURSOR_DIFFICULTY_MIX,
            used_ids,
        )
    )
    validate_corpus_trail([row for row in rows if row.world_type == "corpus_trail"])
    return _prefix_balanced_order(rows, seed + 47)


def _prefix_balanced_order(rows: list, seed: int) -> list:
    """Interleave buckets so small eval prefixes reflect the full mix."""
    rng = random.Random(seed)
    buckets: dict[tuple[str, int], list] = defaultdict(list)
    for row in rows:
        buckets[(row.world_type, row.difficulty)].append(row)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)

    target_counts = {key: len(bucket_rows) for key, bucket_rows in buckets.items()}
    used = {key: 0 for key in buckets}
    total = len(rows)
    ordered = []
    while len(ordered) < total:
        step = len(ordered) + 1
        available = [key for key, bucket_rows in buckets.items() if bucket_rows]
        key = max(
            available,
            key=lambda item: (target_counts[item] * step / total) - used[item],
        )
        ordered.append(buckets[key].pop())
        used[key] += 1
    return ordered


def _world_difficulty_counts(rows: list) -> dict[str, dict[int, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row.world_type][row.difficulty] += 1
    return {world: dict(sorted(counter.items())) for world, counter in sorted(counts.items())}


def write(rows: list, path: Path) -> None:
    export_for_verifiers(rows, str(path))
    save_metadata(rows, str(path.with_suffix(".metadata.json")))
    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["mix"] = {
        "families": dict(FAMILY_MIX),
        "corpus_trail_difficulty_mix": dict(CORPUS_TRAIL_DIFFICULTY_MIX),
        "adaptive_cursor_difficulty_mix": dict(ADAPTIVE_CURSOR_DIFFICULTY_MIX),
        "world_difficulty_distribution": _world_difficulty_counts(rows),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"{path.name}: {len(rows)} rows",
        "world", dict(Counter(r.world_type for r in rows)),
        "world_difficulty", _world_difficulty_counts(rows),
        "turns", dict(Counter(r.optimal_turns for r in rows)),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = build(TRAIN_SIZE, 20260518)
    eval_rows = build(EVAL_SIZE, 20260519)
    train_path = OUT_DIR / "train_context_mix.jsonl"
    eval_path = OUT_DIR / "eval_context_mix.jsonl"
    write(train, train_path)
    write(eval_rows, eval_path)
    print(
        json.dumps(
            {
                "train": str(train_path.relative_to(HERE)),
                "eval": str(eval_path.relative_to(HERE)),
                "note": "default 60% adaptive_cursor / 40% corpus_trail from-scratch context-management mix",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
