#!/usr/bin/env python3
"""Build a 500-per-family stratified-difficulty training set.

For each of the 5 context-management families we generate 100 examples per
difficulty level (1-5) and write a single per-family JSONL. The dataset.py
``generate_dataset`` jitters difficulty ±1 around its ``overall_difficulty``
arg, so we strictly filter to the requested difficulty after generation.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

# Make the script cwd-independent: prepend the env's root directory so the
# top-level ``generators`` package can be imported regardless of where this
# script was launched from.
_ENV_ROOT = Path(__file__).resolve().parent.parent
if str(_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENV_ROOT))

from generators.dataset import (
    CORPUS_DIVE_WORLD_WEIGHTS,
    DETECTIVE_WORLD_WEIGHTS,
    MAZE_WALK_WORLD_WEIGHTS,
    RULE_HUNT_WORLD_WEIGHTS,
    TIMELINE_TRACK_WORLD_WEIGHTS,
    export_for_verifiers,
    generate_dataset,
    save_metadata,
)


HERE = Path(__file__).parent.parent
OUT_DIR = HERE / "my_data"

FAMILIES = [
    ("rule_hunt",      RULE_HUNT_WORLD_WEIGHTS,      11_000),
    ("corpus_dive",    CORPUS_DIVE_WORLD_WEIGHTS,    12_000),
    ("timeline_track", TIMELINE_TRACK_WORLD_WEIGHTS, 13_000),
    ("detective",      DETECTIVE_WORLD_WEIGHTS,      14_000),
    ("maze_walk",      MAZE_WALK_WORLD_WEIGHTS,      15_000),
]

PER_DIFFICULTY = 100  # 100 × 5 difficulties = 500 per family


def build_one_family(label: str, weights: dict, base_seed: int) -> list:
    """Return ~500 stratified examples for one family."""
    out = []
    for d in [1, 2, 3, 4, 5]:
        bucket: list = []
        attempt = 0
        # Each attempt generates 150 candidates; we keep only those whose
        # final difficulty == d. Loop until we have 100 (or give up).
        while len(bucket) < PER_DIFFICULTY and attempt < 6:
            attempt += 1
            t0 = time.time()
            candidates = generate_dataset(
                num_examples=200,
                overall_difficulty=d,
                world_weights=weights,
                seed=base_seed + d * 1000 + attempt,
            )
            kept_before = len(bucket)
            for ex in candidates:
                if ex.difficulty == d and len(bucket) < PER_DIFFICULTY:
                    bucket.append(ex)
            print(
                f"  {label:14s} d={d} attempt={attempt} "
                f"+{len(bucket) - kept_before}/{PER_DIFFICULTY - kept_before} "
                f"in {time.time() - t0:.1f}s "
                f"(have {len(bucket)}/{PER_DIFFICULTY})"
            )
        if len(bucket) < PER_DIFFICULTY:
            print(
                f"  WARNING: {label} d={d} only got {len(bucket)}/"
                f"{PER_DIFFICULTY} after {attempt} attempts"
            )
        out.extend(bucket)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_t0 = time.time()
    summary: list[dict] = []
    for label, weights, base_seed in FAMILIES:
        print(f"\n=== {label} ===")
        t0 = time.time()
        exs = build_one_family(label, weights, base_seed)
        out_path = OUT_DIR / f"train_{label}.jsonl"
        export_for_verifiers(exs, str(out_path))
        meta_path = OUT_DIR / f"train_{label}.metadata.json"
        save_metadata(exs, str(meta_path))

        diffs = Counter(e.difficulty for e in exs)
        opts = Counter(e.optimal_turns for e in exs)
        print(
            f"  → {len(exs)} examples in {time.time() - t0:.1f}s, "
            f"diffs={dict(sorted(diffs.items()))}, "
            f"opt_turns_range=[{min(opts)}..{max(opts)}]"
        )
        summary.append(
            {
                "family": label,
                "n": len(exs),
                "difficulty_distribution": {str(k): v for k, v in sorted(diffs.items())},
                "optimal_turns_distribution": {str(k): v for k, v in sorted(opts.items())},
                "out_path": str(out_path.relative_to(HERE)),
            }
        )

    print(f"\n=== Total wall time: {time.time() - overall_t0:.1f}s ===")
    summary_path = OUT_DIR / "train_500_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
