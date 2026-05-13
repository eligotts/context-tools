#!/usr/bin/env python3
"""Build semantic-route adaptive-cursor datasets.

These examples avoid a manufactured small tool-call budget. ``observe(handle)``
is an ordinary Python function returning a page string; progress depends on
reading each page, updating ledger state, and choosing the correct candidate
tab from a semantic route rule.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

_ENV_ROOT = Path(__file__).resolve().parent.parent
if str(_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENV_ROOT))

from generators.adaptive_cursor import make_example  # noqa: E402
from generators.dataset import export_for_verifiers, save_metadata  # noqa: E402


HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "my_data"


def build(n: int, seed: int) -> list:
    rng = random.Random(seed)
    rows = []
    # Terminal-gated curriculum mix:
    #   d0       15%  protocol retention
    #   d1       45%  main 2-checkpoint learning zone
    #   d2-lite  30%  5-page bridge tasks with limited lifecycle churn
    #   d2-hard   8%  previous d2 shape
    #   d3        2%  hardest shape, kept rare until d2 is reliable
    mix = [(0, 0.15), (1, 0.45), (2, 0.30), (3, 0.08), (4, 0.02)]
    counts = {difficulty: int(n * weight) for difficulty, weight in mix}
    remainder = n - sum(counts.values())
    for difficulty, _ in sorted(mix, key=lambda item: item[1], reverse=True):
        if remainder <= 0:
            break
        counts[difficulty] += 1
        remainder -= 1
    difficulties = [
        difficulty
        for difficulty, _ in mix
        for _ in range(counts[difficulty])
    ]
    used = set()
    while len(rows) < n:
        difficulty = difficulties[len(rows)]
        ex = make_example(rng.randint(0, 2**31 - 1), difficulty)
        while ex.example_id in used:
            ex = make_example(rng.randint(0, 2**31 - 1), difficulty)
        used.add(ex.example_id)
        rows.append(ex)
    rng.shuffle(rows)
    return rows


def write(rows: list, path: Path) -> None:
    export_for_verifiers(rows, str(path))
    save_metadata(rows, str(path.with_suffix(".metadata.json")))
    print(
        f"{path.name}: {len(rows)} rows",
        "difficulty", dict(Counter(r.difficulty for r in rows)),
        "turns", dict(Counter(r.optimal_turns for r in rows)),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = build(600, 20260509)
    eval_rows = build(60, 20260510)
    train_path = OUT_DIR / "train_adaptive_cursor.jsonl"
    eval_path = OUT_DIR / "eval_adaptive_cursor.jsonl"
    write(train, train_path)
    write(eval_rows, eval_path)
    print(
        json.dumps(
            {
                "train": str(train_path.relative_to(HERE)),
                "eval": str(eval_path.relative_to(HERE)),
                "world": "adaptive_cursor",
                "note": "observe(handle) returns a page string; route choices are semantic, not harness boundaries",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
