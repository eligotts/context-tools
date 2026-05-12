#!/usr/bin/env python3
"""Concatenate the 5 per-family train files into a single shuffled
``my_data/train_all.jsonl`` for training.

Idempotent: re-running with the same seed produces the same shuffled
output. The default ``load_environment`` reads ``train_all.jsonl``.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import json

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "my_data"

FAMILIES = [
    "rule_hunt",
    "corpus_dive",
    "timeline_track",
    "detective",
    "maze_walk",
]


def main() -> None:
    rows: list[str] = []
    per_family: Counter = Counter()
    for fam in FAMILIES:
        p = DATA / f"train_{fam}.jsonl"
        if not p.exists():
            print(f"  MISSING: {p}", file=sys.stderr)
            sys.exit(1)
        with open(p) as f:
            for line in f:
                if line.strip():
                    rows.append(line)
                    per_family[fam] += 1
        print(f"  {fam:14s}: {per_family[fam]} rows from {p.relative_to(HERE)}")

    rng = random.Random(42)
    rng.shuffle(rows)

    out = DATA / "train_all.jsonl"
    with open(out, "w") as f:
        for r in rows:
            if not r.endswith("\n"):
                r = r + "\n"
            f.write(r)

    # Quick sanity-check on the written file
    n_lines = sum(1 for _ in open(out))
    fams_in_file: Counter = Counter()
    for line in open(out):
        info = json.loads(line)["info"]
        fams_in_file[info.get("world_type", "?")] += 1

    print(f"\nWrote {n_lines} rows to {out.relative_to(HERE)}")
    print(f"Per-family in output: {dict(fams_in_file)}")


if __name__ == "__main__":
    main()
