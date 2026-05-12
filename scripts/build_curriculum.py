#!/usr/bin/env python3
"""Build the curriculum-round-1 training set for the 2 viable families.

Layout (1000 samples per family, diversity-stratified):

* **detective** (1000 rows): 500 at diff=1 + 500 at diff=2. Single template
  (``find_unique``); diversity comes from constraint set × entity pool
  randomization.
* **timeline_track** (1000 rows): balanced across difficulty AND query
  template — 250 each of:
    (diff=1, owner_at_time)  (diff=1, count_owned_at_time)
    (diff=2, owner_at_time)  (diff=2, count_owned_at_time)

A 20-row held-out ``sweep_<family>.jsonl`` per family is kept aside for
the cap-sweep evals.

Outputs:
  my_data/train_curriculum.jsonl       — 2000 rows, shuffled
  my_data/train_curriculum_<family>.jsonl
  my_data/sweep_<family>.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ENV_ROOT = Path(__file__).resolve().parent.parent
if str(_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENV_ROOT))

from generators.dataset import (  # noqa: E402
    DETECTIVE_WORLD_WEIGHTS,
    TIMELINE_TRACK_WORLD_WEIGHTS,
    export_for_verifiers,
    generate_dataset,
)


HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "my_data"

PER_FAMILY = 1000
SWEEP_SAMPLES = 20


def stratify_detective(seed_base: int) -> list:
    """1000 rows balanced across difficulty AND template.

    250 each of (diff=1, find_unique) (diff=1, count_matching)
                (diff=2, find_unique) (diff=2, count_matching).

    Both templates force multi-turn work at the new entity counts (50/70).
    count_matching additionally forces a *full* scan — no short-circuit on
    first match — which is the main fix preventing the "flail then one-shot"
    pattern observed at the old 25-entity find_unique-only setup.
    """
    out: list = []
    for d, seed, pool_size in [
        (1, seed_base + 1000, 2000),
        (2, seed_base + 2000, 3500),
    ]:
        pool = generate_dataset(
            num_examples=pool_size,
            overall_difficulty=d,
            world_weights=DETECTIVE_WORLD_WEIGHTS,
            seed=seed,
        )
        by_tmpl: dict[str, list] = defaultdict(list)
        for e in pool:
            if e.difficulty != d:
                continue
            by_tmpl[e.query_template].append(e)
        for tmpl in ["find_unique", "count_matching"]:
            bucket = by_tmpl.get(tmpl, [])
            if len(bucket) < 250:
                raise RuntimeError(
                    f"detective d={d} tmpl={tmpl}: only {len(bucket)}/250 — bump pool_size"
                )
            out.extend(bucket[:250])
    return out


def stratify_timeline_track(seed_base: int) -> list:
    """250 each of (diff=1, owner) (diff=1, count) (diff=2, owner) (diff=2, count).

    Need enough candidates to satisfy 250-per-bucket × 2 buckets per
    difficulty. Pool size scaled per difficulty level for jitter.
    """
    out: list = []
    for d, pool_size in [(1, 1500), (2, 2500)]:
        pool = generate_dataset(
            num_examples=pool_size,
            overall_difficulty=d,
            world_weights=TIMELINE_TRACK_WORLD_WEIGHTS,
            seed=seed_base + d * 10_000,
        )
        by_tmpl: dict[str, list] = defaultdict(list)
        for e in pool:
            if e.difficulty != d:
                continue
            by_tmpl[e.query_template].append(e)
        for tmpl in ["owner_at_time", "count_owned_at_time"]:
            bucket = by_tmpl.get(tmpl, [])
            if len(bucket) < 250:
                raise RuntimeError(
                    f"timeline_track d={d} tmpl={tmpl}: only {len(bucket)}/250"
                )
            out.extend(bucket[:250])
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build per-family training sets
    print("=== detective ===")
    det = stratify_detective(seed_base=31_000)
    diffs = Counter(e.difficulty for e in det)
    print(f"  {len(det)} rows  diffs={dict(sorted(diffs.items()))}")
    export_for_verifiers(det, str(OUT_DIR / "train_curriculum_detective.jsonl"))

    print("\n=== timeline_track ===")
    tt = stratify_timeline_track(seed_base=32_000)
    by_diff_tmpl = Counter((e.difficulty, e.query_template) for e in tt)
    print(f"  {len(tt)} rows")
    for k, v in sorted(by_diff_tmpl.items()):
        print(f"    diff={k[0]} tmpl={k[1]:25s}: {v}")
    export_for_verifiers(tt, str(OUT_DIR / "train_curriculum_timeline_track.jsonl"))

    # Build the unified shuffled training set
    print("\n=== unified train_curriculum.jsonl ===")
    rng = random.Random(20250504)
    raw_rows: list[str] = []
    for fam_path in [
        OUT_DIR / "train_curriculum_detective.jsonl",
        OUT_DIR / "train_curriculum_timeline_track.jsonl",
    ]:
        with open(fam_path) as f:
            for line in f:
                if line.strip():
                    raw_rows.append(line)
    rng.shuffle(raw_rows)
    out = OUT_DIR / "train_curriculum.jsonl"
    with open(out, "w") as f:
        for r in raw_rows:
            if not r.endswith("\n"):
                r = r + "\n"
            f.write(r)

    n_lines = sum(1 for _ in open(out))
    fams_in: Counter = Counter()
    diffs_in: Counter = Counter()
    for line in open(out):
        info = json.loads(line)["info"]
        fams_in[info.get("world_type", "?")] += 1
        diffs_in[info.get("difficulty", "?")] += 1
    print(f"  total: {n_lines} rows")
    print(f"  per-family:    {dict(fams_in)}")
    print(f"  per-difficulty: {dict(sorted(diffs_in.items()))}")


if __name__ == "__main__":
    main()
