#!/usr/bin/env python3
"""CLI for generating context-tools training data."""

import argparse
from pathlib import Path

from generators import generate_dataset, save_dataset, save_metadata, export_for_verifiers


def main():
    parser = argparse.ArgumentParser(
        description="Generate context-tools training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d", "--difficulty",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=4,
        help="Difficulty level (1=easy, 5=hard)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("context_tools_data"),
        help="Output directory for generated files",
    )
    parser.add_argument(
        "-n", "--num-train",
        type=int,
        default=1000,
        help="Number of training examples",
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        default=100,
        help="Number of eval examples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Generate training data
    print(f"Generating {args.num_train} training examples (difficulty={args.difficulty})...")
    train = generate_dataset(
        num_examples=args.num_train,
        overall_difficulty=args.difficulty,
        seed=args.seed,
    )

    # Generate eval data
    print(f"Generating {args.num_eval} eval examples...")
    eval_ = generate_dataset(
        num_examples=args.num_eval,
        overall_difficulty=args.difficulty,
        seed=args.seed + 10000,
    )

    # Save files
    train_path = args.output / "train.jsonl"
    eval_path = args.output / "eval.jsonl"
    meta_path = args.output / "metadata.json"

    export_for_verifiers(train, str(train_path))
    export_for_verifiers(eval_, str(eval_path))
    save_metadata(train + eval_, str(meta_path))

    print(f"\nSaved to {args.output}/")
    print(f"  train.jsonl  ({len(train)} examples)")
    print(f"  eval.jsonl   ({len(eval_)} examples)")
    print(f"  metadata.json")


if __name__ == "__main__":
    main()
