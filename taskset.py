"""ContextToolsTaskSet — minimal data wrapper.

Wraps the JSONL output of `generate.py` (rows in verifiers'
``prompt``/``answer``/``info`` shape produced by
``TrainingEpisode.to_dataset_row``) as a ``datasets.Dataset``. The env
(``ContextToolsEnv``) consumes ``taskset.get_dataset()`` directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset


def _load_jsonl(path: str | Path) -> Dataset:
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


class ContextToolsTaskSet:
    """Plain data wrapper over a JSONL of pre-built dataset rows."""

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        *,
        name: str = "context-tools",
    ):
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "my_data" / "eval.jsonl"
        self.name = name
        self._dataset = _load_jsonl(dataset_path)

    def get_instruction(self, info: dict) -> str:
        """Return the task question (used as ``context_window[0]``)."""
        questions = info.get("questions") or []
        if questions and isinstance(questions[0], dict):
            return questions[0].get("query_text", "") or ""
        return ""

    def get_dataset(self) -> Any:
        """Pre-built ``prompt``/``answer``/``info`` rows pass through unchanged."""
        return self._dataset
