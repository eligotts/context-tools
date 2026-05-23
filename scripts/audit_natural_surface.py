#!/usr/bin/env python3
"""Audit packaged active-domain data for stale synthetic visible ids."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent.parent

PACKAGED_DATASETS = [
    HERE / "my_data" / "train_context_mix.jsonl",
    HERE / "my_data" / "eval_context_mix.jsonl",
    HERE / "my_data" / "train_adaptive_cursor.jsonl",
    HERE / "my_data" / "eval_adaptive_cursor.jsonl",
    HERE / "my_data" / "train_corpus_trail.jsonl",
    HERE / "my_data" / "eval_corpus_trail.jsonl",
]

SOURCE_FILES = [
    HERE / "context_tools.py",
    HERE / "generators" / "adaptive_cursor.py",
    HERE / "generators" / "corpus_trail.py",
]

FORBIDDEN = {
    "checkpoint_code": re.compile(r"\bCP\d+\b"),
    "old_start_helper": re.compile(r"\bSTART_HANDLE\b"),
    "old_briefing_helper": re.compile(r"\bBRIEFING_DOC\b"),
    "old_doc_ids_helper": re.compile(r"\bDOC_IDS\b"),
    "old_doc_count_helper": re.compile(r"\bDOC_COUNT\b"),
    "old_policy_ref": re.compile(r"\bPOL-\d+\b"),
    "old_ticket_ref": re.compile(r"\bTCK-\d+\b"),
    "old_doc_ref": re.compile(
        r"\b(?:doc|memo|risk|ticket|handoff|policy)_\d+\b"
    ),
    "old_internal_code": re.compile(r"\b[A-Z]{2}-\d{3}\b"),
    "old_handle": re.compile(r"\bH(?=[A-Z0-9]*\d)[A-Z0-9]{3,}\b"),
    "old_read_doc_arg": re.compile(r"\bread_doc\(doc_id\)"),
}

SLUG = re.compile(r"^[a-z]+-[a-z]+(?:-[a-z]+)?$")


def _public_state(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _public_state(child)
            for key, child in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_public_state(child) for child in value]
    return value


def _scan_text(text: str, label: str, failures: list[str]) -> None:
    for name, pattern in FORBIDDEN.items():
        match = pattern.search(text)
        if match:
            start = max(0, match.start() - 80)
            end = min(len(text), match.end() + 80)
            excerpt = text[start:end].replace("\n", "\\n")
            failures.append(f"{label}: {name}: {excerpt}")


def _scan_row(path: Path, line_no: int, row: dict[str, Any], failures: list[str]) -> None:
    info = row.get("info") or {}
    world = info.get("world_type")
    visible_parts: list[Any] = [
        row.get("prompt"),
        row.get("answer"),
        info.get("expected_answer"),
        info.get("questions"),
    ]
    try:
        state = json.loads(info.get("state") or "{}")
    except Exception as exc:
        failures.append(f"{path}:{line_no}: invalid state json: {exc}")
        state = {}
    visible_parts.append(_public_state(state))
    _scan_text(
        json.dumps(visible_parts, ensure_ascii=False),
        f"{path}:{line_no}",
        failures,
    )

    expected = json.loads(info.get("expected_answer") or row.get("answer") or "null")
    if world == "adaptive_cursor":
        if not isinstance(expected, list):
            failures.append(f"{path}:{line_no}: adaptive answer is not a list")
            return
        for row_idx, answer_row in enumerate(expected):
            if not isinstance(answer_row, list) or len(answer_row) != 4:
                failures.append(
                    f"{path}:{line_no}: adaptive row {row_idx} is malformed"
                )
                continue
            mark = answer_row[0]
            if not isinstance(mark, str) or not SLUG.match(mark):
                failures.append(
                    f"{path}:{line_no}: adaptive row {row_idx} has non-natural mark {mark!r}"
                )
    elif world == "corpus_trail":
        docs = state.get("docs") or {}
        if isinstance(expected, list):
            evidence = expected
        elif isinstance(expected, dict):
            evidence = expected.get("evidence") or []
        else:
            failures.append(f"{path}:{line_no}: corpus answer has unknown schema")
            evidence = []
        for idx, source_id in enumerate(evidence):
            if not isinstance(source_id, str) or not SLUG.match(source_id):
                failures.append(
                    f"{path}:{line_no}: evidence {idx} has non-natural id {source_id!r}"
                )
            if source_id not in docs:
                failures.append(
                    f"{path}:{line_no}: evidence {idx} missing source {source_id!r}"
                )


def main() -> None:
    failures: list[str] = []
    for source_file in SOURCE_FILES:
        _scan_text(source_file.read_text(), str(source_file.relative_to(HERE)), failures)
    for path in PACKAGED_DATASETS:
        with path.open() as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    _scan_row(path.relative_to(HERE), line_no, json.loads(line), failures)
    if failures:
        print(f"natural surface audit failed: {len(failures)} issue(s)")
        for failure in failures[:80]:
            print(failure)
        raise SystemExit(1)
    print("natural surface audit passed")


if __name__ == "__main__":
    main()
