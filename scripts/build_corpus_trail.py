#!/usr/bin/env python3
"""Build answer-first corpus_trail datasets.

The generated tasks are noisy research briefs over a document corpus. They are
answer-first: each example samples a final JSON brief, constructs a hidden
evidence DAG with durable facts, renders source documents plus distractors, and
stores only the rendered corpus in the sandbox state.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

_ENV_ROOT = Path(__file__).resolve().parent.parent
if str(_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENV_ROOT))

from generators.corpus_trail import make_example  # noqa: E402
from generators.dataset import export_for_verifiers, save_metadata  # noqa: E402


HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "my_data"
TRAIN_SIZE = 1000
EVAL_SIZE = 120

# Frontier mix with a real on-ramp for cr=true. d0/d1 are scaffold tiers that
# teach the corpus tool workflow before d2+ applies the neutral-document route
# and context-pressure shape.
DIFFICULTY_MIX = [(0, 0.10), (1, 0.25), (2, 0.45), (3, 0.15), (4, 0.05)]


def _counts(n: int) -> dict[int, int]:
    counts = {difficulty: int(n * weight) for difficulty, weight in DIFFICULTY_MIX}
    remainder = n - sum(counts.values())
    for difficulty, _ in sorted(DIFFICULTY_MIX, key=lambda item: item[1], reverse=True):
        if remainder <= 0:
            break
        counts[difficulty] += 1
        remainder -= 1
    return counts


def build(n: int, seed: int) -> list:
    rng = random.Random(seed)
    counts = _counts(n)
    difficulties = [
        difficulty
        for difficulty, _ in DIFFICULTY_MIX
        for _ in range(counts[difficulty])
    ]
    rows = []
    used = set()
    while len(rows) < n:
        difficulty = difficulties[len(rows)]
        ex = make_example(rng.randint(0, 2**31 - 1), difficulty)
        while ex.example_id in used:
            ex = make_example(rng.randint(0, 2**31 - 1), difficulty)
        used.add(ex.example_id)
        rows.append(ex)
    rng.shuffle(rows)
    validate(rows)
    return rows


def _tokens(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(t) >= 2 and t not in {"the", "and", "for", "with", "from"}
    ]


def _token_counts(text: str) -> Counter:
    return Counter(_tokens(text))


def _render_doc(doc: dict) -> str:
    return (
        f"Document {doc['id']}\n"
        f"Title: {doc['title']}\n"
        f"Date: {doc['date']}\n"
        f"Keywords: {', '.join(doc.get('keywords', []))}\n\n"
        f"{doc['body']}"
    )


def _search(docs: dict, query: str, limit: int = 6) -> list[str]:
    toks = _tokens(query)
    query_text = str(query).lower().strip()
    hits = []
    for doc_id, doc in docs.items():
        title_raw = str(doc.get("title", "")).lower()
        body_raw = str(doc.get("body", "")).lower()
        keywords_raw = " ".join(str(k).lower() for k in doc.get("keywords", []))
        title_tokens = _token_counts(doc.get("title", ""))
        body_tokens = _token_counts(doc.get("body", ""))
        keyword_tokens = _token_counts(" ".join(str(k) for k in doc.get("keywords", [])))
        id_tokens = _token_counts(doc_id)
        score = 0
        if query_text:
            if query_text in str(doc_id).lower():
                score += 12
            if query_text in title_raw:
                score += 10
            if query_text in keywords_raw:
                score += 8
            if query_text in body_raw:
                score += 6
        for tok in toks:
            if tok in id_tokens:
                score += 8
            if tok in title_tokens:
                score += 6
            if tok in keyword_tokens:
                score += 5
            score += min(4, body_tokens.get(tok, 0))
        if score:
            hits.append((score, str(doc.get("date", "")), doc_id))
    hits.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [doc_id for _, _, doc_id in hits[:limit]]


def validate(rows: list) -> None:
    failures = []
    for ex in rows:
        state = ex.state
        docs = state["docs"]
        answer = json.loads(str(ex.expected_answer))
        full_answer = state["_answer"]
        evidence = full_answer["evidence"]
        policy_text = _render_doc(docs[evidence[3]])
        ticket_text = _render_doc(docs[evidence[2]])
        risk_text = _render_doc(docs[evidence[4]])
        policy_match = re.search(r"POL-\d+", policy_text)
        ticket_match = re.search(r"TCK-\d+", ticket_text)
        if not policy_match or not ticket_match:
            failures.append((ex.example_id, "missing policy/ticket id"))
            continue
        answer_leak_markers = (
            "Evidence order",
            "Evidence cross-reference",
            "Identity evidence",
            "Owner/deadline evidence",
            "Blocker evidence",
            "Policy evidence",
            "Final memo evidence",
        )
        if any(marker in risk_text for marker in answer_leak_markers):
            failures.append((ex.example_id, "risk memo leaks evidence chain"))
            continue
        checks = [
            (evidence[0], full_answer["project"]),
            (evidence[1], full_answer["internal_code"]),
            (evidence[2], f"{full_answer['internal_code']} {full_answer['blocker']}"),
            (evidence[3], policy_match.group(0)),
            (evidence[4], f"{ticket_match.group(0)} {policy_match.group(0)}"),
        ]
        route_terms = state.get("_route_terms", {})
        if state.get("_difficulty", 0) >= 2 and route_terms:
            code = str(route_terms.get("internal_code") or full_answer["internal_code"])
            ticket = str(route_terms.get("ticket_id") or ticket_match.group(0))
            policy = str(route_terms.get("policy_id") or policy_match.group(0))
            route_checks = [
                (evidence[1], code),
                (evidence[2], code),
                (evidence[2], ticket),
                (evidence[4], ticket),
                (evidence[3], policy),
                (evidence[4], policy),
            ]
            for gold, query in route_checks:
                if gold not in _search(docs, query, 6):
                    failures.append((ex.example_id, "route term unreachable", gold, query))
                    break
        if state.get("_difficulty") == 0 and answer != evidence:
            failures.append((ex.example_id, "d0 expected answer should be evidence list"))
        if state.get("_difficulty") == 1 and set(answer) != {
            "project",
            "internal_code",
            "evidence",
        }:
            failures.append((ex.example_id, "d1 compact answer schema changed"))
        if state.get("_difficulty") == 2 and set(answer) != {
            "project",
            "internal_code",
            "owner",
            "deadline",
            "decision",
            "evidence",
        }:
            failures.append((ex.example_id, "d2 bridge answer schema changed"))
        for gold, query in checks:
            if gold not in _search(docs, query, 6):
                failures.append((ex.example_id, "unreachable", gold, query))
                break
        if state.get("_difficulty", 0) >= 2:
            initial_hits = _search(docs, full_answer["project"], 6)
            initial_gold_hits = [doc_id for doc_id in initial_hits if doc_id in evidence]
            if evidence[0] not in initial_hits:
                failures.append((ex.example_id, "initial project search misses identity"))
            if len(initial_gold_hits) > 2:
                failures.append(
                    (
                        ex.example_id,
                        "initial project search reveals too much gold evidence",
                        initial_gold_hits,
                    )
                )
        min_margin = 1.2 if state.get("_difficulty", 0) == 0 else 2.0
        if state.get("_append_only_margin", 0) < min_margin:
            failures.append((ex.example_id, "weak append-only margin"))
        cap = int(state.get("max_context_chars") or 1)
        if state.get("_compact_gold_chars", cap) > 0.65 * cap:
            failures.append((ex.example_id, "compact notes too large"))
        durable = state.get("_durable_facts") or []
        if not durable or max(
            f["needed_again_hop"] - f["first_used_hop"] for f in durable
        ) < 3:
            failures.append((ex.example_id, "missing durable fact lag"))
    if failures:
        raise RuntimeError(f"corpus_trail validation failed: {failures[:3]}")


def write(rows: list, path: Path) -> None:
    export_for_verifiers(rows, str(path))
    save_metadata(rows, str(path.with_suffix(".metadata.json")))
    print(
        f"{path.name}: {len(rows)} rows",
        "difficulty", dict(Counter(r.difficulty for r in rows)),
        "turns", dict(Counter(r.optimal_turns for r in rows)),
        "context_caps", dict(Counter(r.state.get("max_context_chars") for r in rows)),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = build(TRAIN_SIZE, 20260516)
    eval_rows = build(EVAL_SIZE, 20260517)
    train_path = OUT_DIR / "train_corpus_trail.jsonl"
    eval_path = OUT_DIR / "eval_corpus_trail.jsonl"
    write(train, train_path)
    write(eval_rows, eval_path)
    print(
        json.dumps(
            {
                "train": str(train_path.relative_to(HERE)),
                "eval": str(eval_path.relative_to(HERE)),
                "world": "corpus_trail",
                "note": "answer-first DAG corpus tasks with durable reusable facts and final-only exact JSON reward",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
