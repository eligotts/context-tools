# context-tools

Sandboxed Python-REPL harness for training models to **manage their own context** across turns. The current default data mix focuses on adaptive-cursor tasks: ordinary REPL tools return cursor pages, and the model has to decide what returned information to keep, compact, overwrite, or discard in `context_window`.

## How it works

Each rollout gets a per-rollout Prime Sandbox running a long-lived Python worker subprocess (re-used from `RLMEnv`). The worker keeps a persistent namespace `dict` across calls; it's plain Python — no Jupyter kernel, no `ipykernel`. The worker uses `ast.parse → exec/eval` so a trailing expression's `repr` lands in the result dict (similar to IPython's `Out[N]`).

The model has one tool — `call_python_repl(code: str)` — and a per-rollout namespace pre-seeded with:

- The world's tool functions (e.g. `observe` / `get_entity` / `look` / `read_event` / etc.) bound to that rollout's hidden state.
- `submit_answer(value)` — terminates the rollout with a final answer.
- (`context_rewrite=True` only) `context_window: list` — the model-owned persistent memory across turns. The task text is shown separately; every `context_window` slot is hard-truncated under the same cap.

The toggle `context_rewrite` selects the prompting flavor:

- **`context_rewrite=True` (default).** Fresh `[system, user]` every turn; the user message renders the static task text plus the hard-truncated current `context_window` (plus the previous turn's code if it errored). The trajectory is never visible to the model.
- **`context_rewrite=False`.** Standard tool-calling flow — the model sees the full conversation history, each `call_python_repl` returns the truncated execution output as a normal tool message.

## Files

```
context_tools.py    ContextToolsEnv (subclasses RLMEnv) + load_environment
taskset.py          ContextToolsTaskSet (data wrapper)
generators/         deterministic, solver-verified data pipeline
scripts/            data-generation CLIs, including build_adaptive_cursor.py and build_corpus_trail.py
my_data/            default adaptive-cursor files plus optional corpus-trail train/eval JSONL files
```

## Task families

All have a single submitted answer per rollout, programmatically verifiable.

| Family | Tools | Scratchpad mechanic | Question shape |
|---|---|---|---|
| `rule_hunt` | `get_entity`, `test` | edit a hypothesis as evidence narrows | submit a parse-tree rule |
| `corpus_dive` | `list_keys`, `read_node` | prune mostly-noise observations | count/sum under a subtree path |
| `timeline_track` | `read_event`, `read_events` | overwrite mutating fixed-schema state | owner/count at time T |
| `detective` | `get_entity`, `query_attribute` | shrink a candidate set via elimination | unique entity satisfying all conjunctive constraints |
| `maze_walk` | `look`, `move` | push/pop discipline (path advance + backtrack) | navigate to goal, submit goal's secret |
| `adaptive_cursor` | `observe` | choose what returned cursor-page content to preserve | checkpoint ledger audit rows |
| `corpus_trail` | `search_docs`, `read_doc` | retain durable source-tagged facts across a noisy research DAG | structured project risk brief with evidence ids |

The default train/eval mix uses `adaptive_cursor/cursor_checkpoint_audit` across difficulties 0-4 with a `3/10/27/50/10` frontier curriculum: small d0/d1 anchors, d2 bridge tasks, d3 as the main pressure point, and a capped d4 tail. There is no small manufactured per-turn tool-call limit. In `context_rewrite=True`, `observe(handle)` is just an ordinary Python function returning a page string; the next prompt is only the hard-truncated render of whatever the model itself placed in `context_window`. The task pressure comes from keeping enough ownership/count state visible while avoiding raw-page append logs that overflow the cap.

`corpus_trail` is an answer-first research family. Each example samples a final JSON brief, constructs a hidden evidence DAG with reusable facts such as aliases and policy rules, renders that DAG into verbose source documents plus distractors, and seeds the REPL with a long `BRIEFING_DOC`. The model must search/read documents and keep compact notes because raw gold documents are several times larger than the per-example context cap. Unlike adaptive-cursor, corpus-trail uses final exact JSON correctness only; there is no partial process reward for this family.

Legacy families are solver-verified at generation time. `corpus_trail` is answer-first instead: the generator builds the answer and hidden evidence DAG first, then renders only the public documents; builder smoke checks verify that every gold evidence document is reachable through the exposed search terms and that compact gold notes fit while raw evidence overflows the per-example cap.

## Quickstart

```bash
prime env install context-tools

# Re-generate the default adaptive-cursor train/eval sets
python scripts/build_adaptive_cursor.py

# Generate the first corpus-trail research train/eval sets
python scripts/build_corpus_trail.py

# Smoke-test eval
prime eval run context-tools -m gpt-4.1-mini -n 5 -r 1
```

`PRIME_API_KEY` (for sandbox provisioning) is read automatically from `~/.prime/config.json` if not in the env. Provider key (e.g. `OPENAI_API_KEY`) you'll need to export yourself.

## Environment arguments

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `dataset_path` | str | `my_data/train_adaptive_cursor.jsonl` | Training JSONL (2,000 adaptive-cursor rows) |
| `eval_path` | str | `my_data/eval_adaptive_cursor.jsonl` | Held-out eval JSONL (200 rows) |
| `context_rewrite` | bool | `True` | True: model curates `context_window`. False: standard tool-calling flow. |
| `max_turns` | int | 15 | Max rollout turns |
| `max_context_chars` | int | 400 | Display cap on rendered model-curated `context_window` slots (cr=True) / per-tool-response cap (cr=False) |
| `max_code_display_chars` | int | 4000 | Display cap on echoed previous code (cr=True only) |
| `show_previous_code` | bool | `False` | (cr=True only) If True, echo prior code every turn; default echoes only on error |
| `tool_call_budget_per_turn` | int | 1000000 | No small manufactured tool-call cap by default; adaptive-cursor progress is gated by semantic route choices in ordinary returned page strings |
| `sandbox_docker_image` | str | `python:3.11-slim` | Sandbox image |
| `code_execution_timeout` | int | 120 | Per-turn code timeout (seconds) |
| `sandbox_cpu_cores` | int | 1 |  |
| `sandbox_memory_gb` | int | 2 |  |
| `sandbox_timeout_minutes` | int | 30 | Hard sandbox lifetime cap |
| `retain_filesystem_after_rollout` | bool | `False` | Keep `/rlm_fs` for post-mortem |

## Rewards

| Reward | Weight | Definition |
|---|---|---|
| `task_reward` | 1.0 | Capped at 1.0. Exact submitted answer gets 1.0. For adaptive-cursor misses, partial credit is terminal-gated: before the correct terminal page is observed, reward is 0. After terminal, partial credit is `0.05 * complete_valid_submit + 0.10 * checkpoint_ids_in_order + 0.85 * submitted_checkpoint_row_fraction`. |
| `correctness_reward` | 0.0 | Exact-answer metric only. |
| `checkpoint_row_submit_fraction` | 0.0 | Metric: exact gold checkpoint rows present in the submitted answer. |
| `checkpoint_row_context_fraction` | 0.0 | Metric: exact gold checkpoint rows visible in the final hard-truncated `context_window`. |
| `valid_checkpoint_submit` | 0.0 | Metric: submitted answer is a non-empty list of 4-field rows. |
| `complete_checkpoint_submit` | 0.0 | Metric: submitted answer has one valid row per expected checkpoint. |
| `checkpoint_ids_in_order` | 0.0 | Metric: submitted rows use the expected checkpoint ids in order. |
| `adaptive_terminal_reached` | 0.0 | Metric: the correct adaptive-cursor terminal page was observed. |

Additional diagnostic metrics include append/edit counts, dynamic overwrite/remove counts, final manifest character count, truncation count, and turn efficiency.

## Data generation invariants

- **100% synthetic, 100% verifiable**: ground truth is a deterministic function of the generated state.
- **100% solvable from observations**: legacy ground truth is recomputed independently before each example is emitted; answer-first corpus tasks are emitted only when their gold evidence docs are reachable through the public search/read surface.
- **Single answer per task**: every example terminates with one `submit_answer(...)` call.
- **Difficulty stratified**: default training set is stratified across difficulties 0-4 for the adaptive-cursor template, with the d1/d2-lite bridge emphasized.
