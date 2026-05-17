---
name: context-tools-calibration
description: Calibrate context-tools domains with Prime evals. Use when assessing a new or changed context_tools data domain, comparing context_rewrite true/false behavior, tuning max turns or context caps, checking whether tasks force non-append context management, or preparing/verifying Prime eval commands for this repository.
---

# Context Tools Calibration

## Purpose

Use this protocol to decide whether a `context_tools` domain is well calibrated for RL training. The goal is not just a high score: the domain should be easy enough when full history is available, hard enough under `context_rewrite=True`, and impossible to solve reliably by raw append-only memory or one-shot code.

## Prime Setup

Prefer the newer local Prime CLI when `prime` is not on `PATH`:

```bash
PRIME=/Users/eligottlieb/Documents/prime/.venv/bin/prime
```

Prime eval installation shells out to `uv`. If `uv` is missing from `PATH`,
restore a local copy and prepend the repo venv for eval commands:

```bash
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -q uv
export PATH="/Users/eligottlieb/dev/lab/environments/context_tools/.venv/bin:$PATH"
```

Confirm auth without printing secrets:

```bash
$PRIME whoami
```

For Qwen thinking-off evals, pass sampling args exactly through the Prime eval CLI:

```bash
-S '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'
```

For direct Pinference smoke checks, `chat_template_kwargs` is top-level request body JSON:

```bash
curl -X POST https://api.pinference.ai/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PRIME_API_KEY" \
  -d '{"model":"Qwen/Qwen3.5-35B-A3B:or87qviuv7jbhd5i84amqofj","messages":[{"role":"user","content":"Reply with exactly: ok"}],"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

Do not run calibration evals until the user has authorized that step.

## Eval Matrix

Run each domain against the untrained/base Qwen model with `context_rewrite=false`:

```bash
$PRIME eval run context-tools \
  -m Qwen/Qwen3.5-35B-A3B \
  -n 30 -r 4 \
  --env-args '{"dataset_path":"my_data/train_DOMAIN.jsonl","eval_path":"my_data/eval_DOMAIN.jsonl","context_rewrite":false,"max_turns":15}' \
  -S '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --save-results --plain
```

Target: pass@1 above 0.50 and pass@4 close to 1.0. If this fails badly, the domain is probably unclear, too hard, incorrectly formatted, or not reliably solvable.

For `corpus_trail`, set `max_context_chars` high enough in the `cr=false`
baseline, normally around the largest per-example `cr=true` cap such as `1800`.
In standard mode this value caps rendered REPL output, not the model-owned
`context_window`; leaving it at the default `400` can make source documents look
hard because `read_doc(...)` output is clipped even though full conversation
history is available.

Current calibrated `corpus_trail` recipe, as of 2026-05-17:

- train/eval files: `my_data/train_corpus_trail.jsonl` and
  `my_data/eval_corpus_trail.jsonl`
- mix: d0 3%, d1 24%, d2 55%, d3 15%, d4 3%
- d2 is the bridge tier: all five evidence documents are still required, but
  the final schema is `project, internal_code, owner, deadline, decision,
  evidence`
- d3/d4 use the full risk brief schema with blocker included
- per-example caps: d0 950, d1 1050, d2 1650, d3 1750, d4 1850
- measured anchors on an 8-example slice:
  base `Qwen/Qwen3.5-35B-A3B`, `cr=false`, `-r 4`: pass@1 0.969, pass@4 1.0
  checkpoint `Qwen/Qwen3.5-35B-A3B:or87qviuv7jbhd5i84amqofj`, `cr=true`,
  `-r 8`: pass@1 0.422, pass@8 1.0

In the calibrated checkpoint run, successes had much less raw appending and
truncation than failures: mean truncation about 3.3 vs 8.5, append count about
10.4 vs 25.8, and final manifest chars about 12.2k vs 31.0k. This is the
desired pressure: the task is not impossible, but raw append-heavy traces lose
state and fail.

Then run the trained checkpoint with `context_rewrite=true`:

```bash
$PRIME eval run context-tools \
  -m Qwen/Qwen3.5-35B-A3B:or87qviuv7jbhd5i84amqofj \
  -n 30 -r 8 \
  --env-args '{"dataset_path":"my_data/train_DOMAIN.jsonl","eval_path":"my_data/eval_DOMAIN.jsonl","context_rewrite":true,"max_turns":15}' \
  -S '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --save-results --plain
```

Target: low pass@1 and meaningfully high pass@8. This means the trained model can sometimes discover and execute the right memory strategy, but the task still has enough difficulty and variance to produce learning signal.

For `cr=true`, use enough rollouts per example to measure capability. A single failed rollout only proves the first sampled strategy failed; calibration needs to know whether the checkpoint can solve at least some examples when it explores. Prefer `-r 8` for the checkpoint once plumbing is confirmed.

When a domain is near the frontier, increase rollouts per sample before making broad data changes. The target shape is not `cr=true` zero reward; it is low pass@1 with nonzero pass@8, meaning the task is inside the trained checkpoint's capability but still requires the right context-management behavior to be sampled and reinforced.

Use smaller smoke slices first, for example `-n 3 -r 3`, only to verify plumbing. Do not treat smoke results as calibration.

For checkpoint `cr=true` plumbing checks, prefer a bounded first probe:

```bash
$PRIME eval run context-tools \
  -m Qwen/Qwen3.5-35B-A3B:or87qviuv7jbhd5i84amqofj \
  -n 1 -r 1 -c 1 -t 1024 --timeout 240 \
  --env-args '{"dataset_path":"my_data/train_DOMAIN.jsonl","eval_path":"my_data/eval_DOMAIN.jsonl","context_rewrite":true,"max_turns":15}' \
  -S '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --save-results --plain --disable-tui
```

Unbounded checkpoint `cr=true` smoke runs can be slow even at `-n 3 -r 3` because failed rollouts may spend all 15 turns producing large code blocks. Once plumbing is confirmed, remove the small `-n/-r` and adjust `-t` intentionally for calibration.

## Trace Inspection

Always inspect saved rollouts by eye before judging a domain:

```bash
$PRIME eval get EVAL_ID --output json --plain
$PRIME eval samples EVAL_ID --output json --plain --page 1 --num 100
```

Local saved traces live under `outputs/evals/.../results.jsonl` when `--save-results` is set. These are often easier for quick grep/JSON slicing than the hub payload.

Look for these properties:

- The rollout must require adaptive observation. It should not be one-shottable by a single code block that already knows the full route.
- Successful `cr=true` traces should show compact, intentional `context_window` writes: durable facts, source ids, route handles, partial aggregates, or rewritten state.
- Raw append-only traces should overflow or lose essential evidence under the truncation cap.
- A success after many wasted turns is not enough. Within `max_turns=15`, the task should require steady progress and visible intermediate decisions.
- The task text should be separate from `context_window` unless the user explicitly asks to test a different prompting regime.

After every eval, inspect both successes and failures by eye before changing data. Classify failures into concrete causes: retrieval route missed, wrong distractor selected, durable fact forgotten, final schema/format mistake, answer guessed, too many raw appends, context overwritten badly, max turns reached after real progress, or max turns reached after loops/no progress. Make targeted data changes that address the observed cause; do not blindly scale all difficulty up or down.

Use trace inspection to decide the next edit. Examples: if many failures read all required evidence but do not submit, clarify submit timing or reduce turn pressure; if failures miss the same evidence type, add better retrieval cues for that evidence; if successes are one-shot or mostly guessed, expand the answer space and add delayed fact reuse; if all `cr=true` samples fail before meaningful progress, add easier buckets while preserving the same task shape.

Do not change the core environment scaffolding to induce behavior: keep the single task description plus separately rendered `context_window` regime intact, and do not add new behavioral prompt nudges to the generic system/template text. Prefer edits to the generated data distribution, answer schema, seeded variables/functions, document/corpus construction, context caps, and solver-verified reachability.

Do not seed the task into `context_window`, do not make task text a managed memory slot, and do not rely on prompt wording to teach the memory policy. Domain prompts may define tools, variables, answer schemas, and source semantics; they should not instruct the model to compact, summarize, rewrite, or otherwise manage context in a specific strategy.

## Calibration Decisions

If base Qwen with `cr=false` is below target, make the domain easier or clearer: reduce hops/noise, increase context cap, simplify final schema, clarify final answer format, or fix generation/verifier bugs.

If trained checkpoint with `cr=true` has high pass@1, make the domain harder: add branching, delayed reuse of facts, distractors, larger raw observations, or tighter context caps.

If trained checkpoint with `cr=true` has low pass@8, add an on-ramp: include easier difficulty buckets, fewer hops, clearer retrieval cues, or slightly larger caps while preserving the same task shape.

If append-only succeeds, increase the raw evidence-to-cap margin or require durable facts that are needed again several steps later.

If rollouts guess final answers, enlarge the answer space and ensure final answers are not inferable from ids, handles, ordering, or template regularities.

## Invariants

Keep `max_turns` at 15 unless the user explicitly asks otherwise. Calibrate difficulty around that limit.

Prefer final correctness as the main reward for new realistic research/search domains. Use process rewards only when they are simple, terminal-gated, and reward genuinely useful progress rather than easy-to-hack artifacts.

For future domains, generate answer-first or solver-verified data. The public tool surface must contain enough evidence to solve the task, and hidden generator metadata must not be available through the sandbox state.
