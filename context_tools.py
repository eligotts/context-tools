"""context-tools — sandboxed Python-REPL harness with two prompting modes.

Each rollout gets its own Prime Sandbox running a long-lived Python worker
(re-used from ``RLMEnv``) that maintains a persistent namespace ``dict``
across calls. The model has one tool — ``call_python_repl(code: str)`` —
and a per-rollout namespace pre-seeded with the world's tool functions
(e.g. ``get_task_deps``) bound to this rollout's ``world_state`` via
closure, plus ``submit_answer(value)`` to terminate the rollout.

The toggle ``context_rewrite`` selects how the model sees its own history.

**``context_rewrite=True`` (default — context-management mode).** The boot
block additionally seeds ``context_window: list`` in the kernel; this list is
the model-owned persistent memory across turns. The task text is rendered
separately as static problem statement, and every character of the model-owned
``context_window`` render is subject to the same hard cap. Items can be any
Python object; the renderer uses ``repr`` for non-strings. Every turn the
model sees a fresh ``[system, user]`` prompt; the user message re-renders the
hard-truncated ``context_window`` (plus the previous turn's code and error, if
it raised). No tool messages, no growing trajectory. The model alone decides
what survives across turns by writing to ``context_window``.

**``context_rewrite=False`` (standard tool-calling mode).** The boot block
does NOT seed ``context_window``. The standard ToolEnv flow is used: the
model calls ``call_python_repl`` like a normal tool, the framework
dispatches it, the tool response (stdout / stderr / ``Out[N]:`` / error,
truncated) is appended as a tool message, and the trajectory accumulates
naturally. The full conversation history is what the model sees on each
turn — the model has no special scratchpad to manage.

In both modes the harness inherits ``RLMEnv`` so the worker subprocess,
FIFO IPC, and sandbox lifecycle come from there unchanged. The worker uses
``ast.parse → exec/eval`` to give a slightly nicer REPL feel (trailing
expression evaluates and its repr lands in the result dict, the way
IPython displays ``Out[N]``), but it's a plain Python subprocess — there
is no Jupyter kernel, no ``ipykernel`` package, no kernel protocol. Just a
persistent ``dict`` namespace plus FIFO IPC.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
from pathlib import Path
from typing import Any

import tenacity as tc
from datasets import Dataset

import verifiers as vf
from verifiers.envs.experimental.rlm_env import (
    RLMEnv,
    RLMExecutor,
    _render_worker_script,
)
from verifiers.envs.experimental.sandbox_mixin import (
    is_retryable_sandbox_read_error,
)
from verifiers.types import State, SystemMessage, UserMessage
from verifiers.utils.message_utils import concat_messages

from taskset import ContextToolsTaskSet

logger = logging.getLogger(__name__)


def _is_retryable_transfer_error(exc: BaseException) -> bool:
    """Treat sandbox gateway 408s as transient transfer errors.

    The upstream RLM retry predicate already handles read timeouts,
    upload/download timeout types, and a few API 5xx cases. Hosted training can
    also surface sandbox-gateway request timeouts as a generic APIError string
    containing HTTP 408. Control-file uploads are idempotent, so retrying them
    is the right failure mode.
    """
    if is_retryable_sandbox_read_error(exc):
        return True
    text = str(exc)
    return any(
        token in text
        for token in (
            "408",
            "Request Timeout",
            "504",
            "Gateway Timeout",
            "429",
            "Too Many Requests",
        )
    )


class _LeanContextToolsExecutor(RLMExecutor):
    """RLM executor variant tuned for this tiny no-network REPL environment.

    Upstream RLM setup assumes root-tool/sub-LLM support and pays per-rollout
    costs we do not need here: installing ``requests`` and tar/extracting a
    tiny filesystem context. Hosted training can hit sandbox gateway 408s when
    doing that across ~100 rollouts at once, so this executor keeps setup
    idempotent and small without changing the Python REPL semantics.
    """

    def __init__(self, env: "ContextToolsEnv") -> None:
        self.env = env
        self._sessions: dict[str, Any] = {}
        self._retained_dirs: set[str] = set()
        self.init_sandbox_client(
            sandbox_client_max_workers=env.sandbox_client_max_workers,
            sandbox_client_max_connections=env.sandbox_client_max_connections,
            sandbox_client_max_keepalive_connections=(
                env.sandbox_client_max_keepalive_connections
            ),
        )

    async def _install_packages(self, session: Any) -> None:
        # No root tools/sub-LLMs are exposed, so the worker never needs the
        # HTTP helper dependency. Avoid a per-rollout network install.
        return None

    async def _upload_directory(
        self, sandbox_id: str, local_dir: str, remote_dir: str
    ) -> None:
        local_path = Path(local_dir)
        files = sorted(path for path in local_path.rglob("*") if path.is_file())
        if not files:
            return

        remote_root = remote_dir.rstrip("/")
        parent_dirs = sorted(
            {
                f"{remote_root}/{path.parent.relative_to(local_path).as_posix()}"
                for path in files
                if path.parent != local_path
            }
        )
        if parent_dirs:
            mkdir_cmd = "mkdir -p " + " ".join(shlex.quote(p) for p in parent_dirs)
            await self.env.with_retry_on_read_errors(self._execute_sandbox_command)(
                sandbox_id,
                f"bash -lc {shlex.quote(mkdir_cmd)}",
                timeout=self.env.max_startup_wait_seconds,
            )

        upload_file = self.env.with_retry_on_read_errors(
            self.sandbox_client.upload_file
        )
        for path in files:
            rel = path.relative_to(local_path).as_posix()
            await upload_file(sandbox_id, f"{remote_root}/{rel}", str(path))

    async def recover_from_timeout(self, state: State) -> bool:
        """Restart the worker after a code timeout.

        The adaptive-cursor boot block deliberately sanitizes the remote
        context.json after reading it so the model cannot dump the whole cursor
        table from the filesystem. A worker restart needs the original staged
        filesystem uploaded again before boot, otherwise observe() comes back
        with an empty page table.
        """
        session = self._sessions.get(state.get("rollout_id", ""))
        if not session or not session.sandbox_id or not session.paths:
            logger.error("Cannot recover from timeout: missing sandbox session")
            return False
        try:
            await self._stop_worker(session)
            sandbox_fs_root = session.sandbox_fs_root or state.get("rlm_fs_root_remote")
            if sandbox_fs_root:
                await self._upload_directory(
                    session.sandbox_id,
                    session.local_fs_root,
                    sandbox_fs_root,
                )
            await self._write_sandbox_files(session, state)
            await self._start_worker(session, state)
        except Exception as e:
            logger.error(f"Failed to recover from code timeout: {e}")
            return False
        state["rlm_worker_ready"] = True
        state["_exec_seq"] = 0
        return True

    async def _write_sandbox_files(self, session: Any, state: State) -> None:
        assert session.paths is not None
        context = {
            "fs_root": state.get("rlm_fs_root_remote") or state.get("rlm_fs_root"),
        }
        context_path = Path(session.local_control_dir) / "rlm_context.json"
        answer_path = Path(session.local_control_dir) / "rlm_answer.json"
        worker_path = Path(session.local_control_dir) / "rlm_worker.py"

        context_path.write_text(json.dumps(context), encoding="utf-8")
        answer_path.write_text(
            json.dumps({"ready": False, "content": ""}), encoding="utf-8"
        )

        worker_script = _render_worker_script(
            session.paths,
            repl_language=self.env.repl_language,
        )
        worker_script = self.env.customize_worker_script(worker_script, state)
        worker_path.write_text(worker_script, encoding="utf-8")

        upload_file = self.env.with_retry_on_read_errors(
            self.sandbox_client.upload_file
        )
        await upload_file(
            session.sandbox_id, session.paths.context_file, str(context_path)
        )
        await upload_file(
            session.sandbox_id, session.paths.answer_file, str(answer_path)
        )
        await upload_file(
            session.sandbox_id, session.paths.worker_path, str(worker_path)
        )


# =============================================================================
# Per-world tool definitions
# =============================================================================
#
# Each world has 2-3 read-only tool functions. We define them in two places:
#
#   - _TOOL_BOOT_BLOCKS[world] — Python source defining the functions over a
#     `_world_state` closure. exec'd into the worker namespace at boot.
#   - _TOOL_SIGNATURES[world]  — pretty-printed signature lines for the
#     system prompt the model sees.
#
# Adding a new world means adding entries to both dicts (and a generator
# under generators/).

# Names of the tool functions defined in each world's boot block. Used by
# `customize_worker_script` to wrap every seeded tool with the per-turn budget
# enforcer.
_TOOL_NAMES_PER_WORLD: dict[str, list[str]] = {
    "rule_hunt": ["get_entity", "test"],
    "corpus_dive": ["list_keys", "read_node"],
    "timeline_track": ["read_event", "read_events"],
    "detective": ["get_entity", "query_attribute"],
    "maze_walk": ["look", "move"],
    "corpus_trail": ["search_docs", "read_doc"],
    # adaptive_cursor intentionally leaves observe unwrapped. There is no small
    # manufactured per-turn tool budget for this family, and wrapping would put
    # the observe object in an easy-to-inspect Python closure.
    "adaptive_cursor": [],
}


_TOOL_BOOT_BLOCKS: dict[str, str] = {
    "rule_hunt": """
def get_entity(entity_id):
    \"\"\"Return the attributes of `entity_id` (a dict).\"\"\"
    entities = _world_state.get('entities', {})
    if entity_id not in entities:
        raise KeyError(f\"Unknown entity: {entity_id!r}\")
    return dict(entities[entity_id])

def test(entity_id):
    \"\"\"Return the hidden rule's label for `entity_id` (True/False).\"\"\"
    labels = _world_state.get('labels', {})
    if entity_id not in labels:
        raise KeyError(f\"Unknown entity: {entity_id!r}\")
    return bool(labels[entity_id])

# Pre-seeded helpers (free; not budget-counted because they are plain
# Python objects, not function calls into the seeded tool layer).
ENTITY_IDS = list(_world_state.get('entities', {}).keys())
ATTRIBUTE_SCHEMA = dict(_world_state.get('attribute_schema', {}))
""",
    "corpus_dive": """
def list_keys(path):
    \"\"\"Return the children names of the node at `path` (relative names,
    not full paths). Children are descended via f\"{path}.{child}\".\"\"\"
    tree = _world_state.get('tree', {})
    if path not in tree:
        raise KeyError(f\"Unknown path: {path!r}\")
    return list(tree[path].get('children', []))

def read_node(path):
    \"\"\"Return the attributes of the node at `path` (NOT including
    children data).\"\"\"
    tree = _world_state.get('tree', {})
    if path not in tree:
        raise KeyError(f\"Unknown path: {path!r}\")
    return dict(tree[path].get('attrs', {}))

ROOT_PATH = _world_state.get('root_path', 'root')
""",
    "timeline_track": """
def read_event(i):
    \"\"\"Return the event at index `i` (0-indexed).\"\"\"
    events = _world_state.get('events', [])
    if not isinstance(i, int) or i < 0 or i >= len(events):
        raise IndexError(f\"Event index out of range: {i}\")
    return dict(events[i])

def read_events(start, end):
    \"\"\"Return events in [start, end). Capped at 5 events per call: if
    end - start > 5, only the first 5 are returned. Index `end` is
    exclusive.\"\"\"
    events = _world_state.get('events', [])
    if start is None or start < 0:
        start = 0
    if end is None or end > len(events):
        end = len(events)
    capped_end = min(end, start + 5)
    if capped_end <= start:
        return []
    return [dict(e) for e in events[start:capped_end]]

# Pre-seeded helpers (free; not budget-counted because they are plain
# Python objects).
N_EVENTS = len(_world_state.get('events', []))
OBJECTS = list(_world_state.get('objects', []))
ACTORS = list(_world_state.get('actors', []))
LOCATIONS = list(_world_state.get('locations', []))
""",
    "detective": """
def get_entity(entity_id):
    \"\"\"Return the full attribute dict of `entity_id`.\"\"\"
    entities = _world_state.get('entities', {})
    if entity_id not in entities:
        raise KeyError(f\"Unknown entity: {entity_id!r}\")
    return dict(entities[entity_id])

def query_attribute(entity_id, attr_name):
    \"\"\"Return a single attribute value for `entity_id`. Useful when
    you only need to test one constraint at a time without paying for
    the rest of the dossier in your context.\"\"\"
    entities = _world_state.get('entities', {})
    if entity_id not in entities:
        raise KeyError(f\"Unknown entity: {entity_id!r}\")
    e = entities[entity_id]
    if attr_name not in e:
        raise KeyError(f\"Entity {entity_id!r} has no attribute {attr_name!r}\")
    return e[attr_name]

# Pre-seeded helpers (free; not budget-counted because they are plain
# Python objects).
ENTITY_IDS = list(_world_state.get('entities', {}).keys())
ATTRIBUTE_SCHEMA = dict(_world_state.get('attribute_schema', {}))
CONSTRAINTS = list(_world_state.get('constraints', []))
""",
    "maze_walk": """
# Pull goal + secrets out of the public _world_state so the model has
# to navigate to learn them. The tools read them via these private
# kernel-level names (which the model could in principle introspect,
# but the public _world_state does not expose them).
_goal_node = _world_state.pop('goal', None)
_node_secrets = _world_state.pop('node_secrets', {})

# Mutable kernel state: the model's current position in the graph.
_current_position = _world_state.get('start')

def look():
    \"\"\"Inspect the current node without moving. Returns a dict with:
        at: str          — current node id
        secret: str      — this node's secret string
        neighbors: list  — adjacent node ids (valid move() targets)
        is_goal: bool    — True if this node is the goal
    \"\"\"
    pos = _current_position
    graph = _world_state.get('graph', {})
    return {
        'at': pos,
        'secret': _node_secrets.get(pos, ''),
        'neighbors': list(graph.get(pos, [])),
        'is_goal': pos == _goal_node,
    }

def move(target):
    \"\"\"Move to a neighbor of the current node. Raises ValueError if
    `target` is not adjacent to the current position. Returns the same
    shape as look() for the new position.\"\"\"
    global _current_position
    pos = _current_position
    graph = _world_state.get('graph', {})
    neighbors = graph.get(pos, [])
    if target not in neighbors:
        raise ValueError(
            f\"{target!r} is not adjacent to current position {pos!r}; \"
            f\"neighbors are {neighbors}\"
        )
    _current_position = target
    return {
        'at': target,
        'secret': _node_secrets.get(target, ''),
        'neighbors': list(graph.get(target, [])),
        'is_goal': target == _goal_node,
    }

# Pre-seeded helpers (free; not budget-counted).
START = _world_state.get('start')
N_NODES = len(_world_state.get('graph', {}))
""",
    "corpus_trail": """
_CORPUS_TRAIL_DOCS = dict(_world_state.get('docs', {}))
BRIEFING_DOC = _world_state.get('briefing_doc', '')
DOC_IDS = list(_world_state.get('doc_ids', sorted(_CORPUS_TRAIL_DOCS.keys())))
DOC_COUNT = len(DOC_IDS)
try:
    _CORPUS_TRAIL_SNIPPET_CHARS = int(_world_state.get('search_snippet_chars', 180) or 180)
except Exception:
    _CORPUS_TRAIL_SNIPPET_CHARS = 180
_CORPUS_TRAIL_SNIPPET_CHARS = max(60, min(220, _CORPUS_TRAIL_SNIPPET_CHARS))

def _corpus_trail_tokens(text):
    import re as _re
    return [
        t for t in _re.findall(r"[a-z0-9]+", str(text).lower())
        if len(t) >= 2 and t not in {"the", "and", "for", "with", "from"}
    ]

def _corpus_trail_token_counts(text):
    counts = {}
    for tok in _corpus_trail_tokens(text):
        counts[tok] = counts.get(tok, 0) + 1
    return counts

def _corpus_trail_render(doc):
    keywords = ", ".join(doc.get('keywords', []))
    return (
        f"Document {doc.get('id', '')}\\n"
        f"Title: {doc.get('title', '')}\\n"
        f"Date: {doc.get('date', '')}\\n"
        f"Keywords: {keywords}\\n\\n"
        f"{doc.get('body', '')}"
    )

def _corpus_trail_snippet(doc, tokens):
    doc_id = str(doc.get('id', ''))
    title = str(doc.get('title', ''))
    body = str(doc.get('body', ''))
    keywords = " ".join(str(k) for k in doc.get('keywords', []))
    haystacks = (
        ('id', _corpus_trail_token_counts(doc_id)),
        ('title', _corpus_trail_token_counts(title)),
        ('metadata', _corpus_trail_token_counts(keywords)),
        ('body', _corpus_trail_token_counts(body)),
    )
    matched_in = []
    for label, token_counts in haystacks:
        if any(tok in token_counts for tok in tokens):
            matched_in.append(label)
    title_lower = title.lower()
    doc_lower = doc_id.lower()
    if doc_lower.startswith('risk_'):
        kind = 'risk memo'
    elif doc_lower.startswith('memo_'):
        kind = 'alias registry'
    elif doc_lower.startswith('handoff_'):
        kind = 'handoff note'
    elif doc_lower.startswith('ticket_'):
        kind = 'ticket note'
    elif doc_lower.startswith('policy_'):
        kind = 'policy note'
    elif 'alias' in title_lower:
        kind = 'alias registry'
    elif 'handoff' in title_lower:
        kind = 'handoff note'
    elif 'policy' in title_lower:
        kind = 'policy note'
    elif 'risk' in title_lower:
        kind = 'risk memo'
    elif 'ticket' in title_lower:
        kind = 'ticket note'
    elif doc_lower.startswith('email_'):
        kind = 'historical email'
    else:
        kind = 'corpus document'
    where = ', '.join(dict.fromkeys(matched_in)) if matched_in else 'corpus'
    return (
        f"Matched query terms in {where}; source kind: {kind}. "
        f"Open with read_doc('{doc_id}') for source text."
    )

def search_docs(query, limit=6):
    \"\"\"Search corpus documents. Returns up to `limit` dicts with id, title,
    date, and a short snippet. Snippets are not source-of-record; call
    read_doc(doc_id) for documents you rely on.\"\"\"
    query_text = str(query).lower().strip()
    tokens = _corpus_trail_tokens(query)
    if not tokens:
        return []
    try:
        limit = int(limit)
    except Exception:
        limit = 6
    limit = max(1, min(10, limit))
    hits = []
    for doc_id, doc in _CORPUS_TRAIL_DOCS.items():
        title_raw = str(doc.get('title', '')).lower()
        body_raw = str(doc.get('body', '')).lower()
        keywords_raw = " ".join(str(k).lower() for k in doc.get('keywords', []))
        title_tokens = _corpus_trail_token_counts(doc.get('title', ''))
        body_tokens = _corpus_trail_token_counts(doc.get('body', ''))
        keyword_tokens = _corpus_trail_token_counts(" ".join(str(k) for k in doc.get('keywords', [])))
        id_tokens = _corpus_trail_token_counts(doc_id)
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
        for tok in tokens:
            if tok in id_tokens:
                score += 8
            if tok in title_tokens:
                score += 6
            if tok in keyword_tokens:
                score += 5
            score += min(4, body_tokens.get(tok, 0))
        if score:
            hits.append((score, str(doc.get('date', '')), doc_id, doc))
    hits.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [
        {
            'id': doc_id,
            'doc_id': doc_id,
            'title': doc.get('title', ''),
            'date': doc.get('date', ''),
            'snippet': _corpus_trail_snippet(doc, tokens),
        }
        for _, _, doc_id, doc in hits[:limit]
    ]

def read_doc(doc_id):
    \"\"\"Return the full text of one corpus document by id.\"\"\"
    key = str(doc_id)
    if key not in _CORPUS_TRAIL_DOCS:
        raise KeyError(f"Unknown document id: {key!r}")
    return _corpus_trail_render(_CORPUS_TRAIL_DOCS[key])
""",
    "adaptive_cursor": """
START_HANDLE = _world_state.get('start_handle', 'START')


class _AdaptiveObserve:
    \"\"\"Callable observe tool with the page table out of obvious function
    defaults/closures. This is not a Python security boundary; it just avoids
    handing the model an easy `observe.__defaults__` dump of the whole cursor
    graph.
    \"\"\"

    __slots__ = ('__pages', '__start', '__terminal_reached')

    def __init__(self, pages, start):
        object.__setattr__(self, '_AdaptiveObserve__pages', dict(pages))
        object.__setattr__(self, '_AdaptiveObserve__start', str(start))
        object.__setattr__(self, '_AdaptiveObserve__terminal_reached', False)

    def __getattribute__(self, name):
        if name in {
            '__dict__', '__slots__', '_pages', '_start', '__pages', '__start',
            '_AdaptiveObserve__pages', '_AdaptiveObserve__start',
            '_AdaptiveObserve__terminal_reached',
        }:
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    def __call__(self, handle):
        \"\"\"Return the cursor page for an opaque handle as a string.

        This is an ordinary Python callable. It does not mutate context_window
        and it does not end the code block. If you need to see the page next
        turn, write the returned value, or a compact summary of it, to
        context_window yourself.
        \"\"\"
        pages = object.__getattribute__(self, '_AdaptiveObserve__pages')
        start = object.__getattribute__(self, '_AdaptiveObserve__start')
        key = str(handle)
        if key == 'START_HANDLE':
            key = str(start)
        if key not in pages:
            raise KeyError(f\"Unknown observation handle: {key!r}\")
        page = pages[key]
        text = page.get('text', page) if isinstance(page, dict) else page
        if isinstance(text, str) and 'Terminal slip: no next tab.' in text:
            object.__setattr__(self, '_AdaptiveObserve__terminal_reached', True)
        return text


observe = _AdaptiveObserve(_world_state.get('pages', {}), START_HANDLE)

# Do not leave the full cursor table in ordinary visible sandbox state. The
# task should be progressed through observe(handle), not by dumping _world_state
# or reading context.json.
_world_state = {'start_handle': START_HANDLE}
for _adaptive_ctx_path_str in ('context.json', _os.path.join(_os.getcwd(), 'context.json'), '/rlm_fs/context.json'):
    try:
        _adaptive_ctx_path = _Path(_adaptive_ctx_path_str)
        if _adaptive_ctx_path.exists():
            with open(_adaptive_ctx_path, 'r') as _adaptive_f:
                _adaptive_ctx = _json.load(_adaptive_f)
            if isinstance(_adaptive_ctx, dict) and isinstance(_adaptive_ctx.get('world_state'), dict):
                _adaptive_ctx['world_state'] = {'start_handle': START_HANDLE}
                with open(_adaptive_ctx_path, 'w') as _adaptive_f:
                    _json.dump(_adaptive_ctx, _adaptive_f)
    except Exception:
        pass
globals().pop('_AdaptiveObserve', None)
""",
}

_TOOL_SIGNATURES: dict[str, list[str]] = {
    "rule_hunt": [
        "- get_entity(entity_id: str) -> dict — attributes of `entity_id` (e.g. {'color': 'red', 'size': 42, 'tags': ['rare'], ...}).",
        "- test(entity_id: str) -> bool — the hidden rule's label for `entity_id`.",
    ],
    "corpus_dive": [
        "- list_keys(path: str) -> list[str] — children names (relative, not full paths) of the node at `path`. To descend, build f\"{path}.{child}\".",
        "- read_node(path: str) -> dict — attributes of the node at `path` (NOT including children data; e.g. {'color': 'red', 'size': 42, 'value': 17, 'kind': 'beta', 'tags': ['rare']}).",
    ],
    "timeline_track": [
        "- read_event(i: int) -> dict — the event at index `i` (0-indexed). Each event has a `type` and `time`, plus type-specific fields (see addendum).",
        "- read_events(start: int, end: int) -> list[dict] — events in [start, end). **Capped at 5 events per call** — requesting more returns only the first 5.",
    ],
    "detective": [
        "- get_entity(entity_id: str) -> dict — full attribute dossier of `entity_id` (all attrs at once).",
        "- query_attribute(entity_id: str, attr_name: str) -> value — single attribute value (cheaper to log than a full dict if you only need to test one constraint).",
    ],
    "maze_walk": [
        "- look() -> dict — inspect the current node without moving (returns {at, secret, neighbors, is_goal}).",
        "- move(target: str) -> dict — move to a neighbor of the current node; raises ValueError if `target` isn't adjacent. Returns the same shape as look() for the new position.",
    ],
    "corpus_trail": [
        "- search_docs(query: str, limit: int = 6) -> list[dict] — search noisy corpus documents and return matching ids, titles, dates, and short snippets.",
        "- read_doc(doc_id: str) -> str — return the full text of one corpus document.",
    ],
    "adaptive_cursor": [
        "- observe(handle: str) -> str — return the cursor page for an opaque handle. It does not mutate context_window; save whatever you need to context_window yourself.",
    ],
}


# Per-world prompt addendum: appended to the system prompt for that world
# only. Use for world-specific submission formats, pre-seeded variables, or
# DSL grammars that don't fit the generic tool-signature line format.
_WORLD_PROMPT_ADDENDUM: dict[str, str] = {
    "rule_hunt": """\

# Rule grammar (rule_hunt only)

The hidden rule is a parse tree. Atoms have shape:
  {"op": "==" | "!=" | "<" | ">" | "<=" | ">=" | "contains",
   "attr": <name>, "value": <v>}
(`contains` works on the `tags` list-attribute: True iff `value` is in
the entity's tags list.)

Combinators:
  {"op": "AND", "args": [<rule>, <rule>, ...]}
  {"op": "OR",  "args": [<rule>, <rule>, ...]}
  {"op": "NOT", "args": [<rule>]}

Pre-seeded kernel variables (free; do NOT cost a tool call):
  ENTITY_IDS         — list of all entity ids in the probe set
  ATTRIBUTE_SCHEMA   — dict mapping each attr name to {type, values|range}

Submit by calling submit_answer(rule_dict). Reward is 1.0 iff your rule
produces the same labels as the hidden rule on a held-out test set the
model never sees directly.
""",
    "corpus_dive": """\

# Path handling (corpus_dive only)

The world is a tree. ``list_keys(path)`` returns *relative* child names
(e.g. ``['a1b', 'f7q']``), NOT full paths. To descend, build the next
path as ``f"{path}.{child}"``.

Pre-seeded kernel variable (free; do NOT cost a tool call):
  ROOT_PATH — the path of the root node (typically ``"root"``).

Submit an integer via ``submit_answer(n)`` — a count or a sum, depending
on the question.
""",
    "timeline_track": """\

# Timeline format (timeline_track only)

Events are indexed 0..N_EVENTS-1. Each event has a ``type`` and ``time``
field, plus type-specific fields:

  CREATE   {object, owner, location, time}   — object comes into existence
  TRANSFER {object, from, to, time}          — owner change
  MOVE     {object, from, to, time}          — location change
  DESTROY  {object, time}                    — object ceases to exist

"At time T" in a question means **after the first T events have been
applied** — i.e. the state reflects events with index ``< T``. Events
with index ``>= T`` have not happened yet.

Pre-seeded kernel variables (free; do NOT cost a tool call):
  N_EVENTS    — total number of events in the timeline
  OBJECTS     — list of object ids that ever appear
  ACTORS      — list of actor names
  LOCATIONS   — list of location names

``read_events(start, end)`` is capped at 5 events per call. To scan the
timeline you'll typically issue several calls per turn (up to the per-
turn budget), then commit a compact summary to ``context_window``.
""",
    "detective": """\

# Constraint format (detective only)

Each constraint is a Python dict ``{"op": <op>, "attr": <name>,
"value": <v>}``. Operators: ``==``, ``!=``, ``<``, ``>``, ``<=``,
``>=``, ``contains`` (for the ``tags`` list-attribute: True iff
``value`` is in the entity's tags list).

Pre-seeded kernel variables (free; do NOT cost a tool call):
  ENTITY_IDS         — list of all entity ids in the pool
  ATTRIBUTE_SCHEMA   — dict mapping each attr to ``{type, values|range}``
  CONSTRAINTS        — the constraint list (Python list of dicts)

Exactly ONE entity in the pool satisfies all constraints; every other
entity violates at least one. Submit the unique entity id via
``submit_answer("<id>")``.
""",
    "maze_walk": """\

# Maze format (maze_walk only)

You are at node ``START`` in a connected graph. Each node has neighbors
(other nodes you can ``move`` to) and an opaque ``secret`` string. The
goal is one specific node; when you reach it, ``is_goal`` is True in
the dict returned by ``look()`` / ``move()``. The goal's ``secret``
is the answer.

Movement rules:
- ``move(target)`` requires ``target`` to be a neighbor of the current
  position; otherwise it raises ``ValueError``. The kernel tracks your
  current position; you can only step one edge at a time.
- ``look()`` is a no-move inspection; it returns the same shape as
  ``move`` but without changing your position.

Pre-seeded kernel variables (free; do NOT cost a tool call):
  START      — id of the starting node
  N_NODES    — total number of nodes in the graph

Submit the goal node's secret via ``submit_answer("<secret>")``.
""",
    "corpus_trail": """\

# Corpus trail format (corpus_trail only)

You are doing a small research synthesis over a noisy document corpus.
``BRIEFING_DOC`` is a long pre-seeded intake note. It is useful for starting
clues, but it is not a citable final evidence source. Search results are only
snippets; use ``read_doc(doc_id)`` for any source you rely on.

Pre-seeded kernel variables (free; do NOT cost a tool call):
  BRIEFING_DOC — long intake note with initial aliases and routing hints
  DOC_IDS      — list of document ids
  DOC_COUNT    — number of documents

Submit the JSON value requested by the task via ``submit_answer(value)``.
Some corpus_trail tasks ask for an ordered list; others ask for a JSON object
with named keys. Do not cite ``BRIEFING_DOC`` as evidence.
""",
    "adaptive_cursor": """\

# Adaptive cursor format (adaptive_cursor only)

You begin with ``START_HANDLE``. Call ``observe(START_HANDLE)`` to get the first
page string. Each page contains ledger facts plus, unless it is terminal, a
route rule and two candidate tabs. The correct next handle is the tab whose
actor label matches the route rule after you interpret the page and update the
ledger state. Handles are opaque word ids and are not enumerable.

``observe(handle)`` is an ordinary Python function: it returns a string, does
not mutate ``context_window``, and does not stop the code block. In
``context_rewrite=True`` mode you will not see that return value next turn
unless your code writes it, or a compact summary of it, to ``context_window``.
Raw pages are verbose and will overflow the visible cap if you keep appending
them.

Efficient loop: read the current raw page from ``context_window``, update
durable REPL state, overwrite/remove the raw page, keep only compact state or a
short REPL-variable breadcrumb, then fetch the next page with
``page = observe(next_handle)`` and decide what part of it to place in
``context_window`` for the next turn.

Submit the final JSON value via ``submit_answer(value)`` after processing the
terminal page.

Checkpoint fields are intentionally actor-local:
- The first field of every row must be the exact checkpoint string id with the
  ``CP`` prefix, for example ``"CP1"`` or ``"CP10"``. Do not submit a bare
  number such as ``1`` or ``10``. A valid row looks like
  ``["CP1", "Alice", 2, 1]``.
- ``interval_transfer_count`` is the winning actor's receipt count since the
  previous checkpoint, not the total transfer count for the interval.
- ``owned_count_at_checkpoint`` is the number of live objects owned by that
  same reported actor at the checkpoint, not the total number of live objects.
- A ``TRANSFER`` receipt is counted for the receiver. Created/opened/first-
  holder objects do not add transfer receipts.
- Destroyed, closed, voided, and left-ledger objects are not live and count for
  nobody unless a later page creates/opens them again.
""",
}

_SUBMIT_ANSWER_SIGNATURE = (
    "- submit_answer(value: int | str | bool | float | list | dict) -> None — "
    "submit your final answer and terminate the rollout."
)


# =============================================================================
# Boot block — exec'd into the worker namespace before the FIFO loop starts
# =============================================================================
#
# The worker's namespace dict already has `answer = {"ready": False, ...}`
# when this runs. We add `_world_state`, the world's tool functions,
# `submit_answer`, and (only when ``context_rewrite=True``) the
# ``context_window`` list.
#
# The ``{context_window_seed_block}`` placeholder receives either the line
# that seeds ``context_window = []`` (rewrite=True) or an
# empty string (rewrite=False). Everything else is identical between modes.

_BOOT_CODE_TEMPLATE = """\
# === context-tools boot block (injected into worker namespace) ===
import json as _json
import os as _os
import reprlib as _reprlib
from pathlib import Path as _Path

_context_tools_repr = _reprlib.Repr()
_context_tools_repr.maxstring = 2000
_context_tools_repr.maxother = 2000
_context_tools_repr.maxlist = 80
_context_tools_repr.maxtuple = 80
_context_tools_repr.maxdict = 80
_context_tools_repr.maxset = 80


def _context_tools_json_safe(value, _depth=0, _seen=None):
    if _seen is None:
        _seen = set()
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value
    if _depth >= 8:
        try:
            return _context_tools_repr.repr(value)
        except Exception:
            return f"<unrepresentable {{type(value).__name__}}>"
    obj_id = id(value)
    if obj_id in _seen:
        return "<recursive>"
    if isinstance(value, dict):
        _seen.add(obj_id)
        try:
            out = {{}}
            for k, v in list(value.items())[:80]:
                safe_key = _context_tools_json_safe(k, _depth + 1, _seen)
                out[str(safe_key)] = _context_tools_json_safe(v, _depth + 1, _seen)
            return out
        finally:
            _seen.discard(obj_id)
    if isinstance(value, (list, tuple)):
        _seen.add(obj_id)
        try:
            return [
                _context_tools_json_safe(v, _depth + 1, _seen)
                for v in list(value)[:80]
            ]
        finally:
            _seen.discard(obj_id)
    if isinstance(value, (set, frozenset)):
        _seen.add(obj_id)
        try:
            items = [
                _context_tools_json_safe(v, _depth + 1, _seen)
                for v in list(value)[:80]
            ]
            return sorted(items, key=repr)
        finally:
            _seen.discard(obj_id)
    try:
        return _context_tools_repr.repr(value)
    except Exception:
        return f"<unrepresentable {{type(value).__name__}}>"

# RLMEnv chdirs the worker to fs_root before our boot block runs, so the
# context.json that RLMEnv wrote there is right next to us. Try a few path
# candidates so we're robust to the worker's startup-cwd changing later.
_world_state = {{}}
_initial_user_text = ''
_ctx_candidates = [
    'context.json',
    _os.path.join(_os.getcwd(), 'context.json'),
    '/rlm_fs/context.json',  # absolute fallback if cwd ever changes
]
for _ctx_path_str in _ctx_candidates:
    _ctx_path = _Path(_ctx_path_str)
    if _ctx_path.exists():
        with open(_ctx_path, 'r') as _f:
            _ctx = _json.load(_f)
        _world_state = _ctx.get('world_state', {{}})
        _initial_user_text = _ctx.get('initial_user_text', '')
        break

# --- Per-turn tool budget --------------------------------------------------
# Every seeded tool function is wrapped with `_budget_wrap`. The counter
# `_tool_call_count` is reset to 0 by the env at the start of every
# call_python_repl exec (see `customize_worker_script`). When the budget is
# hit, ToolBudgetExhausted is raised — any prior assignments persist in the
# kernel, so context_window appends made before the exception still survive.
_TOOL_CALL_BUDGET = {budget}
_tool_call_count = 0


class ToolBudgetExhausted(RuntimeError):
    \"\"\"Raised when the per-turn tool-call budget is hit. End this turn — a
    fresh budget is granted on the next turn.\"\"\"
    pass


def _budget_wrap(fn):
    def wrapped(*args, **kwargs):
        global _tool_call_count
        if _tool_call_count >= _TOOL_CALL_BUDGET:
            raise ToolBudgetExhausted(
                f"Tool budget exhausted ({{_TOOL_CALL_BUDGET}} tool calls/turn). "
                "End this turn; you'll get a fresh budget on the next turn."
            )
        _tool_call_count += 1
        result = fn(*args, **kwargs)
        try:
            wrapped._context_tools_terminal_reached = bool(
                object.__getattribute__(fn, '_AdaptiveObserve__terminal_reached')
            )
        except Exception:
            pass
        return result
    wrapped.__name__ = getattr(fn, '__name__', 'wrapped')
    wrapped.__doc__ = getattr(fn, '__doc__', None)
    wrapped._context_tools_terminal_reached = False
    return wrapped

# --- World tool functions --------------------------------------------------

{tool_defs}

# --- Wrap each seeded tool with the budget enforcer ------------------------

{tool_wrap_lines}

for _ctx_tmp_name in (
    '_ctx', '_ctx_path', '_ctx_path_str', '_ctx_candidates',
    '_adaptive_pages', '_adaptive_ctx', '_adaptive_ctx_path',
    '_adaptive_ctx_path_str',
):
    globals().pop(_ctx_tmp_name, None)


def submit_answer(value):
    \"\"\"Submit your final answer and terminate the rollout.

    submit_answer does NOT count against the tool budget. It is robust to
    the model accidentally rebinding the namespace name ``answer`` to
    something else — we forcibly reassign ``answer`` to a fresh submission
    dict here.

    Primitive values (str/int/float/bool) are stringified directly. Non-
    primitive values (dicts, lists, etc.) are JSON-serialized so they can
    be parsed back losslessly by the rubric. If JSON serialization fails,
    we fall back to ``repr`` (which the rubric can parse with
    ``ast.literal_eval``).\"\"\"
    global answer
    if value is None:
        _content = ''
    elif isinstance(value, (str, int, float, bool)):
        _content = str(value)
    else:
        try:
            _content = _json.dumps(value)
        except Exception:
            _content = repr(value)
    answer = {{'ready': True, 'content': _content}}

{context_window_seed_block}
# === end boot block ===
"""

# Boot-block fragment that seeds ``context_window``. Used only when
# ``context_rewrite=True``. The task text is rendered separately by the env;
# the model-owned scratchpad starts empty so every slot is governed by the same
# hard truncation behavior. In ``context_rewrite=False`` mode this slot is
# replaced with a comment so ``context_window`` never enters the namespace.
_CONTEXT_WINDOW_SEED_BLOCK_REWRITE = "context_window = []"
_CONTEXT_WINDOW_SEED_BLOCK_STANDARD = (
    "# context_rewrite=False: context_window is intentionally NOT seeded."
)


# =============================================================================
# System prompt
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = """\
You operate by writing Python code that runs inside an isolated sandbox. Each
turn you call exactly one tool:

    call_python_repl(code: str)

The code runs in a persistent Python REPL that **survives across turns**.
Variables, imports, and function definitions you make in turn N are still
alive in turn N+1.

# Your context window

You have a Python list called `context_window` already seeded in the kernel.
The task text is shown separately every turn; the contents of
`context_window` are the only model-owned memory you see across turns. Every
turn the conversation history visible to you is reset to just (a) this system
prompt and (b) a freshly-rendered user message.
{visibility_note}
**You will NOT see** stdout/stderr, return values, or anything else from
your code unless it is in `context_window` when the next turn renders. The
rendered `context_window` is hard-truncated with no truncation marker and no
hint that anything exists beyond the visible prefix. For reliable next-turn
visibility, store JSON-serializable values; tuples/sets render back as lists,
and arbitrary objects may render only as lossy repr strings. How you use it is
entirely up to you.

# Your task

Read the task shown in the user message, then use the seeded tool functions
below to gather the information you need.

# Tool functions seeded in the kernel

You can call these directly in your code (they are already defined as
closures over the world state for this rollout):

{tool_signatures}

{tool_budget_section}
{world_addendum}
# Submitting an answer

When you have the final answer, call `submit_answer(value)` where `value` is
the answer. For JSON-answer tasks, pass the corresponding Python list/dict.
This terminates the rollout. Do not wrap with `\\boxed{{}}` — just pass it
directly, e.g. `submit_answer(42)`, `submit_answer("t3x")`, or
`submit_answer([["CP3", "Alice", 2, 1]])`.

# Tips

- Don't `print()` — store anything you want to remember in `context_window`
  instead. Aside from the static task text, it's the only thing you see next
  turn.
- Variables in the kernel persist across turns. They survive — but you
  cannot see their values unless they are in `context_window`.
{error_note}
"""


# Standard tool-calling system prompt — used when ``context_rewrite=False``.
# The model sees the full conversation history each turn (system + initial
# user message + every prior assistant tool-call and tool response). Each
# call to ``call_python_repl`` returns the truncated stdout / stderr /
# trailing-expression repr / error, exactly like a normal REPL tool. There
# is no scratchpad to manage; the trajectory itself is the memory.
SYSTEM_PROMPT_TEMPLATE_STANDARD = """\
You operate by writing Python code that runs inside an isolated sandbox. Each
turn you call exactly one tool:

    call_python_repl(code: str)

The code runs in a persistent Python REPL that **survives across turns**.
Variables, imports, and function definitions you make in turn N are still
alive in turn N+1.

# What the tool returns

The tool result you see after each call is the truncated execution output
of your code:
- ``print()`` output (stdout) and any ``stderr``
- The repr of the trailing expression in your code (rendered as
  ``Out[N]: ...``), if your code ends in an expression
- The full traceback if your code raised an exception (statements before
  the raise still took effect; the one that raised and any after it did
  not)

The combined output is truncated at {max_output_length} characters; if
truncated you'll see a ``... [output truncated]`` marker. The full prior
conversation (every tool call you've made and every tool response) stays
visible to you across turns — you do not need to re-derive earlier
findings, just look back at the trajectory.

# Tool functions seeded in the kernel

You can call these directly in your code (they are already defined as
closures over the world state for this rollout):

{tool_signatures}

{tool_budget_section}
{world_addendum}
# Submitting an answer

When you have the final answer, call `submit_answer(value)` where `value` is
the answer. For JSON-answer tasks, pass the corresponding Python list/dict.
This terminates the rollout. Do not wrap with `\\boxed{{}}` — just pass it
directly, e.g. `submit_answer(42)`, `submit_answer("t3x")`, or
`submit_answer([["CP3", "Alice", 2, 1]])`.
"""


# =============================================================================
# Helpers
# =============================================================================


def _msg_field(msg: Any, field: str, default: Any = None) -> Any:
    """Read a field from a Message that may be a pydantic object or a dict."""
    if msg is None:
        return default
    if isinstance(msg, dict):
        return msg.get(field, default)
    return getattr(msg, field, default)


def check_answer(model_output: str, expected: Any, answer_type: str) -> bool:
    """Type-aware exact-match for the rubric.

    Supports int / str / bool / float. Used by both the live rubric (for
    correctness) and any post-hoc analysis script that needs the same
    semantics as generation-time verification.
    """
    try:
        text = model_output.strip()
    except AttributeError:
        return False
    try:
        if answer_type == "int":
            return int(float(text)) == int(expected)
        if answer_type == "bool":
            return (text.lower() in {"true", "yes", "1"}) == bool(expected)
        if answer_type == "str":
            return text.lower() == str(expected).lower()
        if answer_type == "float":
            return abs(float(text) - float(expected)) < 0.01
        if answer_type == "json":
            try:
                submitted_obj = json.loads(text)
            except Exception:
                import ast

                try:
                    submitted_obj = ast.literal_eval(text)
                except Exception:
                    return False
            try:
                expected_obj = (
                    json.loads(expected)
                    if isinstance(expected, str)
                    else expected
                )
            except Exception:
                expected_obj = expected
            return submitted_obj == expected_obj
    except (ValueError, AttributeError, TypeError):
        return False
    return False


# =============================================================================
# Rewards
# =============================================================================


async def correctness_reward(state: State) -> float:
    """1.0 iff the model's submitted answer matches the expected answer.

    Dispatches on world_type so rule_hunt can verify functional equivalence
    on a held-out test set instead of a string match against the rule's
    text form. All other worlds fall through to type-aware string match.
    """
    world_type = state.get("_world_type", "")
    if world_type == "rule_hunt":
        return _rule_hunt_correctness(state)

    submitted = _submitted_answer_text(state)
    expected = state.get("expected_answer") or ""
    answer_type = state.get("answer_type", "str")
    if not submitted:
        return 0.0
    return 1.0 if check_answer(submitted, expected, answer_type) else 0.0


def _submitted_answer_text(state: State) -> str:
    """Return the submitted answer text across RLMEnv state variants."""
    for key in ("_final_answer", "final_answer"):
        value = state.get(key)
        if value not in (None, ""):
            return str(value)
    value = state.get("answer")
    if isinstance(value, dict) and value.get("ready"):
        content = value.get("content", "")
        return "" if content is None else str(content)
    return ""


def _rule_hunt_correctness(state: State) -> float:
    """Functional-equivalence rubric for rule_hunt.

    1. Parse the model's submitted answer (JSON first, then
       ``ast.literal_eval`` as fallback).
    2. Walk the held-out set stashed on ``state`` and evaluate the model's
       rule on each entity.
    3. Return 1.0 iff every label matches the ground-truth label.
    """
    import ast
    import json as _json

    submitted = _submitted_answer_text(state)
    if not submitted:
        return 0.0
    rule: Any = None
    try:
        rule = _json.loads(submitted)
    except Exception:
        try:
            rule = ast.literal_eval(submitted)
        except Exception:
            return 0.0
    if not isinstance(rule, dict):
        return 0.0

    held_out = state.get("_rule_hunt_held_out") or []
    if not held_out:
        return 0.0

    # Lazy import: keeps the env importable even if the generators package
    # isn't on the path at runtime (only matters at rubric time).
    try:
        from generators.rule_hunt import eval_rule
    except Exception:
        try:
            from .generators.rule_hunt import eval_rule  # type: ignore
        except Exception:
            return 0.0

    for entry in held_out:
        try:
            pred = bool(eval_rule(rule, entry.get("attrs", {})))
        except Exception:
            return 0.0
        if pred != bool(entry.get("label", False)):
            return 0.0
    return 1.0


def _is_correct(state: State) -> bool:
    """Shared correctness check used by ``correctness_reward`` and
    ``context_efficiency_reward`` (the latter gates its bonus on this)."""
    world_type = state.get("_world_type", "")
    if world_type == "rule_hunt":
        return _rule_hunt_correctness(state) >= 1.0
    submitted = _submitted_answer_text(state)
    if not submitted:
        return False
    expected = state.get("expected_answer") or ""
    return check_answer(submitted, expected, state.get("answer_type", "str"))


def _parse_jsonish(value: Any) -> Any:
    """Parse a submitted/context value as JSON or a Python literal."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        import ast

        try:
            return ast.literal_eval(value)
        except Exception:
            return None


def _expected_checkpoint_rows(state: State) -> list[list[Any]]:
    expected = _parse_jsonish(state.get("expected_answer") or "")
    if not isinstance(expected, list):
        return []
    rows: list[list[Any]] = []
    for row in expected:
        if isinstance(row, (list, tuple)) and len(row) == 4:
            rows.append([row[0], row[1], row[2], row[3]])
    return rows


def _submitted_checkpoint_rows(state: State) -> list[list[Any]]:
    submitted = _parse_jsonish(_submitted_answer_text(state))
    if not isinstance(submitted, list):
        return []
    rows: list[list[Any]] = []
    for row in submitted:
        if isinstance(row, (list, tuple)) and len(row) == 4:
            rows.append([row[0], row[1], row[2], row[3]])
    return rows


def _norm_checkpoint_row(row: list[Any]) -> tuple[str, str, int, int] | None:
    try:
        return (str(row[0]), str(row[1]), int(row[2]), int(row[3]))
    except Exception:
        return None


def _row_fraction(rows: list[list[Any]], gold: list[list[Any]]) -> float:
    if not gold:
        return 0.0
    row_set = {
        nr for nr in (_norm_checkpoint_row(row) for row in rows)
        if nr is not None
    }
    gold_rows = [
        nr for nr in (_norm_checkpoint_row(row) for row in gold)
        if nr is not None
    ]
    if not gold_rows:
        return 0.0
    hits = sum(1 for row in gold_rows if row in row_set)
    return hits / len(gold_rows)


def _valid_checkpoint_submit(state: State) -> bool:
    submitted = _parse_jsonish(_submitted_answer_text(state))
    if not isinstance(submitted, list) or not submitted:
        return False
    return all(isinstance(row, (list, tuple)) and len(row) == 4 for row in submitted)


def _complete_checkpoint_submit(state: State) -> bool:
    gold = _expected_checkpoint_rows(state)
    submitted = _submitted_checkpoint_rows(state)
    return bool(gold) and len(submitted) == len(gold) and _valid_checkpoint_submit(state)


def _checkpoint_ids_in_order(state: State) -> bool:
    gold = _expected_checkpoint_rows(state)
    submitted = _submitted_checkpoint_rows(state)
    if not gold or len(submitted) != len(gold):
        return False
    return [str(row[0]) for row in submitted] == [str(row[0]) for row in gold]


def _adaptive_terminal_reached(state: State) -> bool:
    return bool(state.get("_adaptive_terminal_reached"))


def _final_visible_context_text(state: State) -> str:
    """Render final context_window the same way the next cr=True prompt would.

    Partial memory credit is only for rows that survive inside the visible
    hard-truncated prompt, not rows hidden beyond the rendered prefix.
    """
    ctx = state.get("_context_window") or []
    if not ctx:
        return ""
    lines = []
    for i, item in enumerate(ctx):
        entry = item if isinstance(item, str) else repr(item)
        lines.append(f"[{i}] {entry}")
    cap = int(state.get("_max_context_chars_cap", 0) or 400)
    return "\n".join(lines)[:cap]


def _row_visible_in_text(row: tuple[str, str, int, int], text: str) -> bool:
    cp, actor, transfers, owned = row
    json_compact = json.dumps([cp, actor, transfers, owned], separators=(",", ":"))
    json_spaced = json.dumps([cp, actor, transfers, owned])
    repr_single = repr([cp, actor, transfers, owned])
    colon = f"{cp}:{actor}:{transfers}:{owned}"
    comma = f"{cp},{actor},{transfers},{owned}"
    if any(pat in text for pat in (json_compact, json_spaced, repr_single, colon, comma)):
        return True
    # Let compact human manifests count too, e.g. "CP3 Gus 2 1". Keep the
    # gaps short so raw page text with an unrelated checkpoint mention does
    # not accidentally satisfy the row.
    return bool(
        re.search(
            rf"\b{re.escape(cp)}\b.{{0,24}}\b{re.escape(actor)}\b"
            rf".{{0,24}}\b{transfers}\b.{{0,24}}\b{owned}\b",
            text,
            flags=re.S,
        )
    )


def _context_checkpoint_row_fraction(state: State) -> float:
    gold = _expected_checkpoint_rows(state)
    if not gold:
        return 0.0
    text = _final_visible_context_text(state)
    if not text:
        return 0.0
    gold_rows = [
        nr for nr in (_norm_checkpoint_row(row) for row in gold)
        if nr is not None
    ]
    if not gold_rows:
        return 0.0
    hits = sum(1 for row in gold_rows if _row_visible_in_text(row, text))
    return hits / len(gold_rows)


async def checkpoint_row_submit_fraction(state: State) -> float:
    """Metric: exact gold checkpoint rows present in the submitted answer."""
    return _row_fraction(_submitted_checkpoint_rows(state), _expected_checkpoint_rows(state))


async def checkpoint_row_context_fraction(state: State) -> float:
    """Metric: exact gold checkpoint rows visible in final rendered context."""
    return _context_checkpoint_row_fraction(state)


async def valid_checkpoint_submit(state: State) -> float:
    """Metric: submitted answer is a non-empty list of 4-field rows."""
    return 1.0 if _valid_checkpoint_submit(state) else 0.0


async def complete_checkpoint_submit(state: State) -> float:
    """Metric: submitted answer has one valid row per expected checkpoint."""
    return 1.0 if _complete_checkpoint_submit(state) else 0.0


async def checkpoint_ids_in_order(state: State) -> float:
    """Metric: submitted rows use the expected checkpoint ids in order."""
    return 1.0 if _checkpoint_ids_in_order(state) else 0.0


async def adaptive_terminal_reached(state: State) -> float:
    """Metric: the correct adaptive-cursor terminal page was observed."""
    return 1.0 if _adaptive_terminal_reached(state) else 0.0


async def task_reward(state: State) -> float:
    """Main capped reward.

    Adaptive-cursor partial credit is terminal-gated: before the correct
    terminal page is observed, only exact answers receive reward. This avoids
    paying for the shortcut of submitting the first checkpoint row and stopping.
    Once terminal is reached, partial credit is row-dominant and does not reward
    context edits by themselves.

    Other worlds fall back to exact correctness.
    """
    full = await correctness_reward(state)
    if full >= 1.0 or state.get("_world_type") != "adaptive_cursor":
        return min(1.0, full)

    gold = _expected_checkpoint_rows(state)
    if not gold:
        return 0.0
    if not _adaptive_terminal_reached(state):
        return 0.0
    if not _submitted_answer_text(state):
        return 0.0

    complete_valid = 1.0 if _complete_checkpoint_submit(state) else 0.0
    ids_in_order = 1.0 if _checkpoint_ids_in_order(state) else 0.0
    submitted_rows = _row_fraction(_submitted_checkpoint_rows(state), gold)
    partial = 0.05 * complete_valid + 0.10 * ids_in_order + 0.85 * submitted_rows
    return max(0.0, min(1.0, partial))


async def context_efficiency_reward(state: State) -> float:
    """Bonus for correct answers with low average rendered context.

    Value is ``1 - clamp(avg_chars_used / cap, 0, 1)`` so:
      - 1.0 = empty context the whole rollout
      - 0.0 = always at the cap

    Gated on correctness (no bonus for compact-but-wrong rollouts).
    Returns 0.0 in cr=False mode where there's no scratchpad to measure.
    """
    if not state.get("_ctx_render_turns"):
        return 0.0  # cr=False mode (or no renders yet)
    if not _is_correct(state):
        return 0.0
    cap = state.get("_max_context_chars_cap", 0) or 0
    if cap <= 0:
        return 0.0
    n_turns = max(1, state.get("_ctx_render_turns", 0))
    avg = state["_ctx_chars_total"] / n_turns
    util = max(0.0, min(1.0, avg / cap))
    return 1.0 - util


async def manifest_compactness_reward(state: State) -> float:
    """Bonus for correct answers whose curated context manifest stays small.

    This is intentionally stricter than ``context_efficiency_reward``. The
    rendered prompt is always hard-capped, so an append-only policy can hide a
    huge raw log beyond the visible slice and still get the final answer by
    iterating ``context_window`` in code. That behavior is valid Python, but it
    is not the context-management skill we are trying to train. This reward
    gives credit for keeping the *actual* model-owned manifest compact while
    allowing large durable state to live in normal REPL variables behind a
    small breadcrumb.
    """
    if not state.get("_ctx_render_turns"):
        return 0.0  # cr=False mode
    if not _is_correct(state):
        return 0.0

    cap = state.get("_max_context_chars_cap", 0) or 400
    ctx = state.get("_context_window") or []
    manifest_chars = 0
    for item in ctx:
        manifest_chars += len(item if isinstance(item, str) else repr(item))

    # Full credit up to roughly three visible windows of manifest, then decay
    # to zero. This leaves room for compact JSON/dict state but gives raw
    # append logs no bonus.
    char_util = max(0.0, min(1.0, manifest_chars / max(1, cap * 3)))
    char_score = 1.0 - char_util

    # A few slots are fine: task + status + one or two pointers/checkpoints.
    # Long journals get no slot bonus even if each entry is short.
    slots = max(0, len(ctx) - 1)
    slot_util = max(0.0, min(1.0, max(0, slots - 4) / 8))
    slot_score = 1.0 - slot_util

    return 0.75 * char_score + 0.25 * slot_score


async def turn_efficiency_reward(state: State) -> float:
    """Bonus for correct answers solved in fewer turns.

    Value is ``1 - clamp(num_turns / max_turns, 0, 1)`` so:
      - 1.0 = solved in zero turns (impossible; ceiling is ~ 1 - 1/max_turns)
      - 0.0 = used the entire turn budget

    Gated on correctness (no bonus for fast-but-wrong rollouts).
    """
    if not _is_correct(state):
        return 0.0
    cap = state.get("_max_turns_cap", 0) or 0
    if cap <= 0:
        return 0.0
    n_turns = len(state.get("trajectory") or [])
    util = max(0.0, min(1.0, n_turns / cap))
    return 1.0 - util


async def context_truncation_count(state: State) -> float:
    """Number of turns this rollout's rendered context_window was truncated.

    Surfaced as a zero-weight metric for diagnostics — does not contribute
    to the reward signal.
    """
    return float(state.get("_context_truncation_count", 0))


# =============================================================================
# Context-window usage diagnostics (all weight=0; metrics only)
# =============================================================================
#
# These five functions characterize *how* the model uses ``context_window``
# during a rollout: how many appends vs edits, how many of the model's code
# blocks even reference the scratchpad, and how big the list ended up. They
# are recorded in every result row's ``metrics`` field so post-hoc analysis
# (and online eval during training) can chart whether the model is shifting
# from append-only journaling toward edit/overwrite discipline.

_CTX_OP_PATTERNS = {
    "append":   re.compile(r"context_window\.append\b"),
    "extend":   re.compile(r"context_window\.extend\b"),
    "insert":   re.compile(r"context_window\.insert\b"),
    "setitem":  re.compile(r"context_window\[[^\]]+\]\s*="),
    "delitem":  re.compile(r"\bdel\s+context_window\["),
    "pop":      re.compile(r"context_window\.pop\b"),
    "remove":   re.compile(r"context_window\.remove\b"),
    "clear":    re.compile(r"context_window\.clear\b"),
    "reassign": re.compile(r"^\s*context_window\s*=", re.M),
}


def _scan_ctx_ops(state: State) -> dict:
    """Walk the rollout's assistant code blocks and count
    ``context_window`` operations. Cached on ``state`` so the five
    diagnostic metric functions don't each re-scan the trajectory.
    """
    cached = state.get("_ctx_op_stats")
    if cached is not None:
        return cached
    counts = {k: 0 for k in _CTX_OP_PATTERNS}
    n_blocks = 0
    n_blocks_touching_ctx = 0
    for step in state.get("trajectory", []) or []:
        for msg in step.get("completion", []) or []:
            role = (
                msg.get("role")
                if isinstance(msg, dict)
                else getattr(msg, "role", "")
            )
            if role != "assistant":
                continue
            tcs = (
                msg.get("tool_calls")
                if isinstance(msg, dict)
                else getattr(msg, "tool_calls", None)
            ) or []
            for tc in tcs:
                # Tool calls may be: (a) Pydantic ToolCall object,
                # (b) dict with 'arguments'/'function', (c) JSON string.
                if isinstance(tc, str):
                    try:
                        tcd = json.loads(tc)
                    except Exception:
                        continue
                elif isinstance(tc, dict):
                    tcd = tc
                else:
                    args = getattr(tc, "arguments", None)
                    if args is None:
                        fn = getattr(tc, "function", None)
                        args = getattr(fn, "arguments", None) if fn else None
                    tcd = {"arguments": args}
                if not isinstance(tcd, dict):
                    continue
                args = tcd.get("arguments") or (
                    tcd.get("function", {}) or {}
                ).get("arguments", "")
                try:
                    code = (
                        json.loads(args).get("code", "")
                        if isinstance(args, str)
                        else ""
                    )
                except Exception:
                    continue
                n_blocks += 1
                if "context_window" in code:
                    n_blocks_touching_ctx += 1
                for op, pat in _CTX_OP_PATTERNS.items():
                    counts[op] += len(pat.findall(code))
    appends = counts["append"] + counts["extend"] + counts["insert"]
    edits = (
        counts["setitem"]
        + counts["delitem"]
        + counts["pop"]
        + counts["remove"]
        + counts["clear"]
        + counts["reassign"]
    )
    stats = {
        **counts,
        "_n_blocks": n_blocks,
        "_n_blocks_touching_ctx": n_blocks_touching_ctx,
        "_n_appends": appends,
        "_n_edits": edits,
    }
    state["_ctx_op_stats"] = stats
    return stats


async def context_append_count(state: State) -> float:
    """Total ``append`` + ``extend`` + ``insert`` operations across the rollout."""
    return float(_scan_ctx_ops(state)["_n_appends"])


async def context_edit_count(state: State) -> float:
    """Total edit-style operations (``setitem``, ``del``, ``pop``,
    ``remove``, ``clear``, full reassign) across the rollout."""
    return float(_scan_ctx_ops(state)["_n_edits"])


async def context_edit_ratio(state: State) -> float:
    """Edit operations / (edits + appends). 0 = pure-journal append-only,
    1 = pure-state-machine. Range [0, 1]."""
    s = _scan_ctx_ops(state)
    total = s["_n_appends"] + s["_n_edits"]
    return float(s["_n_edits"] / total) if total else 0.0


async def context_final_size(state: State) -> float:
    """Length of ``context_window`` at submission time. High =
    unsummarized log; low = compressed scratchpad."""
    return float(len(state.get("_context_window") or []))


async def context_touch_rate(state: State) -> float:
    """Fraction of the rollout's code blocks that reference
    ``context_window`` at all. 0 = model ignores the scratchpad,
    1 = every code block touches it."""
    s = _scan_ctx_ops(state)
    return float(s["_n_blocks_touching_ctx"] / max(1, s["_n_blocks"]))


async def context_dynamic_append_slots(state: State) -> float:
    """Number of slots appended to context_window across observed snapshots."""
    return float(state.get("_ctx_dynamic_append_slots", 0))


async def context_dynamic_overwrite_slots(state: State) -> float:
    """Number of existing slots whose value changed across snapshots."""
    return float(state.get("_ctx_dynamic_overwrite_slots", 0))


async def context_dynamic_remove_slots(state: State) -> float:
    """Number of slots removed from context_window across snapshots."""
    return float(state.get("_ctx_dynamic_remove_slots", 0))


async def context_dynamic_edit_ratio(state: State) -> float:
    """Dynamic overwrite/remove share of all context_window slot changes."""
    app = state.get("_ctx_dynamic_append_slots", 0)
    edit = (
        state.get("_ctx_dynamic_overwrite_slots", 0)
        + state.get("_ctx_dynamic_remove_slots", 0)
    )
    total = app + edit
    return float(edit / total) if total else 0.0


async def context_final_manifest_chars(state: State) -> float:
    """Serialized character count of the model-curated context_window slots."""
    total = 0
    for item in state.get("_context_window") or []:
        total += len(item if isinstance(item, str) else repr(item))
    return float(total)


def _make_default_rubric() -> vf.Rubric:
    rubric = vf.Rubric()
    # Main reward is capped at 1.0: exact answer wins, otherwise adaptive
    # cursor rows get simple partial credit for valid, derived checkpoint work.
    rubric.add_reward_func(task_reward, weight=1.0)
    # Diagnostic metrics — weight 0; surfaced in `avg_metrics`.
    rubric.add_reward_func(correctness_reward, weight=0.0)
    rubric.add_reward_func(checkpoint_row_submit_fraction, weight=0.0)
    rubric.add_reward_func(checkpoint_row_context_fraction, weight=0.0)
    rubric.add_reward_func(valid_checkpoint_submit, weight=0.0)
    rubric.add_reward_func(complete_checkpoint_submit, weight=0.0)
    rubric.add_reward_func(checkpoint_ids_in_order, weight=0.0)
    rubric.add_reward_func(adaptive_terminal_reached, weight=0.0)
    rubric.add_reward_func(context_efficiency_reward, weight=0.0)
    rubric.add_reward_func(manifest_compactness_reward, weight=0.0)
    # Diagnostic metrics — weight 0; surfaced in `avg_metrics`.
    rubric.add_reward_func(turn_efficiency_reward, weight=0.0)
    rubric.add_reward_func(context_truncation_count, weight=0.0)
    rubric.add_reward_func(context_append_count, weight=0.0)
    rubric.add_reward_func(context_edit_count, weight=0.0)
    rubric.add_reward_func(context_edit_ratio, weight=0.0)
    rubric.add_reward_func(context_final_size, weight=0.0)
    rubric.add_reward_func(context_touch_rate, weight=0.0)
    rubric.add_reward_func(context_dynamic_append_slots, weight=0.0)
    rubric.add_reward_func(context_dynamic_overwrite_slots, weight=0.0)
    rubric.add_reward_func(context_dynamic_remove_slots, weight=0.0)
    rubric.add_reward_func(context_dynamic_edit_ratio, weight=0.0)
    rubric.add_reward_func(context_final_manifest_chars, weight=0.0)
    return rubric


# =============================================================================
# The env
# =============================================================================


class ContextToolsEnv(RLMEnv):
    """Sandboxed Python-REPL harness with two prompting modes.

    Inherits everything from `RLMEnv` (sandbox lifecycle, FIFO transport,
    persistent worker subprocess). The toggle ``context_rewrite`` selects
    one of two prompting flavors:

    **``context_rewrite=True`` (default — context-management mode)**:

    1. **No HTTP interception** — `_setup_interception_and_register` is a
       no-op since we have no sub-LLMs and no proxied root tools.
    2. **Boot block** — `customize_worker_script` injects world tool
       functions, `submit_answer`, and `context_window` into the worker
       namespace before the FIFO loop starts. Also adds a snapshot of
       `context_window` to each per-turn result dict so the env can render
       it next turn.
    3. **Fresh `[system, user]` every turn** — `get_prompt_messages` is
       fully overridden; the model never sees the trajectory. The user
       message re-renders the static task text plus the hard-truncated
       `context_window`, with previous code shown only according to the
       configured error/code echo policy.
    4. **Tool dispatch bypassed** — `env_response` runs the model's code
       via RLM's `_execute_code` (FIFO transport) and returns `[]` so no
       tool message is appended to the trajectory.

    **``context_rewrite=False`` (standard tool-calling mode)**:

    1. Same no-op `_setup_interception_and_register`.
    2. Boot block injects world tool functions, `submit_answer`, and the
       budget wrapper — but NOT `context_window`. No per-turn snapshot.
    3. ``get_prompt_messages``, ``env_response``, ``add_trajectory_step``,
       and ``render_completion`` all defer to the base ``RLMEnv``
       implementation: standard tool-call → tool-response trajectory
       accumulation. The model sees the full prior conversation each turn.
       The default ``call_python_repl`` returns the truncated stdout /
       stderr / trailing-expression repr / traceback. Truncation is
       ``max_context_chars`` (matching the rendered-context cap of the
       other mode).
    """

    def __init__(
        self,
        *,
        dataset: Dataset,
        eval_dataset: Dataset | None = None,
        rubric: vf.Rubric | None = None,
        max_turns: int = 15,
        max_context_chars: int = 400,
        max_code_display_chars: int = 4000,
        show_previous_code: bool = False,
        context_rewrite: bool = True,
        tool_call_budget_per_turn: int = 1_000_000,
        sandbox_docker_image: str = "python:3.11-slim",
        code_execution_timeout: int = 120,
        max_startup_wait_seconds: int = 120,
        sandbox_cpu_cores: int = 1,
        sandbox_memory_gb: int = 2,
        sandbox_disk_size_gb: int = 5,
        sandbox_timeout_minutes: int = 30,
        sandbox_labels: list[str] | None = None,
        sandbox_client_max_workers: int | None = 16,
        sandbox_client_max_connections: int = 100,
        sandbox_client_max_keepalive_connections: int = 50,
        sandbox_transfer_max_retries: int = 8,
        retain_filesystem_after_rollout: bool = False,
        **kwargs: Any,
    ):
        self.max_context_chars = max_context_chars
        self.max_code_display_chars = max_code_display_chars
        self.show_previous_code = show_previous_code
        self.context_rewrite = bool(context_rewrite)
        self.tool_call_budget_per_turn = tool_call_budget_per_turn
        self.sandbox_client_max_workers = sandbox_client_max_workers or 16
        self.sandbox_client_max_connections = sandbox_client_max_connections
        self.sandbox_client_max_keepalive_connections = (
            sandbox_client_max_keepalive_connections
        )

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric or _make_default_rubric(),
            max_turns=max_turns,
            tools=[],
            root_tools=[],
            sub_tools=[],
            enable_sub_llms=False,
            enable_summarization=False,
            repl_language="python",
            sandbox_docker_image=sandbox_docker_image,
            code_execution_timeout=code_execution_timeout,
            max_startup_wait_seconds=max_startup_wait_seconds,
            sandbox_cpu_cores=sandbox_cpu_cores,
            sandbox_memory_gb=sandbox_memory_gb,
            sandbox_disk_size_gb=sandbox_disk_size_gb,
            sandbox_timeout_minutes=sandbox_timeout_minutes,
            sandbox_labels=sandbox_labels or ["context-tools"],
            sandbox_client_max_workers=sandbox_client_max_workers,
            sandbox_client_max_connections=sandbox_client_max_connections,
            sandbox_client_max_keepalive_connections=(
                sandbox_client_max_keepalive_connections
            ),
            sandbox_transfer_max_retries=sandbox_transfer_max_retries,
            retain_filesystem_after_rollout=retain_filesystem_after_rollout,
            pip_install_packages="",
            **kwargs,
        )
        self.sandbox_client_max_workers = sandbox_client_max_workers or 16
        self.sandbox_client_max_connections = sandbox_client_max_connections
        self.sandbox_client_max_keepalive_connections = (
            sandbox_client_max_keepalive_connections
        )
        # RLMEnv creates a generic executor in its constructor. Replace it with
        # a lean version whose setup path matches this environment's needs.
        self._executor.teardown_sandbox_client()
        self._executor = _LeanContextToolsExecutor(self)
        self.with_retry_on_read_errors = tc.AsyncRetrying(
            retry=tc.retry_if_exception(_is_retryable_transfer_error),
            stop=tc.stop_after_attempt(sandbox_transfer_max_retries + 1),
            wait=tc.wait_exponential_jitter(initial=1, max=30),
            before_sleep=tc.before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ).wraps

        # In standard tool-calling mode the default ``call_python_repl``
        # truncates the formatted result at ``self.max_output_length``.
        # Match that to ``max_context_chars`` so the per-tool-response cap
        # matches the rendered-context cap of the other mode.
        if not self.context_rewrite:
            self.max_output_length = max_context_chars

    # ------------------------------------------------------------------------
    # Setup: pre-stage info, skip interception, override system prompt
    # ------------------------------------------------------------------------

    async def _setup_interception_and_register(
        self, state: State, rollout_id: str
    ) -> State:
        """No-op. We don't need the HTTP server / reverse tunnel."""
        state["interception_url"] = ""
        state["root_tool_url"] = ""
        self.active_rollouts[rollout_id] = {
            "client": state.get("client"),
            "model": state.get("model"),
            "sub_model": None,
            "state": state,
        }
        return state

    async def setup_state(self, state: State, **kwargs: Any) -> State:
        info = state.get("info") or {}
        if not isinstance(info, dict):
            info = {}

        # Parse the world state JSON-string from the dataset row.
        world_state = info.get("state", {})
        if isinstance(world_state, str):
            try:
                world_state = json.loads(world_state)
            except Exception:
                world_state = {}
        per_example_context_cap = None
        if isinstance(world_state, dict):
            per_example_context_cap = world_state.get("max_context_chars")
        if per_example_context_cap is None:
            per_example_context_cap = info.get("max_context_chars")

        # Static task text shown each turn. It is not a context_window slot,
        # so the model cannot turn it into an unlimited scratchpad by
        # overwriting index 0.
        questions = info.get("questions") or []
        if isinstance(questions, str):
            try:
                questions = json.loads(questions)
            except Exception:
                questions = []
        initial_user_text = ""
        if questions and isinstance(questions[0], dict):
            initial_user_text = questions[0].get("query_text") or ""
        if not initial_user_text:
            for m in state.get("prompt", []) or []:
                if _msg_field(m, "role") == "user":
                    initial_user_text = _msg_field(m, "content", "") or ""

        # rule_hunt: stash the held-out test set + ground-truth rule on
        # `state` for the rubric to evaluate against. These keys are not
        # underscore-prefixed in the dataset row, so they need an explicit
        # strip from world_state before the sandbox sees it.
        if info.get("world_type") == "rule_hunt" and isinstance(world_state, dict):
            state["_rule_hunt_held_out"] = list(world_state.get("held_out") or [])
            state["_rule_hunt_rule"] = world_state.get("rule")

        # Strip generator-internal keys from the world_state shipped to the
        # sandbox. Convention: any top-level key starting with `_` is
        # private to the generator/solver (question metadata, difficulty
        # axis values, etc.) and never needs to reach the model. Plus the
        # rule_hunt-specific reveal keys (already stashed on `state` above).
        _LEAK_KEYS = {"held_out", "rule"} if info.get("world_type") == "rule_hunt" else set()
        if isinstance(world_state, dict):
            world_state = {
                k: v for k, v in world_state.items()
                if not k.startswith("_") and k not in _LEAK_KEYS
            }

        # Plant info["context"] so RLMEnv writes it to /rlm_fs/context.json
        # in the sandbox (the boot block reads it from there).
        info["context"] = {
            "world_state": world_state,
            "world_type": info.get("world_type", ""),
            "initial_user_text": initial_user_text,
        }
        state["info"] = info

        # Stash extras the env needs at runtime.
        state["_world_type"] = info.get("world_type", "")
        state["expected_answer"] = info.get("expected_answer", "") or ""
        state["answer_type"] = info.get("answer_type", "str")
        state["optimal_turns"] = info.get("optimal_turns", 1)
        state["example_id"] = info.get("example_id", 0)
        state["_initial_user_text"] = initial_user_text

        # Per-turn slots populated by env_response (rewrite=True only;
        # harmless if also present in rewrite=False).
        state["_last_code"] = ""
        state["_last_error"] = None
        state["_context_window"] = []
        state["_final_answer"] = ""
        state["_adaptive_terminal_reached"] = False

        # Tracking slots for the efficiency reward + truncation metric
        # (cr=True only — populated by ``_build_user_message`` each turn).
        try:
            state["_max_context_chars_cap"] = int(per_example_context_cap)
        except Exception:
            state["_max_context_chars_cap"] = self.max_context_chars
        if state["_max_context_chars_cap"] <= 0:
            state["_max_context_chars_cap"] = self.max_context_chars
        state["_max_turns_cap"] = self.max_turns
        state["_ctx_chars_total"] = 0       # running sum of rendered chars
        state["_ctx_render_turns"] = 0      # number of times we rendered
        state["_context_truncation_count"] = 0
        state["_ctx_dynamic_append_slots"] = 0
        state["_ctx_dynamic_overwrite_slots"] = 0
        state["_ctx_dynamic_remove_slots"] = 0

        # Hand off to RLMEnv: provisions sandbox, uploads context.json, writes
        # worker (calling our customize_worker_script along the way), starts
        # the worker. Mutates ``state`` in place; returns None as of
        # verifiers 0.1.13 (see commit 7ac48070 — "Tighten setup_state
        # contract: in-place mutation, return None").
        await super().setup_state(state, **kwargs)

        # Override RLM's prompt with ours.
        state["rlm_system_prompt"] = self._build_system_prompt(state["_world_type"])

        # In standard tool-calling mode, the model's first user message is
        # the same query text that rewrite=True renders as static task text.
        # RLMEnv's base ``get_prompt_messages`` will additionally prepend our
        # ``rlm_system_prompt`` (wrapped as
        # ``<RLM_SCAFFOLDING>``) into the first user message on turn 1.
        if not self.context_rewrite:
            state["prompt"] = [
                UserMessage(content=initial_user_text),
            ]
        return state

    # ------------------------------------------------------------------------
    # Worker script customization
    # ------------------------------------------------------------------------

    def customize_worker_script(self, script: str, state: State) -> str:
        """Splice the boot block + per-turn budget reset (+ context_window
        snapshot when ``context_rewrite=True``) into the worker."""
        script = script.replace("import requests\n", "")
        world_type = state.get("_world_type", "")
        tool_defs = _TOOL_BOOT_BLOCKS.get(world_type, "")
        tool_names = _TOOL_NAMES_PER_WORLD.get(world_type, [])
        wrap_lines = "\n".join(f"{n} = _budget_wrap({n})" for n in tool_names)
        # The seed line for ``context_window`` is conditional on
        # ``self.context_rewrite``. In standard tool-calling mode we leave
        # ``context_window`` undefined in the kernel — the model has no
        # scratchpad to manage and the prompt makes no mention of it.
        ctx_seed_block = (
            _CONTEXT_WINDOW_SEED_BLOCK_REWRITE
            if self.context_rewrite
            else _CONTEXT_WINDOW_SEED_BLOCK_STANDARD
        )
        boot_code = _BOOT_CODE_TEMPLATE.format(
            budget=self.tool_call_budget_per_turn,
            tool_defs=tool_defs,
            tool_wrap_lines=wrap_lines,
            context_window_seed_block=ctx_seed_block,
        )

        # 1. Run boot block right before the worker writes the ready flag.
        ready_marker = 'Path(READY_FLAG).write_text("ready", encoding="utf-8")'
        if ready_marker not in script:
            raise RuntimeError(
                "RLMEnv worker template changed: ready-flag marker missing."
            )
        boot_runner = (
            "_boot_code = " + repr(boot_code) + "\n"
            "exec(compile(_boot_code, '<boot>', 'exec'), namespace, namespace)\n"
        )
        script = script.replace(ready_marker, boot_runner + ready_marker)

        # 2. Reset the per-turn tool budget counter at the top of each turn,
        #    right after the worker increments execution_count.
        budget_reset_marker = "    execution_count += 1"
        if budget_reset_marker not in script:
            raise RuntimeError(
                "RLMEnv worker template changed: execution_count marker missing."
            )
        script = script.replace(
            budget_reset_marker,
            budget_reset_marker + "\n    namespace['_tool_call_count'] = 0",
            1,
        )

        # 3. Snapshot context_window into the per-turn result dict — only
        #    needed when the rewrite flow is active. In standard mode the
        #    framework's default tool-result formatting is what the model
        #    sees; ``context_window`` doesn't exist in the namespace.
        if self.context_rewrite:
            ans_marker = '    result["answer"] = namespace.get("answer", '
            idx = script.find(ans_marker)
            if idx < 0:
                raise RuntimeError(
                    "RLMEnv worker template changed: answer-line marker missing."
                )
            line_end = script.find("\n", idx)
            snapshot = (
                "    try:\n"
                "        _ctx_safe = namespace.get(\"_context_tools_json_safe\")\n"
                "        _ctx_raw = list(namespace.get(\"context_window\", []))\n"
                "        if callable(_ctx_safe):\n"
                "            result[\"context_window\"] = _ctx_safe(_ctx_raw)\n"
                "        else:\n"
                "            result[\"context_window\"] = _ctx_raw\n"
                "    except Exception:\n"
                "        try:\n"
                "            result[\"context_window\"] = [repr(x) for x in list(namespace.get(\"context_window\", []))]\n"
                "        except Exception:\n"
                "            result[\"context_window\"] = []\n"
                "    try:\n"
                "        _obs = namespace.get(\"observe\")\n"
                "        _terminal = getattr(_obs, \"_context_tools_terminal_reached\", None)\n"
                "        if _terminal is None:\n"
                "            _terminal = object.__getattribute__(_obs, \"_AdaptiveObserve__terminal_reached\")\n"
                "        result[\"adaptive_terminal_reached\"] = bool(_terminal)\n"
                "    except Exception:\n"
                "        result[\"adaptive_terminal_reached\"] = False\n"
            )
            script = script[: line_end + 1] + snapshot + script[line_end + 1 :]
        return script

    # ------------------------------------------------------------------------
    # Prompt construction (fresh [system, user] every turn)
    # ------------------------------------------------------------------------

    def _build_system_prompt(self, world_type: str) -> str:
        """Build the system prompt for this rollout's world.

        ``context_rewrite=True`` uses the context-window template (the
        original behavior, untouched). ``context_rewrite=False`` uses the
        standard tool-calling template — no mention of ``context_window``,
        and a description of what each tool response will contain.
        """
        sigs = list(_TOOL_SIGNATURES.get(world_type, []))
        sigs.append(_SUBMIT_ANSWER_SIGNATURE)
        world_addendum = _WORLD_PROMPT_ADDENDUM.get(world_type, "")
        if self.tool_call_budget_per_turn >= 1_000_000:
            tool_budget_section = (
                "**Tool-call budget.** There is no small manufactured limit on "
                "seeded tool calls per code block. The seeded tools are "
                "ordinary Python functions; any need to stop, inspect, and "
                "continue comes from the task content, not from harness "
                "control flow."
            )
        else:
            tool_budget_section = (
                "**Per-turn tool budget.** Each `call_python_repl` invocation "
                "gets a fresh budget of "
                f"{self.tool_call_budget_per_turn} tool calls (across all the "
                "seeded tools above). The (N+1)-th call in a single turn raises "
                "``ToolBudgetExhausted`` and any statements after the failed "
                "call do not run. Assignments made *before* the budget hit DO "
                "survive in the kernel. `submit_answer` does not count against "
                "the budget."
            )

        if not self.context_rewrite:
            return SYSTEM_PROMPT_TEMPLATE_STANDARD.format(
                tool_signatures="\n".join(sigs),
                tool_budget_section=tool_budget_section,
                max_output_length=self.max_context_chars,
                world_addendum=world_addendum,
            )

        if self.show_previous_code:
            visibility_note = (
                "The user message shows the current contents of "
                "`context_window` plus the code you wrote last turn (and "
                "any error it produced)."
            )
            error_note = (
                "- If your code raises, the traceback is shown to you in the "
                "next user message under your previous code. The kernel "
                "namespace is preserved — statements that executed before "
                "the exception still took effect; the one that raised and "
                "any statements after it did not."
            )
        else:
            visibility_note = (
                "The user message shows the current contents of "
                "`context_window`. **The code you wrote last turn is NOT "
                "echoed back to you on success** — it is shown only when "
                "it raised an error (so you can diagnose alongside the "
                "traceback). On a successful turn, your previous code "
                "leaves no trace in the next prompt."
            )
            error_note = (
                "- If your code raises, the previous code AND the traceback "
                "are shown to you in the next user message so you can "
                "diagnose. On successful turns the code is NOT shown — only "
                "your `context_window` is. The kernel namespace is preserved "
                "either way: statements that executed before any exception "
                "still took effect; the one that raised and any after it "
                "did not."
            )

        return SYSTEM_PROMPT_TEMPLATE.format(
            tool_signatures="\n".join(sigs),
            tool_budget_section=tool_budget_section,
            visibility_note=visibility_note,
            error_note=error_note,
            world_addendum=world_addendum,
        )

    def _build_user_message(self, state: State) -> str:
        ctx = state.get("_context_window") or []
        task_text = state.get("_initial_user_text") or ""
        task_block = task_text if isinstance(task_text, str) else repr(task_text)
        context_cap = int(
            state.get("_max_context_chars_cap") or self.max_context_chars
        )

        # Render the model-owned scratchpad under one hard cap. There is no
        # protected index, truncation marker, or omitted-count hint; whatever is
        # beyond the visible prefix is simply absent from the next prompt.
        if ctx:
            managed_lines = []
            for i, item in enumerate(ctx):
                entry = item if isinstance(item, str) else repr(item)
                managed_lines.append(f"[{i}] {entry}")
            managed_raw = "\n".join(managed_lines)
            visible_managed = managed_raw[:context_cap]
            truncated = len(visible_managed) < len(managed_raw)
            used = len(visible_managed)
            ctx_block = visible_managed
        else:
            used = 0
            truncated = False
            ctx_block = "(empty)"

        # Track per-turn rendered ctx size + truncation hits for the
        # efficiency reward and truncation metric.
        state["_ctx_chars_total"] = state.get("_ctx_chars_total", 0) + used
        state["_ctx_render_turns"] = state.get("_ctx_render_turns", 0) + 1
        if truncated:
            state["_context_truncation_count"] = (
                state.get("_context_truncation_count", 0) + 1
            )

        last_code = state.get("_last_code") or ""
        last_error = state.get("_last_error") or ""

        def _truncate(s: str) -> str:
            if len(s) > self.max_code_display_chars:
                return s[: self.max_code_display_chars] + "\n... [truncated]"
            return s

        # Build the optional previous-code / error block. Two display modes:
        #   show_previous_code=True  → echo the previous code every turn
        #                              (plus error block when present)
        #   show_previous_code=False → echo previous code ONLY on errors,
        #                              so successful turns leave no trace —
        #                              the model has to commit progress to
        #                              context_window itself if it wants to
        #                              remember what it did.
        code_section = ""
        if self.show_previous_code:
            if last_code:
                code_block = f"```python\n{_truncate(last_code)}\n```"
            else:
                code_block = "(this is your first turn; no previous code)"
            if last_error:
                err_block = (
                    "\n\n=== ERROR (your previous code raised; statements "
                    "after the exception did NOT run) ===\n"
                    f"```\n{_truncate(last_error.rstrip())}\n```"
                )
            else:
                err_block = ""
            code_section = (
                f"\n\n=== previous code ===\n{code_block}{err_block}"
            )
        else:
            # Off mode: only echo when the previous code errored.
            if last_error and last_code:
                code_section = (
                    "\n\n=== previous code (errored — shown so you can "
                    "diagnose; otherwise previous code is NOT echoed) ===\n"
                    f"```python\n{_truncate(last_code)}\n```\n\n"
                    "=== ERROR (statements after the exception did NOT run) "
                    "===\n"
                    f"```\n{_truncate(last_error.rstrip())}\n```"
                )
            elif last_error and not last_code:
                # Defensive: shouldn't happen since errors imply code ran,
                # but render the error alone if it does.
                code_section = (
                    "\n\n=== ERROR ===\n"
                    f"```\n{_truncate(last_error.rstrip())}\n```"
                )

        return (
            "=== task ===\n"
            f"{task_block}\n\n"
            "=== context_window ===\n"
            f"{ctx_block}"
            f"{code_section}\n\n"
            "Now write the next call_python_repl(...) code."
        )

    async def get_prompt_messages(self, state: State) -> vf.Messages:
        # Standard tool-calling mode: defer to RLMEnv's default, which
        # builds the prompt by injecting our ``rlm_system_prompt`` on turn
        # 1 and walking the accumulated trajectory on later turns. The
        # model sees the full history.
        if not self.context_rewrite:
            return await super().get_prompt_messages(state)

        # Context-management mode: fresh ``[system, user]`` every turn.
        # The user message renders ``context_window`` (and, on errors,
        # the previous code + traceback). The trajectory is invisible.
        return [
            SystemMessage(content=state.get("rlm_system_prompt", "")),
            UserMessage(content=self._build_user_message(state)),
        ]

    # ------------------------------------------------------------------------
    # Tool dispatch bypass + immediate exec  (rewrite=True only)
    # ------------------------------------------------------------------------

    async def env_response(
        self, messages: vf.Messages, state: State, **kwargs: Any
    ) -> vf.Messages:
        """Run the model's code via the worker; do not append tool messages.

        Only used when ``context_rewrite=True``. In standard mode we defer
        to ``super().env_response``, which dispatches the registered
        ``call_python_repl`` tool the normal way (formatting stdout /
        stderr / ``Out[N]:`` / traceback into a tool message that the
        framework appends to the trajectory).
        """
        if not self.context_rewrite:
            return await super().env_response(messages, state, **kwargs)

        last = messages[-1] if messages else None
        if last is None or _msg_field(last, "role") != "assistant":
            return []
        for tc in _msg_field(last, "tool_calls") or []:
            if _msg_field(tc, "name") != "call_python_repl":
                continue
            args_raw = _msg_field(tc, "arguments")
            try:
                args = (
                    json.loads(args_raw) if isinstance(args_raw, str)
                    else (args_raw or {})
                )
            except Exception:
                args = {}
            code = args.get("code", "") or ""
            state["_last_code"] = code

            try:
                result = await self._execute_code(code, state)
            except Exception as exc:  # noqa: BLE001 — surface infra errors
                state["_last_error"] = (
                    f"Sandbox error: {type(exc).__name__}: {exc}"
                )
                continue

            if result.get("status") == "error":
                state["_last_error"] = result.get("result") or "Unknown error"
            else:
                state["_last_error"] = None

            cw = result.get("context_window")
            if isinstance(cw, list):
                old_cw = list(state.get("_context_window") or [])
                shared = min(len(old_cw), len(cw))
                overwritten = sum(
                    1
                    for i in range(shared)
                    if repr(old_cw[i]) != repr(cw[i])
                )
                appended = max(0, len(cw) - len(old_cw))
                removed = max(0, len(old_cw) - len(cw))
                state["_ctx_dynamic_overwrite_slots"] = (
                    state.get("_ctx_dynamic_overwrite_slots", 0) + overwritten
                )
                state["_ctx_dynamic_append_slots"] = (
                    state.get("_ctx_dynamic_append_slots", 0) + appended
                )
                state["_ctx_dynamic_remove_slots"] = (
                    state.get("_ctx_dynamic_remove_slots", 0) + removed
                )
                state["_context_window"] = list(cw)
            if result.get("adaptive_terminal_reached"):
                state["_adaptive_terminal_reached"] = True

            ans = result.get("answer") or {}
            if isinstance(ans, dict) and ans.get("ready"):
                state["_final_answer"] = ans.get("content", "") or ""
                state["final_answer"] = state["_final_answer"]
        return []

    async def add_trajectory_step(self, state: State, trajectory_step: Any) -> None:
        """Append + immediately run env_response so submit_answer doesn't
        burn an extra turn waiting for the next prompt-building cycle.

        Rewrite=True only. In standard mode the base implementation handles
        this through normal tool-call → tool-response cycling.
        """
        if not self.context_rewrite:
            return await super().add_trajectory_step(state, trajectory_step)

        state["trajectory"].append(trajectory_step)
        prompt = list(trajectory_step.get("prompt") or [])
        completion = list(trajectory_step.get("completion") or [])
        if completion:
            await self.env_response(prompt + completion, state)

    async def render_completion(self, state: State) -> None:
        """Save the full per-turn flow as the result row's completion.

        ``context_rewrite=True``: default ``MultiTurnEnv.render_completion``
        (and our previous override which only concatenated assistant
        messages) doesn't surface what the model actually saw each turn —
        but our ``get_prompt_messages`` rebuilds the user message fresh
        every turn (rendered ``context_window`` + last code + any error).
        Without those user messages in the saved row, you can only see the
        model's outputs, not the inputs. Output shape:
        ``[user_turn1, assistant_turn1, user_turn2, assistant_turn2, ...]``.
        The system prompt is omitted (it's static and identical every turn
        — easier to reconstruct from ``state["rlm_system_prompt"]`` if you
        need it). For each step we pick the first user message in the
        step's ``prompt`` field (our ``get_prompt_messages`` always returns
        ``[system, user]`` so there is exactly one).

        ``context_rewrite=False``: the trajectory naturally accumulates
        (assistant + tool messages each turn), so we let the base
        implementation render it as-is.
        """
        if not self.context_rewrite:
            return await super().render_completion(state)

        traj = state.get("trajectory") or []
        if not traj:
            state["completion"] = []
            return
        msgs: list[Any] = []
        for step in traj:
            prompt = list(step.get("prompt") or [])
            completion = list(step.get("completion") or [])
            for m in prompt:
                if _msg_field(m, "role") == "user":
                    msgs = concat_messages([msgs, [m]])
                    break
            msgs = concat_messages([msgs, completion])
        state["completion"] = msgs


# =============================================================================
# Loader
# =============================================================================


def load_environment(
    dataset_path: str | Path | None = None,
    eval_path: str | Path | None = None,
    max_turns: int = 15,
    max_context_chars: int = 400,
    max_code_display_chars: int = 4000,
    show_previous_code: bool = False,
    context_rewrite: bool = True,
    tool_call_budget_per_turn: int = 1_000_000,
    sandbox_docker_image: str = "python:3.11-slim",
    code_execution_timeout: int = 120,
    sandbox_cpu_cores: int = 1,
    sandbox_memory_gb: int = 2,
    sandbox_timeout_minutes: int = 30,
    sandbox_client_max_workers: int | None = None,
    sandbox_client_max_connections: int = 100,
    sandbox_client_max_keepalive_connections: int = 50,
    sandbox_transfer_max_retries: int = 8,
    retain_filesystem_after_rollout: bool = False,
) -> ContextToolsEnv:
    """Build the env from on-disk JSONL via the TaskSet abstraction.

    Args:
        show_previous_code: (rewrite=True only.) If True, every user message
            echoes the model's previous-turn code (and any error). If False
            (default), the previous code is shown ONLY when it raised an
            error — successful turns leave no echo, forcing the model to
            log progress to ``context_window`` itself if it wants to
            remember what it just did.
        context_rewrite: If True (default) the environment runs in
            context-management mode — the model has a ``context_window``
            list seeded in the kernel, sees a fresh ``[system, user]``
            every turn (with the rendered ``context_window``), and never
            sees the trajectory. If False the environment runs in standard
            tool-calling mode — no ``context_window`` is seeded, the model
            sees the full conversation history each turn, and each
            ``call_python_repl`` returns the truncated execution output as
            a normal tool response.
    """
    here = Path(__file__).parent
    # Default training set: 60% adaptive_cursor ledger tasks plus 40%
    # corpus_trail research tasks. Built by ``scripts/build_context_mix.py``.
    if dataset_path is None:
        dataset_path = here / "my_data" / "train_context_mix.jsonl"
    if eval_path is None:
        eval_path = here / "my_data" / "eval_context_mix.jsonl"

    train = ContextToolsTaskSet(dataset_path=dataset_path, name="context-tools")
    eval_ts = ContextToolsTaskSet(dataset_path=eval_path, name="context-tools-eval")

    return ContextToolsEnv(
        dataset=train.get_dataset(),
        eval_dataset=eval_ts.get_dataset(),
        max_turns=max_turns,
        max_context_chars=max_context_chars,
        max_code_display_chars=max_code_display_chars,
        show_previous_code=show_previous_code,
        context_rewrite=context_rewrite,
        tool_call_budget_per_turn=tool_call_budget_per_turn,
        sandbox_docker_image=sandbox_docker_image,
        code_execution_timeout=code_execution_timeout,
        sandbox_cpu_cores=sandbox_cpu_cores,
        sandbox_memory_gb=sandbox_memory_gb,
        sandbox_timeout_minutes=sandbox_timeout_minutes,
        sandbox_client_max_workers=sandbox_client_max_workers,
        sandbox_client_max_connections=sandbox_client_max_connections,
        sandbox_client_max_keepalive_connections=(
            sandbox_client_max_keepalive_connections
        ),
        sandbox_transfer_max_retries=sandbox_transfer_max_retries,
        retain_filesystem_after_rollout=retain_filesystem_after_rollout,
    )
