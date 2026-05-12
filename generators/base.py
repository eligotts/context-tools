"""Base classes and data structures for episode generation.

Schema evolution vs. parallel_tools
-----------------------------------
parallel_tools had one flat `TrainingExample` per row. context_tools lifts that
into a two-level shape:

    TrainingEpisode
        rules:     list[dict]        # family E (planned)
        mutations: list[dict]        # family G (planned)
        questions: list[Question]    # len == 1 for single-question; >1 for family B

    Question
        template_name, query_text, expected_answer, answer_type,
        target_entities (BFS start set),
        parameter_refs (family A: entity+attr lookups whose value becomes a filter),
        optimal_turns (as measured by ReferenceSolver)

`TrainingExample` is retained as a thin back-compat adapter: world generators can
still return one and the dataset layer wraps it into a TrainingEpisode with a
single Question. That way the existing world implementations don't need to be
rewritten before the new features land.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
import json
from typing import Any


# =============================================================================
# Question / Episode / TrainingExample
# =============================================================================


@dataclass
class ParameterRef:
    """A lookup whose resolved value is used as a filter/threshold in the query.

    Used by Family A (deferred-parameter queries). The solver fetches the named
    attribute of `entity` on turn 1 (in parallel with BFS layer 1) and caches it,
    then applies `role` when computing the final answer.

    Example:
        ParameterRef(entity="h2x", attr="priority", role="min_priority")
        → "... tasks whose priority >= priority-of-h2x"
    """

    entity: str
    attr: str     # which attribute of the entity: "priority" | "value" | "duration" | "birth_year" | ...
    role: str     # how the resolved value is used: "min_priority" | "threshold" | ...


@dataclass
class Question:
    """One natural-language question posed against a world state."""

    template_name: str
    query_text: str
    expected_answer: str              # always stringified for storage consistency
    answer_type: str                  # "int" | "str" | "bool" | "float"
    target_entities: list[str] = field(default_factory=list)
    parameter_refs: list[ParameterRef] = field(default_factory=list)
    optimal_turns: int = 1

    def to_dict(self) -> dict:
        return {
            "template_name": self.template_name,
            "query_text": self.query_text,
            "expected_answer": self.expected_answer,
            "answer_type": self.answer_type,
            "target_entities": list(self.target_entities),
            "parameter_refs": [asdict(p) for p in self.parameter_refs],
            "optimal_turns": self.optimal_turns,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        return cls(
            template_name=d["template_name"],
            query_text=d["query_text"],
            expected_answer=d["expected_answer"],
            answer_type=d["answer_type"],
            target_entities=list(d.get("target_entities", [])),
            parameter_refs=[ParameterRef(**p) for p in d.get("parameter_refs", [])],
            optimal_turns=int(d.get("optimal_turns", 1)),
        )


@dataclass
class TrainingEpisode:
    """A complete episode: one world state + 1..N questions + optional rules/mutations."""

    episode_id: int
    world_type: str
    system_prompt: str
    state: dict
    questions: list[Question]

    # Reserved for later families. Empty in v1.
    rules: list[dict] = field(default_factory=list)
    mutations: list[dict] = field(default_factory=list)

    # Episode-level summaries (set by the dataset builder / solver).
    optimal_turns: int = 1
    difficulty: int = 1
    depth: int = 0
    breadth: int = 0

    # --- Single-question view: back-compat with the parallel_tools row shape. -------

    def _first_q(self) -> Question:
        return self.questions[0]

    def to_dataset_row(self) -> dict:
        """Convert to a verifiers-compatible dataset row.

        For single-question episodes, the row matches the parallel_tools format
        exactly so `context_tools.py` can consume without changes. Multi-question
        episodes additionally populate `info["questions"]` with the full list.
        """
        q0 = self._first_q()
        info = {
            "example_id": self.episode_id,
            "world_type": self.world_type,
            "state": json.dumps(self.state),
            "optimal_turns": self.optimal_turns,
            "expected_answer": q0.expected_answer,
            "answer_type": q0.answer_type,
            "difficulty": self.difficulty,
            "depth": self.depth,
            "breadth": self.breadth,
            "query_template": q0.template_name,
            # New fields — always present, empty for single-question/no-rule episodes.
            "parameter_refs": [asdict(p) for p in q0.parameter_refs],
            "target_entities": list(q0.target_entities),
            "rules": list(self.rules),
            "questions": [q.to_dict() for q in self.questions],
        }
        return {
            "prompt": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": q0.query_text},
            ],
            "answer": q0.expected_answer,
            "example_id": self.episode_id,
            "info": info,
        }

    def to_jsonl(self) -> str:
        return json.dumps({
            "episode_id": self.episode_id,
            "world_type": self.world_type,
            "system_prompt": self.system_prompt,
            "state": self.state,
            "questions": [q.to_dict() for q in self.questions],
            "rules": list(self.rules),
            "mutations": list(self.mutations),
            "optimal_turns": self.optimal_turns,
            "difficulty": self.difficulty,
            "depth": self.depth,
            "breadth": self.breadth,
        })


@dataclass
class TrainingExample:
    """Back-compat single-question example.

    Existing world generators (abstract_graph, task_dag, etc.) return this shape.
    The dataset layer wraps each one into a TrainingEpisode with len(questions)==1.
    Retained verbatim from parallel_tools so migration stays additive.
    """

    example_id: int
    world_type: str
    system_prompt: str
    user_query: str
    state: dict
    optimal_turns: int
    expected_answer: int | str | bool
    answer_type: str
    difficulty: int
    depth: int
    breadth: int
    query_template: str

    # Optional enrichments: only populated by newer generators.
    target_entities: list[str] = field(default_factory=list)
    parameter_refs: list[ParameterRef] = field(default_factory=list)
    rules: list[dict] = field(default_factory=list)

    def to_episode(self) -> TrainingEpisode:
        q = Question(
            template_name=self.query_template,
            query_text=self.user_query,
            expected_answer=str(self.expected_answer),
            answer_type=self.answer_type,
            target_entities=list(self.target_entities),
            parameter_refs=list(self.parameter_refs),
            optimal_turns=self.optimal_turns,
        )
        return TrainingEpisode(
            episode_id=self.example_id,
            world_type=self.world_type,
            system_prompt=self.system_prompt,
            state=self.state,
            questions=[q],
            rules=list(self.rules),
            optimal_turns=self.optimal_turns,
            difficulty=self.difficulty,
            depth=self.depth,
            breadth=self.breadth,
        )

    def to_dataset_row(self) -> dict:
        return self.to_episode().to_dataset_row()

    def to_jsonl(self) -> str:
        return self.to_episode().to_jsonl()


# =============================================================================
# Query templates + WorldGenerator base (unchanged interface)
# =============================================================================


@dataclass
class QueryTemplate:
    """Describes a query template with its characteristics."""

    name: str
    description: str
    min_turns: int
    max_turns: int
    answer_type: str


class WorldGenerator(ABC):
    """Abstract base class for world generators."""

    world_type: str

    @abstractmethod
    def get_query_templates(self) -> list[QueryTemplate]:
        """Return the list of available query templates for this world."""
        pass

    @abstractmethod
    def generate_state(
        self,
        depth: int,
        breadth: int,
        rng: Any,
        *,
        difficulty: int | None = None,
    ) -> dict:
        """Generate a random world state with the given parameters.

        ``difficulty`` (1–5) is the new primary scaling axis for the
        context-management families (rule_hunt, corpus_dive, timeline_track).
        When set, it supersedes ``depth``/``breadth`` for those generators.
        Older world generators ignore it (kept on the signature for uniform
        dispatch from ``dataset.generate_example``).
        """
        pass

    @abstractmethod
    def generate_query(
        self,
        template: QueryTemplate,
        state: dict,
        target_turns: int,
        rng: Any,
    ) -> tuple[str, Any, int, int]:
        """Generate a specific query from a template.

        Returns: (query_string, expected_answer, actual_depth, actual_breadth)
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt describing this world's tools."""
        pass

    # --- Optional hooks subclasses can override to expose richer info. -------

    def generate_query_rich(
        self,
        template: QueryTemplate,
        state: dict,
        target_turns: int,
        rng: Any,
    ) -> dict:
        """Return a dict with at least {query_text, expected_answer, actual_depth,
        actual_breadth, target_entities, parameter_refs}.

        Default: wraps `generate_query` with empty target_entities/parameter_refs.
        Overriding this lets a world advertise its BFS starting points and any
        deferred-parameter refs without having to reshape every call site.
        """
        q, ans, d, b = self.generate_query(template, state, target_turns, rng)
        return {
            "query_text": q,
            "expected_answer": ans,
            "actual_depth": d,
            "actual_breadth": b,
            "target_entities": [],
            "parameter_refs": [],
        }

