"""context-tools training data generators."""

from .base import (
    ParameterRef,
    Question,
    QueryTemplate,
    TrainingEpisode,
    TrainingExample,
    WorldGenerator,
)
from .dataset import (
    generate_dataset,
    save_dataset,
    save_metadata,
    load_dataset,
    export_for_verifiers,
    DEFAULT_WORLD_WEIGHTS,
    RULE_HUNT_WORLD_WEIGHTS,
    CORPUS_DIVE_WORLD_WEIGHTS,
    TIMELINE_TRACK_WORLD_WEIGHTS,
    DETECTIVE_WORLD_WEIGHTS,
    MAZE_WALK_WORLD_WEIGHTS,
    ADAPTIVE_CURSOR_WORLD_WEIGHTS,
    CORPUS_TRAIL_WORLD_WEIGHTS,
    CONTEXT_MGMT_WORLD_WEIGHTS,
)
from .rule_hunt import RuleHuntWorld
from .corpus_dive import CorpusDiveWorld
from .timeline_track import TimelineTrackWorld
from .detective import DetectiveWorld
from .maze_walk import MazeWalkWorld
from .adaptive_cursor import AdaptiveCursorWorld
from .corpus_trail import CorpusTrailWorld

__all__ = [
    # Schema
    "ParameterRef",
    "Question",
    "QueryTemplate",
    "TrainingEpisode",
    "TrainingExample",
    "WorldGenerator",
    # Dataset
    "generate_dataset",
    "save_dataset",
    "save_metadata",
    "load_dataset",
    "export_for_verifiers",
    "DEFAULT_WORLD_WEIGHTS",
    "RULE_HUNT_WORLD_WEIGHTS",
    "CORPUS_DIVE_WORLD_WEIGHTS",
    "TIMELINE_TRACK_WORLD_WEIGHTS",
    "DETECTIVE_WORLD_WEIGHTS",
    "MAZE_WALK_WORLD_WEIGHTS",
    "ADAPTIVE_CURSOR_WORLD_WEIGHTS",
    "CORPUS_TRAIL_WORLD_WEIGHTS",
    "CONTEXT_MGMT_WORLD_WEIGHTS",
    # World generators
    "RuleHuntWorld",
    "CorpusDiveWorld",
    "TimelineTrackWorld",
    "DetectiveWorld",
    "MazeWalkWorld",
    "AdaptiveCursorWorld",
    "CorpusTrailWorld",
]
