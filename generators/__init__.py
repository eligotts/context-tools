"""context-tools training data generator.

Synthetic data generators for the 5 context-management task families:
``rule_hunt``, ``corpus_dive``, ``timeline_track``, ``detective``,
``maze_walk``. Each family targets a distinct scratchpad-management
mechanic (edit / prune / overwrite / shrink-set / push-pop).
"""

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
    CONTEXT_MGMT_WORLD_WEIGHTS,
)
from .rule_hunt import RuleHuntWorld
from .corpus_dive import CorpusDiveWorld
from .timeline_track import TimelineTrackWorld
from .detective import DetectiveWorld
from .maze_walk import MazeWalkWorld
from .adaptive_cursor import AdaptiveCursorWorld

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
    "CONTEXT_MGMT_WORLD_WEIGHTS",
    # World generators
    "RuleHuntWorld",
    "CorpusDiveWorld",
    "TimelineTrackWorld",
    "DetectiveWorld",
    "MazeWalkWorld",
    "AdaptiveCursorWorld",
]
