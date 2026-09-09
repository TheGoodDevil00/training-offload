"""
tournament package
==================
Overnight Drone-Footage Model Tournament for Offline Annotation.
Sequential successive-halving training, evaluation, and ranking on single GPU (RTX 4050 6GB).
"""

from .config import (
    CandidateConfig,
    TournamentConfig,
    TournamentPreset,
    VRAMProfile,
    get_preset_config,
)

__all__ = [
    "CandidateConfig",
    "TournamentConfig",
    "TournamentPreset",
    "VRAMProfile",
    "get_preset_config",
]
