"""Enums for wmglab_neuron."""
from enum import Enum, unique


@unique
class FiberModel(Enum):
    """Fiber models."""

    MRG_INTERPOLATION = 0
    MRG_DISCRETE = 1
    SUNDT = 2
    TIGERHOLM = 3
    RATTAY = 4
    SCHILD97 = 5


@unique
class Protocol(Enum):
    """Protocol types."""

    ACTIVATION_THRESHOLD = 0
    BLOCK_THRESHOLD = 1
    FINITE_AMPLITUDES = 2
