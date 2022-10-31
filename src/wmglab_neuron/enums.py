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
    SCHILD94 = 6


@unique
class ThresholdCondition(Enum):
    """ThresholdCondition."""

    ACTIVATION = 0
    BLOCK = 1


@unique
class BoundsSearchMode(Enum):
    """Bounds search modes."""

    PERCENT_INCREMENT = 0
    ABSOLUTE_INCREMENT = 1


@unique
class TerminationMode(Enum):
    """Termination modes."""

    PERCENT_DIFFERENCE = 0
    ABSOLUTE_DIFFERENCE = 1
