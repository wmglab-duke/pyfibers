# isort: skip_file
"""Initializer for core PyFibers code."""
import os

from contextlib import suppress

import neuron
import importlib.metadata

# assign version number from pyproject.toml
__version__ = importlib.metadata.version(__package__ or __name__)

# load all NEURON files
MOD_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MOD')
with suppress(RuntimeError):
    neuron.load_mechanisms(MOD_dir)

# load all python files
from .fiber import Fiber, build_fiber, build_fiber_3d  # noqa: E402
from .model_enum import FiberModel, register_custom_fiber  # noqa: E402
from .stimulation import (  # noqa: E402
    ScaledStim,
    IntraStim,
    Stimulation,
    BoundsSearchMode,
    TerminationMode,
    ThresholdCondition,
    BisectionMean,
)

__all__ = [
    'build_fiber',
    'build_fiber_3d',
    'Fiber',
    'ScaledStim',
    'IntraStim',
    'Stimulation',
    'FiberModel',
    'register_custom_fiber',
    'BoundsSearchMode',
    'TerminationMode',
    'ThresholdCondition',
    'BisectionMean',
]
