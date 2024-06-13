# isort: skip_file
"""Defines imports for the NEURON_Files module."""
import os

from contextlib import suppress

from neuron import h
import neuron
import importlib.metadata

# assign version number from pyproject.toml
__version__ = importlib.metadata.version(__package__ or __name__)

# load all NEURON files
MOD_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MOD')
with suppress(RuntimeError):
    neuron.load_mechanisms(MOD_dir)

# load all python files
from .fiber import build_fiber, Fiber, FiberModel  # noqa: E402
from .stimulation import ScaledStim, BoundsSearchMode, TerminationMode, ThresholdCondition, BisectionMean  # noqa: E402

__all__ = [
    'build_fiber',
    'Fiber',
    'ScaledStim',
    'FiberModel',
    'BoundsSearchMode',
    'TerminationMode',
    'ThresholdCondition',
    'BisectionMean',
]
