"""Plugin package."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
from contextlib import suppress

import neuron
from neuron import h

# assign version number from pyproject.toml
__version__ = importlib.metadata.version(__package__ or __name__)

# load all NEURON files
MOD_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MOD')
# call nrnivmodl to compile the mod files
subprocess.run(['nrnivmodl', MOD_dir], shell=True)
with suppress(RuntimeError):
    neuron.load_mechanisms(MOD_dir)
