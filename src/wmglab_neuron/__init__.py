# isort: skip_file
"""Defines imports for the NEURON_Files module.

The copyrights of this software are owned by Duke University.
Please refer to the LICENSE and README.md files for licensing instructions.
The source code can be found on the following GitHub repository: https://github.com/wmglab-duke/ascent
"""
import os

from contextlib import suppress

from neuron import h
import neuron

MOD_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MOD')

with suppress(RuntimeError):
    h.nrn_load_dll(os.path.join(MOD_dir, 'nrnmech.dll'))
    print('Loaded neuron dll')
    neuron.load_mechanisms(MOD_dir)
    print('Loaded neuron mechanism files')

from .enums import FiberModel  # noqa: E402
from .fiber_z import FiberTypeParameters  # noqa: E402
from .fiber import FiberBuilder, _Fiber  # noqa: E402
from .stimulation import Stimulation  # noqa: E402

__all__ = ['FiberBuilder', '_Fiber', 'Stimulation', 'FiberModel', 'FiberTypeParameters']
