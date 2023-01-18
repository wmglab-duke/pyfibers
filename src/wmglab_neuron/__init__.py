# isort: skip_file
"""Defines imports for the NEURON_Files module.

The copyrights of this software are owned by Duke University.
Please refer to the LICENSE and README.md files for licensing instructions.
The source code can be found on the following GitHub repository: https://github.com/wmglab-duke/ascent
"""
from contextlib import suppress

from neuron import h
import neuron

with suppress(RuntimeError):
    h.nrn_load_dll('src/MOD/nrnmech.dll')
    neuron.load_mechanisms('src/MOD')
    print('Loaded neuron mechanism files')

from .enums import FiberModel
from .fiber_z import FiberTypeParameters
from .fiber import FiberBuilder, _Fiber
from .stimulation import Stimulation

__all__ = ['FiberBuilder', 'Recording', 'Stimulation', 'FiberModel', 'FiberTypeParameters']
