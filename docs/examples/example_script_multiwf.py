"""Example use case of PyFibers.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers

NOTE this is for development only
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from pyfibers import FiberModel, ScaledStim, build_fiber

sys.path.append(r'C:\nrn\lib\python')


nodecount = 133

model = FiberModel.MRG_INTERPOLATION  # type of fiber model

# create fiber
fiber = build_fiber(diameter=5.7, fiber_model=model, temperature=37, n_sections=nodecount)
pots = []
# create curve of potentials
for spot in [100, -100]:
    pots.append(fiber.point_source_potentials(0, 250, spot + fiber.length / 2, 1, 1))

fiber.potentials = np.vstack(pots)

plt.plot(fiber.potentials[0, :])
plt.plot(fiber.potentials[1, :])

# parameters
time_step = 0.001  # timestep
time_stop = 15  # duration of simulation
# Create callable waveforms: waveform1 is 0.05 ms positive then zeros, waveform2 is 0.2 ms positive then zeros
waveform = [
    interp1d(np.array([0, 0.05, time_stop]), np.array([1, 0, 0]), kind='previous', bounds_error=False, fill_value=0.0),
    interp1d(np.array([0, 0.2, time_stop]), np.array([1, 0, 0]), kind='previous', bounds_error=False, fill_value=0.0),
]

# Create instance of ScaledStim class
stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

# turn on saving gating parameters and Vm before running the simulations for thresholds
fiber.record_gating()
fiber.record_vm()

# run threshold search
amp, ap = stimulation.find_threshold(fiber)

# report the threshold amplitude with print statement to terminal
print(f'Threshold for 5.7 micron {model}: {amp} (mA)')

# run a finite amp (i.e., one amplitude, not in a bisection search as was done above)
ap, time = stimulation.run_sim(-1, fiber)

# run with multiple stim amps
ap, time = stimulation.run_sim([-1, 1], fiber)
