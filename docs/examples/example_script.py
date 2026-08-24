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

from pyfibers import FiberModel, ScaledStim, build_fiber  # noqa: E402

sys.path.append(r'C:\nrn\lib\python')  # noqa: E800


n_nodes = 21

model = FiberModel.MRG_INTERPOLATION  # type of fiber model

# create fiber
fiber = build_fiber(diameter=5.7, fiber_model=model, temperature=37, n_nodes=n_nodes, passive_end_nodes=2)
fiber.set_xyz(5, 10, -100)

# create curve of potentials
fiber.potentials = fiber.point_source_potentials(0, 2000, fiber.length / 2, 1, 1)
plt.plot(fiber.potentials)

# parameters
time_step = 0.005  # timestep
time_stop = 15  # duration of simulation
# Create callable waveform: 0.05 ms positive, 0.05 ms negative, then zeros
time_points = np.array([0, 0.05, 0.1, time_stop])
waveform_values = np.array([1, -1, 0, 0])
waveform = interp1d(time_points, waveform_values, kind='previous', bounds_error=False, fill_value=0.0)

# Create instance of ScaledStim class
stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

# turn on saving gating parameters and Vm before running the simulations for thresholds
fiber.record_gating()
fiber.record_vm()

# run threshold search
amp, ap = stimulation.find_threshold(fiber, condition="activation", stimamp_top=-200)

# report the threshold amplitude with print statement to terminal
print(f'Threshold for 5.7 micron {model}: {amp} (mA)')

# run a finite amp (i.e., one amplitude, not in a bisection search as was done above)
# ap, time = stimulation.run_sim(-1, fiber) # noqa: E800

plt.figure()
for key, value in fiber.gating.items():
    plt.plot(stimulation.time, value[fiber.loc_index(0.6)], label=key)
plt.legend()
ax2 = plt.gca().twinx()
plt.sca(ax2)
plt.plot(stimulation.time, fiber.vm[fiber.loc_index(0.6)], label='vm', color='k')
plt.ylabel('Vm')
plt.xlim(0, 4)

str(fiber)
