"""Example use case of wmglab_neuron.

NOTE this is for development only
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(r'C:\nrn\lib\python')  # noqa: E800

from wmglab_neuron import FiberModel, ScaledStim, build_fiber  # noqa: E402

nodecount = 133

model = FiberModel.MRG_INTERPOLATION  # type of fiber model

# create fiber
fiber = build_fiber(diameter=5.7, fiber_model=model, temperature=37, n_sections=nodecount)

# create curve of potentials
fiber.potentials = fiber.point_source_potentials(0, 250, fiber.length / 2, 1, 1)
plt.plot(fiber.potentials)

# create biphasic square wave to use as a stimulation waveform
waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(49600)))

# parameters
time_step = 0.001  # timestep|
time_stop = 15  # duration of simulation

# Create instance of ScaledStim class
stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

# turn on saving gating parameters and Vm before running the simulations for thresholds
fiber.set_save_gating()
fiber.set_save_vm()

# run threshold search
amp, ap = stimulation.find_threshold(fiber)

# report the threshold amplitude with print statement to terminal
print(f'Threshold for 5.7 micron {model}: {amp} (mA)')

# run a finite amp (i.e., one amplitude, not in a bisection search as was done above)
ap, time = stimulation.run_sim(-1, fiber)
