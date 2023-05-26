"""Example use case of wmglab_neuron.

NOTE this is for development only
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

sys.path.append(r'C:\nrn\lib\python')

from wmglab_neuron import FiberModel, ScaledStim, build_fiber  # noqa: E402

nodecount = 133

model = FiberModel.MRG_INTERPOLATION

# create fiber
fiber = build_fiber(diameter=5.7, fiber_model=model, temperature=37, n_sections=nodecount)

# create curve of potentials
fiber.potentials = norm.pdf(np.linspace(-1, 1, nodecount), 0, 0.05) * 100
plt.plot(fiber.potentials)
# create biphasic square wave
waveform = np.concatenate((np.ones(200), -np.ones(200), np.zeros(49600)))

# parameters
time_step = 0.001
time_stop = 50

# Create instance of ScaledStim class
stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

fiber.set_save_gating()
fiber.set_save_vm()

# run threshold search
amp, ap = stimulation.find_threshold(fiber, stimamp_top=-1, stimamp_bottom=-0.1)

print(f'Threshold for 5.7 micron {model}: {amp} (mA)')

# run a finite amp
ap, time = stimulation.run_sim(-1, fiber)
