"""Example use case of pyfibers.

NOTE this is for development only
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from pyfibers import FiberModel, build_fiber  # noqa: E402
from pyfibers.stimulation import IntraStim  # noqa: E402

sys.path.append(r'C:\nrn\lib\python')  # noqa: E800


n_nodes = 21

model = FiberModel.MRG_INTERPOLATION  # type of fiber model

# create fiber
fiber = build_fiber(diameter=5.7, fiber_model=model, temperature=37, n_nodes=n_nodes, passive_end_nodes=2)

# parameters
time_step = 0.005  # timestep
time_stop = 15  # duration of simulation

# Create instance of ScaledStim class
stimulation = IntraStim(dt=time_step, tstop=time_stop, istim_loc=0.5)

# turn on saving gating parameters and Vm before running the simulations for thresholds
fiber.record_gating()
fiber.record_vm()

# run threshold search
amp, ap = stimulation.find_threshold(fiber, stimamp_top=1, stimamp_bottom=0)

# report the threshold amplitude with print statement to terminal
print(f'Threshold for 5.7 micron {model}: {amp} (mA)')

# run a finite amp (i.e., one amplitude, not in a bisection search as was done above)
# ap, time = stimulation.run_sim(-1, fiber) # noqa: E800

plt.figure()
for key, value in fiber.gating.items():
    plt.plot(stimulation.time, value[fiber.loc_index(0.5)], label=key)
plt.legend()
ax2 = plt.gca().twinx()
plt.sca(ax2)
plt.plot(stimulation.time, fiber.vm[fiber.loc_index(0.5)], label='vm', color='k')
plt.ylabel('Vm')
plt.xlim(0, 4)

str(fiber)
