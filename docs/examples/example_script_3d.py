"""Example use case of pyfibers.

NOTE this is for development only
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from pyfibers import FiberModel, ScaledStim, build_fiber_3d  # noqa: E402

sys.path.append(r'C:\nrn\lib\python')  # noqa: E800


# %% Define a custom 3D path for the fiber

# Define parameters for the spiral
height = 1000  # total height of the spiral
turns = 5  # number of turns in the spiral
radius = 1000  # radius of the spiral
points_per_turn = 20  # points per turn

# Generate the spiral coordinates
t = np.linspace(0, 2 * np.pi * turns, points_per_turn * turns)
x = radius * np.cos(t)
y = radius * np.sin(t)
z = np.linspace(0, height, points_per_turn * turns)

# Combine into a single array for path coordinates
path_coordinates = np.column_stack((x, y, z))

# Plot the generated path to visualize it
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.set_title("3D Spiral Path for Fiber")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
plt.show()

# Calculate the length along the fiber
fiber_length = np.sqrt(np.sum(np.diff(path_coordinates, axis=0) ** 2, axis=1)).sum()

# Generate the Gaussian curve of potentials
# Set the mean at the center of the fiber and standard deviation as a fraction of the fiber length
mean = fiber_length / 2
std_dev = fiber_length / 50  # Smaller value gives a narrower peak

# Create an array representing the linear distance along the fiber
linear_distance = np.linspace(0, fiber_length, len(path_coordinates))

# Generate Gaussian distributed potentials
potentials = np.exp(-((linear_distance - mean) ** 2 / (2 * std_dev**2)))

# Normalize potentials for better visualization or use
potentials /= np.max(potentials)

potentials *= 500

# Plotting the Gaussian distribution of potentials along the fiber
plt.figure()
plt.plot(linear_distance, potentials, label='Gaussian Potentials')
plt.xlabel('Distance along fiber (arbitrary units)')
plt.ylabel('Potential (normalized)')
plt.title('Gaussian Distribution of Potentials along Fiber')
plt.legend()
plt.show()
# %%
# Choose a fiber model
model = FiberModel.MRG_INTERPOLATION

# Create the fiber using the 3D path
fiber = build_fiber_3d(
    diameter=10, fiber_model=model, temperature=37, path_coordinates=path_coordinates, passive_end_nodes=2
)

# Resample the potentials to match the fiber coordinates
resampled_potentials = fiber.resample_potentials_3d(
    potentials=potentials, potential_coords=path_coordinates, center=True, inplace=True
)
plt.figure()
plt.plot(fiber.longitudinal_coordinates, resampled_potentials, label='Resampled Potentials', marker='o')
plt.legend()

# # Calculate point source potentials at all fiber coordinates
x, y, z = 800, 800, 500  # Point source location
i0 = 1  # Current of the point source
sigma = 1  # Anisotropic conductivity
point_source_potentials = fiber.point_source_potentials(x, y, z, i0, sigma)
fiber.potentials = point_source_potentials

plt.figure()
plt.plot(fiber.longitudinal_coordinates, fiber.potentials, label='Point Source Potentials', marker='o')
plt.legend()

# Additional setup for simulation using ScaledStim class
waveform = np.concatenate((np.ones(100), -np.ones(100), np.zeros(1000000)))
time_step = 0.005
time_stop = 15

# Create instance of ScaledStim
stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

# Setup and run simulation
fiber.record_gating()
fiber.record_vm()
amp, ap = stimulation.find_threshold(fiber)
print(f'Threshold for 5.7 micron {model}: {amp} mA')

# run a finite amp (i.e., one amplitude, not in a bisection search as was done above)
ap, time = stimulation.run_sim(-1, fiber)

plt.figure()
for key, value in fiber.gating.items():
    plt.plot(stimulation.time, value[fiber.loc_index(0.9)], label=key)
plt.legend()
ax2 = plt.gca().twinx()
plt.sca(ax2)
plt.plot(stimulation.time, fiber.vm[fiber.loc_index(0.9)], label='vm', color='k')
plt.ylabel('Vm')
plt.xlim(0, 4)
