"""Example use case of pyfibers.

NOTE this is for development only
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(r'C:\nrn\lib\python')  # noqa: E800

from pyfibers import FiberModel, build_fiber_3d  # noqa: E402

# %% Define a custom 3D path for the fiber

# Define parameters for the spiral
height = 1000  # total height of the spiral
turns = 5  # number of turns in the spiral
radius = 1000  # radius of the spiral
points_per_turn = 20  # points per turn

# Generate a right angle for fiber path
x = [0, 10000, 10000]
y = [0, 0, 10000]
z = [0, 0, 0]
path_coordinates = np.array([x, y, z]).T
# plot xy

# %%
# Choose a fiber model
model = FiberModel.MRG_INTERPOLATION

# Create the fiber using the 3D path
fiber = build_fiber_3d(
    diameter=16, fiber_model=model, temperature=37, path_coordinates=path_coordinates, passive_end_nodes=2
)
plt.plot(x, y, lw=10, color='gray', alpha=0.3)
plt.title('Fiber path')

# plot fiber coordinates
plt.plot(fiber.coordinates[:, 0], fiber.coordinates[:, 1], marker='o', markersize=5, lw=0)
plt.title('Fiber coordinates')
plt.show()

# %%
# Create the fiber using the 3D path
fiber = build_fiber_3d(
    diameter=16,
    fiber_model=model,
    temperature=37,
    path_coordinates=path_coordinates,
    passive_end_nodes=2,
    center_shift=True,
    shift_ratio=0.5,
)
plt.plot(x, y, lw=10, color='gray', alpha=0.3)
plt.title('Fiber path')

# plot fiber coordinates
plt.plot(fiber.coordinates[:, 0], fiber.coordinates[:, 1], marker='o', markersize=5, lw=0)
# plot node coordinates
plt.plot(fiber.coordinates[:, 0][::11], fiber.coordinates[:, 1][::11], marker='o', markersize=5, lw=0, c='r')
plt.title('Fiber coordinates')
# set aspect equal
plt.gca().set_aspect('equal')
plt.show()
# %%
