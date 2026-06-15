# Supplying extracellular potentials

PyFibers works with extracellular potentials imposed by stimulation electrodes. Extracellular potentials can be derived from analytical calculations or numerical simulations (e.g., finite element method). Typically, potentials are calculated using a unit stimulus (e.g., 1 mA) and then scaled to the desired stimulus amplitude.

:::{seealso}
PyFibers is integrated as the **fiber backend** of **ASCENT** (Automated Simulations to Characterize Electrical Nerve Thresholds) {cite:p}`musselman_ascent_2021`. ASCENT repository: [github.com/wmglab-duke/ascent](https://github.com/wmglab-duke/ascent).
:::

## Overview

PyFibers supports two primary approaches for extracellular potentials:

1. **Analytical calculations**: built-in point source calculations.
2. **Numerical simulations**: use potentials from external FEM or other field solvers for complex geometries.

## Analytical calculations

Each fiber class instance has a method {py:meth}`~pyfibers.fiber.Fiber.point_source_potentials` that calculates extracellular potentials due to a point source in an infinite homogeneous volume conductor.

### Isotropic case
```python
pt_x = 0  # μm
pt_y = 0  # μm
pt_z = fiber.length / 2  # μm
I = 1  # mA, unit stimulus
sigma = 0.3  # S/m
fiber.point_source_potentials(pt_x, pt_y, pt_z, I, sigma, inplace=True)
```

### Anisotropic case
```python
sigma_x = 0.3  # S/m
sigma_y = 0.3  # S/m
sigma_z = 0.1  # S/m
fiber.point_source_potentials(
    pt_x, pt_y, pt_z, I, (sigma_x, sigma_y, sigma_z), inplace=True
)
```

## Numerical simulations (e.g., finite element models)

Extracellular potentials can be calculated using numerical methods such as the finite element method (FEM) in third-party modeling software (e.g., COMSOL, FEniCS) and then imported into PyFibers. Such electrical potentials from external sources can be used in PyFibers with two main approaches:

1. **Resampling (recommended)**: From your solver, generate high-resolution (small spacing) potentials along the length of your fiber. PyFibers can then interpolate these high-resolution potentials onto the fiber's coordinate system. This is the recommended approach for most use cases, as it allows for flexible fiber discretization (testing new longitudinal alignments or fiber diameters on the same path).

2. **Direct assignment**: assign the FEM-computed potentials directly to the fiber if the potentials are already sampled at the same locations as the center the fiber's sections.

### Resampling

:::{seealso}
{doc}`tutorials/4_resampling_potentials` shows a step-by-step resampling workflow.
:::

```{note}
It is usually easiest to use a 1D fiber (even if your fiber follows a 3D path) when resampling potentials. In this case, calculate the arc lengths along your 3D trajectory and use those as the coordinates for resampling the potentials onto the fiber. This approach allows you to use the flexible resampling tools in PyFibers regardless of the actual 3D geometry.
```

When importing potentials from finite element models, follow these key steps:

1. **Use unit stimulus**: Calculate potentials using a unit reference (e.g., 1 mA) in your FEM simulation. PyFibers scales these via ``stimamp`` in :class:`~pyfibers.stimulation.ScaledStim`.

2. **Obtain potentials at high resolution**: Your FEM simulation should output potentials at many points along the fiber path (e.g., every 10 µm along the fiber path) for smooth interpolation.

3. **Resample to fiber coordinates**: Ensure your coordinates represent the distance along the fiber path, not 3D Cartesian coordinates. If your FEM output provides 3D coordinates, convert them to arc-length. Then use {py:meth}`~pyfibers.fiber.Fiber.resample_potentials` to interpolate the high-resolution potentials onto your fiber's coordinate system.
   ```python
   # Example: Convert 3D coordinates to arc-length
   from scipy.spatial.distance import euclidean

   coords_3d = np.array([[x1, y1, z1], [x2, y2, z2], ...])  # From FEM
   arc_lengths = np.zeros(len(coords_3d))
   arc_lengths[1:] = np.cumsum(
       [euclidean(coords_3d[i - 1], coords_3d[i]) for i in range(1, len(coords_3d))]
   )
   # Resample the high-resolution FEM potentials onto the fiber's coordinates
   fiber.resample_potentials(potentials_fem, arc_lengths_fem, inplace=True)
   ```

### Direct assignment

You can also directly assign your FEM potentials to the fiber if they are already sampled at the correct locations:

```{note}
If you do not plan to resample (i.e., your FEM potentials are already computed at the exact locations of the fiber sections), it is often easier to use a 3D fiber model—unless your path is truly 1D. In this case, you can directly assign the potentials to the fiber sections, as the coordinates will already match.
```

```python
# Get the coordinates where you need potentials
coords = fiber.coordinates  # Get from fiber
# Use your method to get electrical potentials at the fiber coordinates.
potentials = np.loadtxt("my_electrical_potentials.txt")  # Get from FEM
fiber.potentials = potentials
```

```{caution}
The length of `fiber.sections` and the potentials array must match for direct assignment.
```

OR, if your FEM potentials are already at the correct spacing:
```python
fiber.potentials = np.loadtxt("fem_results.csv")
```

:::{seealso}
{doc}`tutorials/8_fiber_paths_3d` shows how to use curved 3D fiber paths and align FEM potentials to the fiber.
:::

## Electrical Potentials from Multiple Stimulation Sources

```{note}
Stimulation amplitudes passed to {py:meth}`~pyfibers.stimulation.ScaledStim.run_sim` for multiple sources can be a single float that is applied to all sources, or a list of floats for each source. At present, threshold searches using {py:meth}`~pyfibers.stimulation.Stimulation.find_threshold` only support passing a single "stimamp" input to {py:meth}`~pyfibers.stimulation.ScaledStim.run_sim`. To scale potentials from multiple sources independently during a threshold search, you must either scale them in advance using superposition (see below), or create a custom threshold search function (see {doc}`Algorithms in PyFibers <algorithms>`).
```

### Superposition

In the case where each source delivers a weighted version of the same stimulus, the potentials can be weighted and summed under the principle of superposition. For example, the potentials due to two point sources of opposite polarity can be calculated as:

```python
pt_x1 = 0  # μm
pt_y1 = 0  # μm
pt_z1 = fiber.length / 2 + 100  # μm
I1 = 1  # mA
sigma1 = 0.3  # S/m
pt_x2 = 0  # μm
pt_y2 = 0  # μm
pt_z2 = fiber.length / 2 - 100  # μm
I2 = -1  # mA
sigma2 = 0.3  # S/m
# Calculate extracellular potentials
fiber.potentials = fiber.point_source_potentials(pt_x1, pt_y1, pt_z1, I1, sigma1)
fiber.potentials += fiber.point_source_potentials(pt_x2, pt_y2, pt_z2, I2, sigma2)
```

For potentials from an external source, they can be similarly weighted:

```python
potentials_array_1 = np.array([1, 2, 3, 2, 1, 0, 0])
potentials_array_2 = np.array([0, 0, 1, 2, 3, 2, 1])
fiber.potentials = 1 * potentials_array_1 + -1 * potentials_array_2
```

### Different Waveforms for Each Source

In the case where each source delivers a different stimulus (for example, one stimulus is a square wave and the other is a sinusoidal wave), the potentials must be weighted during each time step of the simulation. Thus, the potentials must be provided as a 2D array, where each row corresponds to the potentials from a different source.

```python
potentials_array_1 = np.array([1, 2, 3, 2, 1, 0, 0])
potentials_array_2 = np.array([0, 0, 1, 2, 3, 2, 1])
potentials_array = np.vstack((potentials_array_1, potentials_array_2))
```

Similarly, multiple waveforms must then be provided to the {py:class}`~pyfibers.stimulation.ScaledStim` class instance as a list, where each callable in the list corresponds to the waveform from a different source.

```python
waveform = scipy.interpolate.interp1d(
    [0, 0.1, 0.2, time_stop], [0, 1, 0, 0], kind="previous"
)
waveform = scipy.interpolate.interp1d(
    [0, 0.1, 0.2, time_stop], [0, -1, 0, 0], kind="previous"
)
stimulation = ScaledStim(fiber, [waveform1, waveform2])
```

## Algorithm details

:::{seealso}
{doc}`algorithms` explains how {py:class}`~pyfibers.stimulation.ScaledStim` scales spatial potentials with waveforms each time step, including multiple sources and superposition.
:::
