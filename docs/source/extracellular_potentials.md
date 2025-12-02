# Supplying Extracellular Potentials

PyFibers works with extracellular potentials imposed by stimulation electrodes. Extracellular potentials can be derived from a number of sources. The two primary methods are analytical calculations (e.g., point source approximation) and numerical simulations (e.g., finite element method). When using numerical simulations, the extracellular potentials are calculated in a separate software package (e.g., COMSOL) and then imported into PyFibers. Typically, the extracellular potentials are calculated using a unit stimulus (e.g., 1 mA) and then scaled to the desired stimulus amplitude (e.g., the stimamp parameter of {py:meth}`~pyfibers.stimulation.ScaledStim.run_sim`).

## Analytical Calculations

Each fiber class instance has a method {py:meth}`~pyfibers.fiber.Fiber.point_source_potentials` that calculates the extracellular potentials due to a point source in an infinite homogeneous volume conductor. For the isotropic case:

```python
pt_x = 0  # μm
pt_y = 0  # μm
pt_z = fiber.length / 2  # μm
I = 1  # mA, unit stimulus
sigma = 0.3  # S/m
# Calculate extracellular potentials
fiber.point_source_potentials(pt_x, pt_y, pt_z, I, sigma, inplace=True)
```

For the anisotropic case, provide the conductivity as a tuple of three conductivities:

```python
sigma_x = 0.3  # S/m
sigma_y = 0.3  # S/m
sigma_z = 0.1  # S/m
fiber.point_source_potentials(
    pt_x, pt_y, pt_z, I, (sigma_x, sigma_y, sigma_z), inplace=True
)
```

## Numerical Simulations (e.g., Finite Element Models)

Extracellular potentials can be calculated using numerical methods such as the finite element method (FEM) in software packages like COMSOL, ANSYS, or FEniCS. These simulations provide high-resolution potential distributions that can be imported into PyFibers for fiber simulation studies. Such electrical potentials from external sources can be used in PyFibers with two main approaches:

1. **Resampling (Recommended)**: Interpolate the high-resolution FEM potentials onto the fiber's coordinate system using the provided resampling methods in PyFibers. This is the recommended approach for most use cases, as it allows for flexible fiber discretization (testing new longitudinal alignments or fiber diameters on the same path).

2. **Direct Assignment**: Assign the FEM-computed potentials directly to the fiber if the potentials are already sampled at the same locations as the fiber's sections.

### Resampling

```{note}
It is usually easiest to use a 1D fiber (even if your fiber follows a 3D path) when resampling potentials. In this case, calculate the arc lengths along your 3D trajectory and use those as the coordinates for resampling the potentials onto the fiber. This approach allows you to use the flexible resampling tools in PyFibers regardless of the actual 3D geometry.
```

When importing potentials from finite element models, follow these key steps:

1. **Use unit stimulus**: Calculate potentials using a unit current (e.g., 1 mA) in your FEM simulation. PyFibers will scale these potentials to the desired stimulation amplitude.

2. **Obtain potentials at high resolution**: Your FEM simulation should output potentials at many points along the fiber path (e.g., every 10 um along the fiber path) for smooth interpolation.

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
   fiber.resample_potentials(arc_lengths_fem, potentials_fem, inplace=True)
   ```

### Direct Assignment

You can also directly assign your FEM potentials to the fiber if they are already sampled at the correct locations:

```{note}
If you do not plan to resample (i.e., your FEM potentials are already computed at the exact locations of the fiber sections), it is often easier to use a 3D fiber model—unless your path is not truly 3D. In this case, you can directly assign the potentials to the fiber sections, as the coordinates will already match.
```

```python
# Get the coordinates where you need potentials
coords = fiber.coordinates  # Get from fiber
potentials = np.loadtxt("my_electrical_potentials.txt")  # Get from FEM
fiber.potentials = potentials
```

```{caution}
The length of `fiber.sections` and the potentials array must match for direct assignment.
```

## Electrical Potentials from Multiple Stimulation Sources

```{note}
PyFibers supports one stimulation amplitude for scaling input potentials (as an argument to {py:meth}`~pyfibers.stimulation.ScaledStim.run_sim`). Threshold searches using {py:meth}`~pyfibers.stimulation.Stimulation.find_threshold` similarly provide a single "stimamp" input to {py:meth}`~pyfibers.stimulation.ScaledStim.run_sim`. To scale potentials from multiple sources independently, you must either scale the potentials as you provide them to the fiber object, or use your own custom `run_sim()` method (see [Custom Simulations](custom_stim.md)).
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

The mathematical process which PyFibers uses to calculate the extracellular potentials at runtime is described in our [algorithms documentation](algorithms.md).
