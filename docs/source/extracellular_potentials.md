# Supplying Extracellular Potentials
PyFibers works with extracellular potentials imposed by stimulation electrodes. Extracellular potentials can be derived from a number of sources. The two primary methods are analytical calculations (e.g., point source approximation) and numerical simulations (e.g., finite element method). When using numerical simulations, the extracellular potentials are calculated in a separate software package and then imported into PyFibers.

## Analytical Calculations
Each fiber class instance has a method point_source_potentials that calculates the extracellular potentials due to a point source in an infinite homogeneous volume conductor. For the isotropic case:
```python
pt_x = 0  # μm
pt_y = 0  # μm
pt_z = fiber.length / 2  # μm
I = 1  # mA
sigma = 0.3  # S/m
# Calculate extracellular potentials
fiber.point_source_potentials(pt_x, pt_y, pt_z, I, sigma, inplace=True)
```
where x, y, z are the coordinates of the point source and I is the current amplitude. The method returns the extracellular potentials at the center of each section of the fiber.

For the anisotropic case, provide the conductivity as a tuple of three conductivities:
```python
sigma_x = 0.3  # S/m
sigma_y = 0.3  # S/m
sigma_z = 0.1  # S/m
fiber.point_source_potentials(
    pt_x, pt_y, pt_z, I, (sigma_x, sigma_y, sigma_z), inplace=True
)
```

## Numerical Simulations
Extracellular potentials must be obtained at the correct spacing along the fiber path, where each potential is at the correct distance along the fiber path corresponding to the center of each section comprising the fiber.
If the potentials were obtained at the correct spacing, they can be directly supplied via an attribute,
```python
fiber.potentials = np.array([0, 1, 2, 3, 2, 1, 0])
```
or, the potentials can be linearly interpolated to the correct spacing using the method `resample_potentials`:
```python
arc_lengths = np.array([0, 1, 2, 3, 4, 5, 6])
potentials = np.array([0, 1, 2, 3, 2, 1, 0])
fiber.resample_potentials(arc_lengths, potentials, inplace=True)
```

## Electrical potentials from multiple sources
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

### Different waveforms for each source
In the case where each source delivers a different stimulus (for example, one stimulus is a square wave and the other is a sinusoidal wave), the potentials must be weighted during each time step of the simulation. Thus, the potentials must be provided as a 2D array, where each row corresponds to the potentials from a different source.
```python
potentials_array_1 = np.array([1, 2, 3, 2, 1, 0, 0])
potentials_array_2 = np.array([0, 0, 1, 2, 3, 2, 1])
potentials_array = np.vstack((potentials_array_1, potentials_array_2))
```

Similarly, multiple waveforms must then be provided to the ScaledStim class instance, where each row corresponds to the waveform from a different source.
```python
waveform1 = np.array([1, 2, 3, 2, 1, 0, 0])
waveform2 = np.array([0, 0, 1, 2, 3, 2, 1])
waveforms = np.vstack((waveform1, waveform2))
stimulation = ScaledStim(fiber, waveforms)
```

The mathematical process which PyFibers uses to calculate the extracellular potentials at runtime is described in our paper:
(link to paper)[]
