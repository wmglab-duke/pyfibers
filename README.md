Note add badges

# wmglab-neuron
This package implements biophysical models of axons in the NEURON simulation environment using Python. You can run simulations of stimulation of axons, with the ability to search for activation and block thresholds. You can also add your own fiber models or custom simulations.

We have implemented the following models:
- MRG
- SUNDT
- TIGERHOLM
- RATTAY

## Installation
*Note that these instructions are temporary for development*
1. Install NEURON and add to PATH
2. Clone the package to your machine
3. Install the package using pip
   - Open anaconda prompt
   - Navigate to the repo root directory
   - `pip install .`
4. To compile the mod files, in anaconda prompt, run the command `wmglab_neuron_install` (Note: if you receive a message that the wmglab_neuron_install command is not found, find the executable for this command in the `bin` path of your python directory and run it there.)
### Migrating from v0.0.1 to v0.0.2
1. Change all usages and imports of the Stimulation class to ScaledStim.
  - `from wmglab_neuron import ScaledStim`
  - `stimulation = ScaledStim(args)`
2. Remove potentials from instantiation of Stimulation, instead add to the relevant fiber (`fiber.potentials=potentials`)
3. Remove fiber as an argument to instantiation of Stimulation, instead pass the fiber when calling:
  - run_sim() (`stimulation.run_sim(stimamp,fiber)`)
  - find_threshold(`stimulation.find_threshold(fiber)`).
4. Replace imports of FiberBuilder and calls of FiberBuilder.generate() with the "build_fiber" function.
  - `from wmglab_neuron import build_fiber`
  - `fiber = build_fiber(args)`
## Usage
### Creating a fiber model
Use the FiberBuilder class to create fiber object. This fiber object consists of NEURON sections with the proper mechanisms applied, which can then be run in simulations using our provided `ScaledStim` class, or in your own custom simulation setup.

```python
# create fiber
fiber = build_fiber(
    fiber_model=FiberModel.MRG_DISCRETE,
    n_sections=133,
    temperature=37,
    diameter=5.7,
)
```
### Running a Simulatiion
Once you have a fiber object, you can create a `ScaledStim` instance. Provide a list of potentials to accompany the fiber and a time-varying waveform.
```python
# Create instance of ScaledStim class
fiber.potentials = potential_values
stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

# run threshold search
amp, _ = stimulation.find_threshold(fiber)
print(f"Threshold for 5.7 micron fiber: {amp} (mA)")
```
For more examples, see the documentation.

## Contributing
If you have a contribution which you feel would be of value, please open a Pull Request.

## Authors and acknowledgment
Based on the original paper by authors
Developed by Elie, Daniel, and Eric

## License
See LICENSE
