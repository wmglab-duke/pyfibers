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
## Usage
### Creating a fiber model
Use the FiberBuilder class to create fiber object. This fiber object consists of NEURON sections with the proper mechanisms applied, which can then be run in simulations using our provided `Stimulation` class, or in your own custom simulation setup.

```python
# create fiber
fiber = FiberBuilder.generate(
    fiber_model=FiberModel.MRG_DISCRETE,
    n_sections=133,
    temperature=37,
    diameter=5.7,
)
```
### Running a Simulatiion
Once you have a fiber object, you can create a `Stimulation` instance. Provide a list of potential values along the fiber sections, and a time varying waveform.
```python
# Create instance of Stimulation class
stimulation = Stimulation(
    fiber, waveform=waveform, potentials=potentials, dt=time_step, tstop=time_stop
)

# run threshold search
amp, _ = stimulation.find_threshold()
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
