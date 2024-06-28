# Extending Functionality

##  How to Create a New Fiber Model

Creating a new fiber model involves defining a subclass that inherits from either `_HomogeneousFiber` or `HeterogeneousFiber`, depending on whether the fiber model has homogeneous or heterogeneous sections. Below are the simplified steps and examples for each type of fiber.

### Creating a New Homogeneous Fiber Model

Homogeneous fibers have identical sections throughout the model. You can create a new homogeneous fiber model by inheriting from `_HomogeneousFiber`.

#### Steps to Create a New Homogeneous Fiber Model

1. **Inherit from `_HomogeneousFiber`**: Your new class should inherit from `_HomogeneousFiber`.

2. **Initialize the Subclass**: Define the `__init__` method, call the superclass initializer, and set any model-specific parameters.

3. **Define the `generate` Method**: Implement the `generate` method to call `generate_homogeneous` with appropriate arguments.

4. **Define the Node Creation Method**: Implement a static method for creating the specific node type for the fiber model.

#### Example

```python
from neuron import h
from pyfibers import FiberModel
from pyfibers.fiber import _HomogeneousFiber

h.load_file("stdrun.hoc")


class MyHomogeneousFiber(_HomogeneousFiber):
    def __init__(self, fiber_model: FiberModel, diameter: float, **kwargs):
        super().__init__(fiber_model=fiber_model, diameter=diameter, **kwargs)
        self.gating_variables = {
            "h": "h_myModel",
            "m": "m_myModel",
            "n": "n_myModel",
        }
        self.myelinated = False
        self.delta_z = 10.0
        self.v_rest = -65

    def generate(self, n_sections: int, length: float) -> _HomogeneousFiber:
        return self.generate_homogeneous(n_sections, length, self.create_my_node)

    @staticmethod
    def create_my_node(node: h.Section) -> None:
        node.insert("myModel")
        node.Ra = 150
        node.cm = 1.0
        node.ena = 50
        node.ek = -77
```

### Creating a New Heterogeneous Fiber Model

Heterogeneous fibers have different types of sections throughout the model. You can create a new heterogeneous fiber model by inheriting from `HeterogeneousFiber`.

#### Steps to Create a New Heterogeneous Fiber Model

1. **Inherit from `HeterogeneousFiber`**: Your new class should inherit from `HeterogeneousFiber`.

2. **Initialize the Subclass**: Define the `__init__` method, call the superclass initializer, and set any model-specific parameters.

3. **Define the `generate` Method**: Implement the `generate` method to call `generate` with the appropriate function list.

4. **Define Section Creation Methods**: Implement methods to create the various sections (e.g., nodes, myelin, etc.) required for the fiber model.

#### Example

```python
from neuron import h
from pyfibers import FiberModel
from pyfibers.fiber import HeterogeneousFiber

h.load_file("stdrun.hoc")


class MyHeterogeneousFiber(HeterogeneousFiber):
    def __init__(self, fiber_model: FiberModel, diameter: float, **kwargs):
        super().__init__(fiber_model=fiber_model, diameter=diameter, **kwargs)
        self.gating_variables = {
            "h": "h_myModel",
            "m": "m_myModel",
            "s": "s_myModel",
        }
        self.myelinated = True
        self.v_rest = -80

    def generate(self, n_sections: int, length: float) -> HeterogeneousFiber:
        function_list = [
            self.create_node,
            self.create_myelin,
        ]
        return super().generate(n_sections, length, function_list)

    def create_node(self, ind: int) -> h.Section:
        section = h.Section(name=f"node{ind}")
        section.insert("myModel")
        section.cm = 2.5
        section.Ra = 54.7
        return section

    def create_myelin(self, ind: int) -> h.Section:
        section = h.Section(name=f"myelin{ind}")
        section.cm = 0
        section.Ra = 54.7
        return section
```

By following these simplified steps, you can create new fiber models based on the specific characteristics of your desired fiber. Define the necessary methods and parameters that match your fiber model requirements.

### Making your fiber model discoverable as a plugin
Other research groups may wish to create their own fiber models in the PyFibers environment, and publish them as a separate repository. Such fiber models can be made discoverable as plugins which will become automatically available in PyFibers after installation. To make your fiber model discoverable as a plugin, follow these steps:

1. **Create a Python package for your fiber model**: Create a Python package for your fiber model and include the necessary files and classes. We recommend following Python's [packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

2. Specify the entry point for your fiber model. The entry point tells pyfibers that your fiber model should be imported, and where to find the class that defines it. Entry points can be specified in multiple ways, here we will provide an example using pyproject.toml:

```toml
[project.entry-points."pyfibers.plugins"]
my_fiber_model = "my_fiber_model_package.my_fiber_model_module:MyFiberModelClass"
```
3. Make the NEURON mechanisms available to the Python interpreter. This requires (1) compiling the NEURON mechanisms (see PyFibers's compile script for an example of how to do this) and (2) making the compiled mechanisms available to the Python interpreter (by calling neuron.load_mechanisms() in your __init__.py file, see PyFibers's __init__.py for an example of how to do this).

4. **Publish your fiber model package**: Publish your fiber model package to a repository such as PyPI. Once published, other users can install your fiber model package using pip. After installation, your fiber model will be automatically discovered and available for use in PyFibers. The name of the fiber model will be given by the "submodels" attribute of your class (see the instructions for creating fiber models above).

## Custom simulations
This section provides examples of how to run custom simulations using fibers. These examples demonstrate different methods, including directly using NEURON's IClamp and h.run, leveraging the Stimulation class with a custom run_sim function, and using the Stimulation.pre_run_setup method with manual assignment of extracellular potentials.

Before running any simulations, we must create a model fiber. An example of how to do so is given below:
```python
from pyfibers import FiberModel, build_fiber

# Define the parameters for the fiber
fiber_diameter = 5.7
fiber_model = FiberModel.MRG_INTERPOLATION
temperature = 37
n_sections = 133
passive_end_nodes = 2

# Build the fiber
fiber = build_fiber(
    diameter=fiber_diameter,
    fiber_model=fiber_model,
    temperature=temperature,
    n_sections=n_sections,
    passive_end_nodes=passive_end_nodes,
)
```

### Example 1: Custom Simulation Using NEURON IClamp and `h.run`

In this example, we demonstrate how to set up a custom simulation using the NEURON `IClamp` and `h.run` without using the simulation classes included in PyFibers.

#### Steps:

1. **Create and configure the fiber as above.**

2. **Set up and configure intracellular stimulation**: #TODO add import statements for NEURON
    ```python
    stim = h.IClamp(fiber.sections[10](0.5))  # Apply stimulation to the 10th section
    stim.amp = 0.1  # nA
    stim.delay = 1  # ms
    stim.dur = 5  # ms
    ```

3. **Set up recording vectors**:
    ```python
    t_vec = h.Vector().record(h._ref_t)  # Time vector
    v_vec = h.Vector().record(fiber.sections[10](0.5)._ref_v)  # Membrane potential vector
    ```

4. **Run the simulation**:
    ```python
    h.finitialize(-65)
    h.continuerun(20)
    ```

---

### Example 2: Custom Simulation with a Custom `run_sim` Function

In this example, we demonstrate how to set up a custom simulation by providing a custom `run_sim` function to the `Stimulation` class. You could also achieve this by creating a subclass and overriding the `run_sim` method.

#### Steps:

1. **Create and configure the fiber as above**


2. **Define the custom `run_sim` function**:
    ```python
    def custom_run_sim(stimulation, *args, **kwargs):
        print("Running custom simulation:")

        stimulation.pre_run_setup(fiber)

        # Example of a custom simulation loop
        for i in range(int(stimulation.tstop / stimulation.dt)):
            # Custom simulation logic here

            # Advance to next time step
            h.fadvance()

        return 1  # return desired data here.
    ```

3. **Set up the `Stimulation` instance with the custom `run_sim` function**:
    ```python
    from pyfibers import Stimulation

    stimulation = Stimulation(dt=0.001, tstop=20, custom_run_sim=custom_run_sim)

    stimulation.run_sim(fiber)
    ```

---

### Example 3: Custom Simulation Using `Stimulation.pre_run_setup` and Manual Extracellular Potentials

In this example, we demonstrate how to set up a custom simulation by using the `Stimulation.pre_run_setup` method and then manually assigning extracellular potentials in a loop.

#### Steps:

1. **Create and configure the fiber as above**:

2. **Set up the `Stimulation` instance**:
    ```python
    from pyfibers import Stimulation

    stimulation = Stimulation(dt=0.001, tstop=20)

    # Pre-run setup
    stimulation.pre_run_setup(fiber)
    ```

3. **Manually assign extracellular potentials in the simulation loop**:
    ```python
    stimamp = 0.1  # Example stimulation amplitude

    for i in range(int(stimulation.tstop / stimulation.dt)):
        # Manually assign extracellular potentials
        e_stims = [stimamp * pot for pot in fiber.potentials]
        stimulation._update_extracellular(fiber, e_stims)

        # Advance the simulation
        h.fadvance()
    ```

---

These examples demonstrate different ways to set up and run custom simulations using fibers with or without the `Stimulation` class. Each method allows for flexibility in how the simulations are defined and executed.

## Custom Data Recording

This section provides examples of how to record custom data during simulations. These examples demonstrate different methods, including using the fiber method to record a custom variable from all nodes and accessing the `istim` variable from `ScaledStim` to record its current.

### Example 1: Record a Custom Variable from All Nodes

In this example, we demonstrate how to use the fiber method to record a custom variable (e.g., membrane potential) from all nodes.

#### Steps:

1. **Set up recording vectors for a custom variable (e.g., membrane potential) from all nodes**:
    ```python
    custom_var_vecs = [h.Vector().record(node(0.5)._ref_v) for node in fiber.nodes]
    ```

2. **Access the recorded data after running a simulation**:
    ```python
    for i, vec in enumerate(custom_var_vecs):
        print(f"Node {i} membrane potential: {vec.to_python()}")
    ```

### Example 2: Access `istim` Variable from `ScaledStim` and Record Its Current

In this example, we demonstrate how to access the `istim` variable from the `ScaledStim` class and record its current.

#### Steps:

1. **Set up the `ScaledStim` instance**:
    ```python
    from pyfibers import ScaledStim
    import numpy as np

    # Set up the ScaledStim instance
    stimulation = ScaledStim(waveform=waveform, dt=0.001, tstop=20)

    # Add intracellular stimulation
    stimulation.set_intracellular_stim()
    ```

2. **Set up recording for `istim` current**:
    ```python
    istim_current_vec = h.Vector().record(stimulation.istim._ref_i)
    ```

3. **Run the simulation and access the recorded data**:
    ```python
    stimulation.run_sim(stimamp=0.1, fiber=fiber)

    print(f"istim current: {istim_current_vec.to_python()}")
    ```
