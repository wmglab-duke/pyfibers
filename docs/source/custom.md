# Extending Functionality

## How to Create a New Fiber Model

Creating a new fiber model involves defining a subclass of `Fiber` and implementing methods that describe the specific mechanisms and ultrastructure of the fiber. This section provides a step-by-step guide on how to create a new fiber model.

### Steps to Create a New Fiber Model

Note that these steps will differ if you are creating a "homogeneous" fiber model (i.e., all sections of the fiber are identical, typically unmyelinated fibers) or a "heterogeneous" fiber model (i.e., sections of the fiber have different properties, such as nodes and myelin). We will begin with an example of creating a homogeneous fiber model and then extend it to a heterogeneous fiber model.

1. **Inherit from `Fiber`**: Your new class should inherit from `Fiber`.

2. **Specify the Submodels**: Define the `submodels` attribute as a list of the submodels that your fiber model uses. Often, this list will contain only a single item. Each item in the list should be capitalized and only letters and underscores; these strings are used to define the name of each fiber model as accessed from the FiberModel enum (in the example below, one would access the FiberModel by calling `FiberModel.MY_HOMOGENEOUS_FIBER_MODEL`). If your class has multiple subtypes (such as in ```{py:class} pyfibers.models.MRGFiber```), you can define multiple submodels. When you create an instance of your fiber model, it will gain the enum as the self.fiber_model attribute. Check the name by accessing self.fiber_model.name.

2. **Initialize the Subclass**: Define the `__init__` method, call the superclass initializer, and set any model-specific parameters. At minimum, set self.v_rest (resting membrane potential) and self.myelinated (whether the fiber is myelinated). It also is recommended to specify gating variables if you want to be able to record these values during simulations. Finally, set the delta_z parameter, which is the distance from the center of one node to the next node.

3. **Define the Node Creation Method(s)**: Implement method(s) that create the specific sections of the fiber model. For a homogeneous fiber model, you will create a single method. For a heterogeneous fiber model, you will create multiple methods. These methods should return a NEURON section object representing the node or section.

4. **Define the `generate` Method**: Implement the `generate` method, which calls the superclass `generate` method with a list of functions that create the specific sections of the fiber model. For a homogeneous fiber model, you will pass a single function in the list. For a heterogeneous fiber model, you will pass a list of functions.

```{note}
You may create a fiber model in the /models directory, or in a separate folder (e.g., `/plugins`) and make it discoverable as a plugin (recommended). To make your fiber model discoverable as a plugin, follow [these instructions](#making-your-fiber-model-discoverable-as-a-plugin).
```

### Example

#### Creating a New Homogeneous Fiber Model

If your model consists of identical sections throughout (typically unmyelinated fibers), you will pass a single function in the list to the `generate` method. For these models it is recommended to use `{py:method} pyfibers.fiber.Fiber.nodebuilder` to create the nodes, then add/override any mechanisms or properties as needed.

```python
from neuron import h
from pyfibers import FiberModel
from pyfibers.fiber import Fiber, nodebuilder

h.load_file("stdrun.hoc")


class MyHomogeneousFiber(Fiber):
    submodels = ["MY_HOMOGENEOUS_FIBER_MODEL"]

    def __init__(
        self, fiber_model: FiberModel, diameter: float, delta_z: float = 8.333, **kwargs
    ):
        """Initialize MyHomogeneousFiber class.

        :param diameter: Fiber diameter [microns]
        :param delta_z: Length of each segment [microns]
        :param kwargs: Keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, diameter=diameter, **kwargs)
        self.gating_variables = {
            "h": "h_myModel",
            "m": "m_myModel",
            "n": "n_myModel",
        }
        self.myelinated = False
        self.v_rest = -65
        self.delta_z = delta_z

    def generate(self, **kwargs) -> Fiber:
        """Generate the fiber model sections with NEURON.

        :param kwargs: Arguments to pass to the base class generate method
        :return: Fiber object
        """
        return super().generate([self.create_my_node], **kwargs)

    def create_my_node(self, ind: int, node_type: str) -> h.Section:
        """Create a node for MyHomogeneousFiber model.

        :param ind: Index of the node
        :param node_type: Type of the node
        :return: NEURON section representing the node
        """
        node = self.nodebuilder(ind, node_type)
        node.insert("myModel")
        node.Ra = 100
        node.cm = 1.0
        node.ena = 50
        node.ek = -77
        return node
```

#### Creating a New Heterogeneous Fiber Model

For fibers with multiple section types (e.g., nodes, myelin), you will pass a list of functions to the `generate` method. You MUST calculate delta_z (the distance from the center of one node to the next node) here in order to allow PyFibers to verify that the fiber is correctly built. You may still use `{py:method} pyfibers.fiber.Fiber.nodebuilder` to create the nodes, then add/override any mechanisms or properties as needed.

```python
from neuron import h
from pyfibers import FiberModel
from pyfibers.fiber import Fiber, nodebuilder

h.load_file("stdrun.hoc")


class MyHeterogeneousFiber(Fiber):
    submodels = ["MY_HETEROGENEOUS_FIBER_MODEL"]

    def __init__(
        self, fiber_model: FiberModel, diameter: float, delta_z: float = 8.333, **kwargs
    ):
        """Initialize MyHeterogeneousFiber class.

        :param diameter: Fiber diameter [microns]
        :param delta_z: Length of each segment [microns]
        :param kwargs: Keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, diameter=diameter, **kwargs)
        self.gating_variables = {
            "h": "h_myModel",
            "m": "m_myModel",
            "s": "s_myModel",
        }
        self.myelinated = True
        self.v_rest = -80
        self.delta_z = 1.5 + self.diameter * 100

    def generate(self, **kwargs) -> Fiber:
        """Generate the fiber model sections with NEURON.

        :param kwargs: Arguments to pass to the base class generate method
        :return: Fiber object
        """
        function_list = [
            self.create_node,
            self.create_myelin,
        ]

        return super().generate(function_list, **kwargs)

    def create_node(self, ind: int, node_type: str) -> h.Section:
        """Create a node for MyHeterogeneousFiber model.

        :param ind: Index of the node
        :param node_type: Type of the node
        :return: NEURON section representing the node
        """
        name = f"{node_type} node {index}"
        node = h.Section(name=name)
        node.diam = self.diameter * 0.7
        node.L = 1.5
        node.insert("extracellular")
        node.insert("myModel")
        node.cm = 2.5
        node.Ra = 54.7
        return node

    def create_myelin(self, ind: int, node_type: str) -> h.Section:
        """Create a myelin section for MyHeterogeneousFiber model.

        :param ind: Index of the section
        :param node_type: Type of the section
        :return: NEURON section representing the myelin section
        """
        name = f"myelin {index}"
        section = h.Section(name=name)
        section = self.nodebuilder(ind, node_type)
        section.cm = 0
        section.Ra = 54.7
        return section
```

By following these steps, you can create new fiber models based on the specific characteristics of your desired fiber. Define the necessary methods and parameters that match your fiber model requirements.

### Making your fiber model discoverable as a plugin
Other research groups may wish to create their own fiber models in the PyFibers environment, and publish them as a separate repository. Such fiber models can be made discoverable as plugins which will become automatically available in PyFibers after installation. To make your fiber model discoverable as a plugin, follow these steps:

1. **Create a Python package for your fiber model**: Create a Python package for your fiber model and include the necessary files and classes. We recommend following Python's [packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

2. Specify the entry point for your fiber model. The entry point tells pyfibers that your fiber model should be imported, and where to find the class that defines it. Entry points can be specified in multiple ways, here we will provide an example using pyproject.toml:

```toml
[project.entry-points."pyfibers.fiber_plugins"]
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

2. **Set up and configure intracellular stimulation**:
    ```python
    from neuron import h

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

### Example 2: Custom Simulation Using `Stimulation.pre_run_setup` and Manual Extracellular Potentials

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

### Example 3: Custom Simulation with a Custom `run_sim` Function

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

### Example 4: Custom `run_sim` in a subclass of `Stimulation`

In this example, we demonstrate how to create a subclass of `Stimulation` and override the `run_sim` method to provide custom simulation logic.

#### Steps:

1. **Create a subclass of `Stimulation`**:
    ```python
    from pyfibers import Stimulation


    class CustomStimulation(Stimulation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    ```

2. **Override the `run_sim` method**:
    ```python
    def run_sim(self, fiber):
        print("Running custom simulation:")

        self.pre_run_setup(fiber)

        # Example of a custom simulation loop
        for i in range(int(self.tstop / self.dt)):
            # Custom simulation logic here

            # Advance to next time step
            h.fadvance()

        return 1  # return desired data here.
    ```
3. **Set up the `CustomStimulation` instance and run the simulation**:
    ```python
    stimulation = CustomStimulation(dt=0.001, tstop=20)
    stimulation.run_sim(fiber)
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
