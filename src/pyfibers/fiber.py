"""The copyrights of this software are owned by Duke University."""

from __future__ import annotations

import math
import typing
import warnings
from typing import TYPE_CHECKING, Callable

import numpy as np
from neuron import h
from numpy import ndarray

h.load_file('stdrun.hoc')

if TYPE_CHECKING:
    from .model_enum import FiberModel


def build_fiber(
    fiber_model: FiberModel, diameter: float, n_sections: int = None, length: float = None, **kwargs
) -> object:  # Replace with the correct type of your fiber models
    """Generate a fiber model in NEURON.

    :param fiber_model: fiber model to use
    :param diameter: fiber diameter [um]
    :param n_sections: number of fiber coordinates to use
    :param length: length of the fiber
    :param kwargs: keyword arguments to pass to the fiber model class
    :return: generated instance of fiber model class
    """
    assert (length is not None) or (n_sections is not None), "Must specify either length or n_sections"
    assert (length is None) or (n_sections is None), "Can't specify both length and n_sections"

    fiber_class = fiber_model.value

    fiber_instance = fiber_class(diameter=diameter, fiber_model=fiber_model, **kwargs)
    fiber_instance.generate(n_sections, length)
    fiber_instance.potentials = np.zeros(len(fiber_instance.coordinates))
    assert len(fiber_instance) == fiber_instance.nodecount, "Node count does not match number of nodes"
    return fiber_instance


class Fiber:
    """Base class for model fibers."""

    def __init__(  # TODO update tests
        self: Fiber,
        fiber_model: FiberModel,
        diameter: float,
        temperature: float = 37,
        passive_end_nodes: int | bool = True,
    ) -> None:
        """Initialize Fiber class.

        :param diameter: fiber diameter [um]
        :param fiber_model: name of fiber model type
        :param temperature: temperature of model [degrees celsius]
        :param passive_end_nodes: if True, set passive properties at the end nodes
        """
        # fiber arguments
        self.diameter = diameter
        self.fiber_model = fiber_model
        self.temperature = temperature
        self.passive_end_nodes = (
            passive_end_nodes  # TODO error if longer than half the fiber length, warning if more than a quarter
        )

        # recording
        self.gating: dict[str, h.Vector] = None
        self.gating_variables: dict[str, str] = {}
        self.vm: list = None
        self.apc: list = None
        self.im: list = None

        # intrinsic activity
        self.nc: h.NetCon = None
        self.syn: h.ExpSyn = None
        self.stim: h.NetStim = None

        # fiber attributes
        self.nodecount: int = None
        self.delta_z: float = None
        self.sections: list = []
        self.nodes: list = []
        self.coordinates: ndarray = np.array([])
        self.length: float = None
        self.potentials: ndarray = np.array([])

    def __call__(self: Fiber, loc: float) -> h.Section:
        """Call the fiber nodes with NEURON style indexing.

        :param loc: location in the fiber (from 0 to 1)
        :return: node at the given location
        """  # noqa: DAR201
        assert 0 <= loc <= 1, "Location must be between 0 and 1"
        return self.nodes[self.loc_index(loc)]

    def __str__(self: Fiber) -> str:
        """Return a string representation of the fiber."""  # noqa: DAR201
        return (
            f"{self.fiber_model.name} fiber of diameter {self.diameter} μm and length {self.length:.2f} μm "
            f"\n\tnode count: {len(self)}, section count: {len(self.sections)}"
        )

    def __repr__(self: Fiber) -> str:
        """Return a string representation of the fiber."""  # noqa: DAR201
        # TODO: make this more informative for developers
        return self.__str__()

    def __len__(self: Fiber) -> int:
        """Return the number of nodes in the fiber."""  # noqa: DAR201
        assert self.nodecount == len(self.nodes), "Node count does not match number of nodes"
        return len(self.nodes)

    def __getitem__(self: Fiber, item: int) -> h.Section:
        """Return the node at the given index."""  # noqa: DAR201, DAR101
        return self.nodes[item]

    def __iter__(self: Fiber) -> typing.Iterator[h.Section]:
        """Return an iterator over the nodes in the fiber."""  # noqa: DAR201
        return iter(self.nodes)

    def __contains__(self: Fiber, item: h.Section) -> bool:
        """Return True if the section is in the fiber."""  # noqa: DAR201, DAR101
        return item in self.sections

    def _connect_sections(self: Fiber) -> None:
        """Connect the fiber sections."""
        for i in range(len(self.sections) - 1):
            self.sections[i + 1].connect(self.sections[i])

    def _calculate_coordinates(self: Fiber) -> None:
        """Generate and validate fiber coordinates."""
        start_coords = np.array([0] + [section.L for section in self.sections[:-1]])  # start of each section
        end_coords = np.array([section.L for section in self.sections])  # end of each section
        self.coordinates: np.ndarray = np.cumsum(  # type: ignore
            (start_coords + end_coords) / 2
        )  # center of each section
        self.length = np.sum([section.L for section in self.sections])  # actual length of fiber
        assert np.isclose(
            self.coordinates[-1] - self.coordinates[0],  # center to center length
            self.delta_z * (self.nodecount - 1),  # expected length of fiber
        ), "Fiber length is not correct."

    def loc(self: Fiber, loc: float) -> h.Section:
        """Return the node at the given location (Using the same convention as NEURON).

        :param loc: location in the fiber (from 0 to 1)
        :return: node at the given location
        """
        return self(loc)

    def loc_index(self: Fiber, loc: float) -> int:
        """Return the index of the node at the given location (Using the same convention as NEURON).

        :param loc: location in the fiber (from 0 to 1)
        :return: index of the node at the given location
        """
        return int(loc * (len(self) - 1))

    def resample_potentials(
        self: Fiber,
        potentials: np.ndarray,
        potential_coords: np.ndarray,
        center: bool = False,
        inplace: bool = False,
    ) -> np.ndarray:
        """Use linear interpolation to resample the high-res potentials to the proper fiber coordinates.

        :param potentials: high-res potentials
        :param potential_coords: coordinates of high-res potentials
        :param center: if True, center the potentials around the fiber midpoint
        :param inplace: if True, replace the potentials in the fiber with the resampled potentials
        :return: resampled potentials
        """
        potential_coords, potentials = np.array(potential_coords), np.array(potentials)

        if not center:
            potential_coords = potential_coords - potential_coords[0]
            target_coords = self.coordinates
        else:
            target_coords = self.coordinates - (self.coordinates[0] + self.coordinates[-1]) / 2
            potential_coords = potential_coords - (potential_coords[0] + potential_coords[-1]) / 2

        newpotentials = np.interp(target_coords, potential_coords, potentials)

        assert (np.amax(potential_coords) >= np.amax(target_coords)) and (
            np.amin(potential_coords) <= np.amin(target_coords)
        ), "Potential coordinates must span the fiber"

        if inplace:
            self.potentials = newpotentials
            assert len(self.potentials) == len(self.coordinates), "Potentials and coordinates must be the same length"

        return newpotentials

    def apcounts(self: Fiber, thresh: float = -30) -> None:
        """Create a list of NEURON APCount objects at all nodes along the axon.

        :param thresh: the threshold value for Vm to pass for an AP to be detected [mV]
        """
        self.apc = [h.APCount(node(0.5)) for node in self]
        for apc in self.apc:
            apc.thresh = thresh

    def record_values(self: Fiber, ref_attr: str) -> list[h.Vector]:
        """Record values from all nodes along the axon.

        :param ref_attr: attribute to record from each node
        :return: list of NEURON vectors for recording values during simulation
        """
        if self.passive_end_nodes:
            return (
                [None] * self.passive_end_nodes
                + [
                    h.Vector().record(getattr(node(0.5), ref_attr))
                    for node in self.nodes[self.passive_end_nodes : -self.passive_end_nodes]
                ]
                + [None] * self.passive_end_nodes
            )
        return [h.Vector().record(getattr(node(0.5), ref_attr)) for node in self.nodes]

    def set_save_vm(self: Fiber) -> None:
        """Record membrane voltage (mV) along the axon."""
        self.vm = self.record_values('_ref_v')

    def set_save_im(self: Fiber) -> None:
        """Record membrane current (nA) along the axon."""
        self.im = self.record_values('_ref_i_membrane')

    def set_save_gating(self: Fiber) -> None:
        """Record gating parameters for axon nodes."""
        assert self.gating_variables, "Gating variables not defined for this fiber type"

        self.gating = {}
        for name, var in self.gating_variables.items():
            self.gating[name] = self.record_values(f"_ref_{var}")

    def point_source_potentials(
        self: Fiber,
        x: float,
        y: float,
        z: float,
        i0: float,
        sigma: float | tuple,
        inplace: bool = False,
    ) -> np.ndarray:
        """Calculate extracellular potentials at all fiber coordinates due to a point source.

        :param x: x-coordinate of point source [um]
        :param y: y-coordinate of point source [um]
        :param z: z-coordinate of point source [um]
        :param i0: current of point source [mA]
        :param sigma: conductivity of extracellular medium [S/m].
            Float for isotropic, tuple of length 3 (x,y,z) for anisotropic
        :param inplace: whether to update self.potentials in-place, defaults to False
        :return: potentials at all fiber coordinates [mV]
        """
        # Calculate distance between source and sections
        xs = ys = np.zeros(len(self.coordinates))
        xs = x - xs
        ys = y - ys
        zs = z - self.coordinates
        # convert to meters
        xs *= 1e-6
        ys *= 1e-6
        zs *= 1e-6

        if isinstance(sigma, (float, int)):
            # Isotropic case
            potentials = i0 / (4 * np.pi * sigma * np.sqrt(xs**2 + ys**2 + zs**2))
        else:  # TODO double check that this is correct
            # Anisotropic case
            sigma_x, sigma_y, sigma_z = sigma
            potentials = i0 / (
                4 * np.pi * np.sqrt(sigma_y * sigma_z * xs**2 + sigma_x * sigma_z * ys**2 + sigma_x * sigma_y * zs**2)
            )

        if inplace:
            self.potentials = potentials

        return potentials

    def measure_cv_raw(self: Fiber, start: float = 0.25, end: float = 0.75, tolerance: float = 0.005) -> float:
        """Estimate fiber conduction velocity using ap times at specific points.

        :param start: Starting position for conduction velocity measurement [0, 1]
        :param end: Ending position for conduction velocity measurement [0, 1]
        :param tolerance: Default tolerance for determining linearity (ms)
        :raises ValueError: if conduction is not (roughly) linear between start and end
        :return: Conduction velocity [m/s]
        """
        # Check that both nodes (start and end) have detected APs
        start_ind, end_ind = self.loc_index(start), self.loc_index(end)
        for ind in [start_ind, end_ind]:
            assert self.apc[ind].n > 0, f"No detected APs at node {ind}."

        # Check that conduction is linear between start and end nodelist
        aptimes = [self.apc[ind].time for ind in range(start_ind, end_ind + 1)]
        if not np.allclose(np.diff(aptimes), np.diff(aptimes)[0], atol=tolerance):
            raise ValueError("Conduction is not linear between the specified nodes.")

        # Calculate conduction velocity from AP times
        coords = [self.coordinates[i] for i, section in enumerate(self.sections) if section in self.nodes]
        distance = np.abs(coords[start_ind] - coords[end_ind])
        time = np.abs(aptimes[-1] - aptimes[0])
        distance *= 1e-6  # convert to meters
        time *= 1e-3  # convert to seconds

        return distance / time

    def add_intrinsic_activity(
        self: Fiber,
        loc: float = None,
        loc_index: int = None,
        avg_interval: float = 1,
        num_stims: int = 1,
        start_time: float = 1,
        noise: float = 0,
        synapse_tau: float = 0.1,
        synapse_reversal_potential: float = 0,
        netcon_weight: float = 0.1,
    ) -> None:
        """Add intrinsic activity to a fiber.

        Intrinsic activity is generated by:
            - a spike generator (NetStim)
            - a synapse (ExpSyn) on a fiber node
            - a network connection (NetCon) between the spike generator and the synapse

        When the synapse receives a spike, it inject an exponentially decaying current.
        The strength of the current injection is affected by the weight of the connection (netcon_weight).
        Since the time constant, reversal potential, and weight all affect the current injection,
        the default values may not always be sufficient.
        See NEURON documentation for more details.

        :param loc: location in the fiber (from 0 to 1). Mutually exclusive with loc_index.
        :param loc_index: index of the node at the given location. Mutually exclusive with loc.
        :param avg_interval: average interval between spikes [ms]
        :param num_stims: number of spikes to deliver
        :param start_time: time to start delivering spikes [ms]
        :param noise: noise distribution of spike intervals
            for regular intervals (no randomness), set to 0
            for poisson process, set to 1
            for semi-random, set between 0 and 1
        :param synapse_tau: time constant for decay of the synapse current [ms]
        :param synapse_reversal_potential: reversal potential of the synapse current [mV]
        :param netcon_weight: weight of connection between spike generator and node stimulus [uS]
        """
        # Get the node at the specified location
        assert (loc is None) != (loc_index is None), "Must specify either loc or loc_index"
        node = self[loc_index] if loc_index is not None else self(loc)

        # create the spike generator
        self.stim = h.NetStim()
        self.stim.interval = avg_interval
        self.stim.number = num_stims
        self.stim.start = start_time
        self.stim.noise = noise

        # create the synapse on the node
        self.syn = h.ExpSyn(node(0.5))
        self.syn.tau = synapse_tau
        self.syn.e = synapse_reversal_potential
        # Record the current injected by the synapse
        self.syn_current = h.Vector().record(self.syn._ref_i)

        # Connect the spike generator to the synapse
        self.nc = h.NetCon(self.stim, self.syn)
        self.nc.weight[0] = netcon_weight


class _HomogeneousFiber(Fiber):
    """Initialize Homogeneous (all sections are identical) class."""

    def __init__(
        self: _HomogeneousFiber, fiber_model: FiberModel, diameter: float, delta_z: float = 8.333, **kwargs
    ) -> None:
        """Initialize _HomogeneousFiber class.

        :param fiber_model: name of fiber model type
        :param diameter: fiber diameter [microns]
        :param delta_z: length of each node/section
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, diameter=diameter, **kwargs)
        self.delta_z: float = delta_z
        self.v_rest: int = None

    def generate_homogeneous(
        self: _HomogeneousFiber, n_sections: int, length: float, modelfunc: Callable, *args, **kwargs
    ) -> Fiber:
        """Build fiber model sections with NEURON.

        :param n_sections: number of fiber coordinates from COMSOL
        :param length: length of fiber [um] (mutually exclusive with n_sections)
        :param modelfunc: function to generate fiber model (mechanisms and attributes)
        :param args: arguments to pass to modelfunc
        :param kwargs: keyword arguments to pass to modelfunc
        :return: Fiber object
        """
        if self.diameter > 2:
            warnings.warn(
                f"C fibers are typically <=2 um in diameter, received D={self.diameter}. Proceed with caution.",
                stacklevel=2,
            )

        # Determine number of nodecount
        self.nodecount = int(n_sections) if length is None else math.floor(length / self.delta_z)

        # Create fiber sections
        self.sectionbuilder(modelfunc, *args, **kwargs)

        # use section attribute L to generate coordinates, every section is the same length
        self._calculate_coordinates()

        return self

    def nodebuilder(self: _HomogeneousFiber, nodefunc: Callable) -> Callable:
        """Generate a node and apply the specific model described by nodefunc.

        :param nodefunc: function to build node
        :return: nrn.h.Section
        """

        def wrapper(*args, name: str = 'node', **kwargs) -> None:
            node = h.Section(name=name)
            self.sections.append(node)

            node.diam = self.diameter
            node.nseg = 1
            node.L = self.delta_z
            node.insert('extracellular')
            node.xc[0] = 0  # short circuit
            node.xg[0] = 1e10  # short circuit
            node.v = self.v_rest

            return nodefunc(node, *args, **kwargs)

        return wrapper

    @staticmethod
    def passive_node(node: h.Section, v_rest: int) -> None:
        """Set passive properties for a node.

        :param node: section object to set passive properties for
        :param v_rest: resting membrane potential [mV]
        """
        node.insert('pas')
        node.g_pas = 0.0001
        node.e_pas = v_rest
        node.Ra = 1e10

    def sectionbuilder(self: _HomogeneousFiber, modelnodefunc: Callable, *args, **kwargs) -> None:
        """Create and connect NEURON sections for an unmyelinated fiber.

        :param modelnodefunc: function to build node mechanisms and attributes
        :param args: arguments to pass to modelnodefunc
        :param kwargs: keyword arguments to pass to modelnodefunc
        """
        self.sections = []
        for i in range(self.nodecount):
            if self.passive_end_nodes and (i < self.passive_end_nodes or i >= self.nodecount - self.passive_end_nodes):
                name = f"passive node {i}"
                self.nodebuilder(self.passive_node)(self.v_rest, name=name)
            else:
                name = f"active node {i}"
                self.nodebuilder(modelnodefunc)(*args, name=name, **kwargs)
        self._connect_sections()
        self.nodes = self.sections


class HeterogeneousFiber(Fiber):
    """Superclass for different types of fiber models."""

    def __init__(self: HeterogeneousFiber, fiber_model: FiberModel, diameter: float, **kwargs) -> None:
        """Initialize HeterogeneousFiber class.

        :param fiber_model: fiber model to use
        :param diameter: fiber diameter [um]
        :param kwargs: keyword arguments to pass to the fiber model class
        """
        super().__init__(fiber_model=fiber_model, diameter=diameter, **kwargs)
        self.delta_z = None
        self.v_rest = None

    def generate(
        self: HeterogeneousFiber, n_sections: int, length: float, function_list: list[Callable]
    ) -> HeterogeneousFiber:
        """Build fiber model sections with NEURON.

        :param n_sections: number of sections to comprise fiber
        :param length: length of the fiber
        :param function_list: list of functions to generate fiber sections. Each function should generate a section.
            the function list should start with the node of Ranvier and end with the section before the next node
        :return: generated instance of fiber model class
        """
        # Determine geometrical parameters for fiber based on fiber model
        if length is not None:
            n_sections = math.floor(length / self.delta_z) * len(function_list) + 1
        else:
            assert (n_sections - 1) % len(function_list) == 0, (
                f"n_sections must be 1 + {len(function_list)}n where n is an integer one less than the "
                "number of nodes of Ranvier."
            )

        # Determine number of nodecount
        self.nodecount = int(1 + (n_sections - 1) / len(function_list))

        # Create fiber sections
        self.create_sections(function_list)

        # Generate fiber coordinates and validate them
        self._calculate_coordinates()

        return self

    def create_sections(self: HeterogeneousFiber, function_list: list[Callable]) -> HeterogeneousFiber:
        """Create and connect NEURON sections for a myelinated fiber type using specified functions.

        :param function_list: list of functions to generate fiber sections. Each function should generate a section.
            the function list should start with the node of Ranvier and end with the section before the next node
        :return: generated instance of fiber model class
        """
        nsegments = (self.nodecount - 1) * len(function_list) + 1
        for ind in range(1, nsegments + 1):
            function = function_list[(ind - 1) % len(function_list)]
            section = function(ind)
            if (ind - 1) % len(function_list) == 0:
                self.nodes.append(section)
            self.sections.append(section)

        self._connect_sections()

        return self
