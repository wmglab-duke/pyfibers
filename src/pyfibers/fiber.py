"""The copyrights of this software are owned by Duke University."""

from __future__ import annotations

import math
import typing
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from nd_line.nd_line import nd_line
from neuron import h
from numpy import ndarray

h.load_file('stdrun.hoc')

if TYPE_CHECKING:
    from .model_enum import FiberModel


def build_fiber(
    fiber_model: FiberModel,
    diameter: float,
    length: float = None,
    n_sections: int = None,
    n_nodes: int = None,
    **kwargs,
) -> Fiber:
    """Generate a fiber model in NEURON.

    :param fiber_model: fiber model to use
    :param diameter: fiber diameter [um]
    :param n_sections: number of fiber longitudinal coordinates to use
    :param length: length of the fiber
    :param n_nodes: number of nodes in the fiber
    :param kwargs: keyword arguments to pass to the fiber model class
    :raises ValueError: if more than one of length, n_sections, or n_nodes is provided
    :return: generated instance of fiber model class
    """
    # must provide one of length, n_sections, or n_nodes, and only one
    if sum(x is not None for x in [length, n_sections, n_nodes]) != 1:
        raise ValueError("Must provide exactly one of length, n_sections, or n_nodes")

    assert "is_3d" not in kwargs, "is_3d is set automatically, try using build_fiber_3d() instead"

    fiber_class = fiber_model.value

    fiber_instance = fiber_class(diameter=diameter, fiber_model=fiber_model, **kwargs)

    fiber_instance.generate(n_nodes=n_nodes, n_sections=n_sections, length=length)

    fiber_instance.coordinates = np.concatenate(
        (
            np.zeros((len(fiber_instance.longitudinal_coordinates), 1)),
            np.zeros((len(fiber_instance.longitudinal_coordinates), 1)),
            fiber_instance.longitudinal_coordinates.reshape(-1, 1),
        ),
        axis=1,
    )
    fiber_instance.potentials = np.zeros(len(fiber_instance.longitudinal_coordinates))
    fiber_instance.time = h.Vector().record(h._ref_t)  # TODO move to generate

    assert len(fiber_instance) == fiber_instance.nodecount, "Node count does not match number of nodes"

    return fiber_instance


def build_fiber_3d(
    fiber_model: FiberModel,
    diameter: float,
    path_coordinates: ndarray,
    **kwargs,
) -> Fiber:
    """Generate a 3D fiber model in NEURON.

    :param fiber_model: fiber model to use
    :param diameter: fiber diameter [um]
    :param path_coordinates: x,y,z-coordinates of the fiber [um] (required)
    :param kwargs: keyword arguments to pass to the fiber model class
    :raises ValueError: if n_sections, n_nodes, or length is provided
    :return: generated instance of fiber model class
    """
    # path_coordinates must be provided
    if path_coordinates is None:
        raise ValueError("path_coordinates must be provided for 3D fibers")

    if 'n_sections' in kwargs or 'n_nodes' in kwargs or 'length' in kwargs:
        raise ValueError("For 3D fibers, cannot specify n_sections, n_nodes, or length")

    # Calculate the length from path_coordinates
    nd = nd_line(path_coordinates)

    # Create the fiber instance using the base class method
    fiber_instance = build_fiber(
        fiber_model=fiber_model,
        diameter=diameter,
        length=nd.length,
        **kwargs,
    )
    fiber_instance._set_3d()

    # Make the 3D fiber coordinates an intrinsic property of the Fiber object.
    fiber_instance.coordinates = np.array([nd.interp(p) for p in fiber_instance.longitudinal_coordinates])
    # TODO: support shifting fiber coordinates along fiber path

    return fiber_instance


class Fiber:
    """Base class for model fibers."""

    def __init__(  # TODO update tests
        self: Fiber,
        fiber_model: FiberModel,
        diameter: float,
        temperature: float = 37,
        passive_end_nodes: int | bool = True,
        is_3d: bool = False,
    ) -> None:
        """Initialize Fiber class.

        :param diameter: fiber diameter [um]
        :param fiber_model: name of fiber model type
        :param temperature: temperature of model [degrees celsius]
        :param passive_end_nodes: if True, set passive properties at the end nodes
        :param is_3d: if True, fiber is 3D
        """
        # fiber arguments
        self.diameter = diameter
        self.fiber_model = fiber_model
        self.temperature = temperature
        self.passive_end_nodes = (
            passive_end_nodes  # TODO error if longer than half the fiber length, warning if more than a quarter
        )
        self.__is_3d = is_3d

        # recording
        self.gating: dict[str, h.Vector] = None
        self.gating_variables: dict[str, str] = {}
        self.vm: list = None
        self.apc: list = None
        self.im: list = None
        self.vext: list = None
        self.time: list = None

        # intrinsic activity
        self.nc: h.NetCon = None
        self.syn: h.ExpSyn = None
        self.stim: h.NetStim = None

        # fiber attributes
        self.myelinated: bool = None
        self.nodecount: int = None
        self.delta_z: float = None
        self.sections: list = []
        self.nodes: list = []
        self.coordinates: ndarray = np.array([])
        self.length: float = None
        self.potentials: ndarray = np.array([])

    def __call__(self: Fiber, loc: float, target: str = 'nodes') -> h.Section:
        """Call the fiber nodes/sections with NEURON style indexing.

        Note: Does not consider the length of the sections.
        Instead, this method returns the section at the loc position in its respective list.
        For nodes, this is identical to that location along the fiber length.

        :param loc: location in the fiber (from 0 to 1)
        :param target: can be nodes or sections
        :return: node/section at the given location
        """
        assert 0 <= loc <= 1, "Location must be between 0 and 1"
        if target == 'sections':
            return self.sections[self.loc_index(loc, target=target)]
        assert target == 'nodes', 'target can either be "nodes" or "sections"'
        return self.nodes[self.loc_index(loc, target=target)]

    loc = __call__  # alias to maintain old functionality

    def __str__(self: Fiber) -> str:
        """Return a string representation of the fiber."""  # noqa: DAR201
        return (
            f"{self.fiber_model.name} fiber of diameter {self.diameter} μm and length {self.length:.2f} μm "
            f"\n\tnode count: {len(self)}, section count: {len(self.sections)}. "
            f"\n\tFiber is {'' if self.is_3d() else 'not '}3d."
        )

    def __repr__(self: Fiber) -> str:
        """Return a detailed string representation of the fiber."""  # noqa: DAR201
        return (
            f"{self.__class__.__name__}("
            f"fiber_model={self.fiber_model}, "
            f"diameter={self.diameter}, "
            f"temperature={self.temperature}, "
            f"passive_end_nodes={self.passive_end_nodes}, "
            f"is_3d={self.__is_3d}, "
            f"gating_variables={self.gating_variables}, "
            f"nc={self.nc}, "
            f"syn={self.syn}, "
            f"stim={self.stim}, "
            f"myelinated={self.myelinated}, "
            f"nodecount={self.nodecount}, "
            f"delta_z={self.delta_z}, "
            f"sections={self.sections}, "
            f"nodes={self.nodes}, "
            f"length={self.length}, "
            f"coordinates={self.coordinates.tolist()}, "
            f"potentials={self.potentials.tolist()}), "
            f"longitudinal_coordinates={self.longitudinal_coordinates.tolist()}"
        )

    def __len__(self: Fiber, target: str = 'nodes') -> int:
        """Return the number of nodes/sections in the fiber."""  # noqa: DAR201, DAR101
        assert self.nodecount == len(self.nodes), "Node count does not match number of nodes"
        if target == 'sections':
            return len(self.sections)
        assert target == 'nodes', 'target can either be "nodes" or "sections"'
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
        self.longitudinal_coordinates: np.ndarray = np.cumsum(  # type: ignore
            (start_coords + end_coords) / 2
        )  # center of each section
        self.length = np.sum([section.L for section in self.sections])  # actual length of fiber
        assert np.isclose(
            self.longitudinal_coordinates[-1] - self.longitudinal_coordinates[0],  # center to center length
            self.delta_z * (self.nodecount - 1),  # expected length of fiber
        ), "Fiber length is not correct."

    def loc_index(self: Fiber, loc: float, target: str = 'nodes') -> int:
        """Return the index of the node at the given location (Using the same convention as NEURON).

        Note: Does not consider the length of the sections.
        Instead, this method index of the section at the loc position in its respective list.
        For nodes, this is identical to that location along the fiber length.

        :param loc: location in the fiber (from 0 to 1)
        :param target: can be nodes or sections
        :return: index of the node/section at the given location
        """
        assert 0 <= loc <= 1, "Location must be between 0 and 1"
        if target == 'sections':
            return int(loc * (len(self.sections) - 1))
        assert target == 'nodes', 'target can either be "nodes" or "sections"'
        return int(loc * (len(self) - 1))

    def is_3d(self: Fiber) -> bool:  # noqa: D102
        return self.__is_3d

    def _set_3d(self: Fiber) -> None:  # noqa: D102
        self.__is_3d = True

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

        assert len(potential_coords.shape) == 1, "Potential coordinates must be a 1D array"
        assert len(potentials.shape) == 1, "Potentials must be a 1D array"
        assert len(potential_coords) == len(potentials), "Potentials and coordinates must be the same length"
        assert len(potential_coords) >= 2, "Must provide at least two points for resampling"
        assert np.all(np.diff(potential_coords) > 0), "Potential coordinates must be monotonically increasing"

        if not center:
            potential_coords = potential_coords - potential_coords[0]
            target_coords = self.longitudinal_coordinates
        else:
            target_coords = (
                self.longitudinal_coordinates
                - (self.longitudinal_coordinates[0] + self.longitudinal_coordinates[-1]) / 2
            )
            potential_coords = potential_coords - (potential_coords[0] + potential_coords[-1]) / 2

        newpotentials = np.interp(target_coords, potential_coords, potentials)

        assert (np.amax(potential_coords) >= np.amax(target_coords)) and (
            np.amin(potential_coords) <= np.amin(target_coords)
        ), "Potential coordinates must span the fiber"

        if inplace:
            self.potentials = newpotentials
            assert len(self.potentials) == len(
                self.longitudinal_coordinates
            ), "Potentials and coordinates must be the same length"

        return newpotentials

    def apcounts(self: Fiber, thresh: float = -30) -> None:
        """Create a list of NEURON APCount objects at all nodes along the axon.

        :param thresh: the threshold value for Vm to pass for an AP to be detected [mV]
        """
        self.apc = [h.APCount(node(0.5)) for node in self]
        for apc in self.apc:
            apc.thresh = thresh

    def record_values(self: Fiber, ref_attr: str, allsec: bool = False) -> list[h.Vector]:
        """Record values from all nodes along the axon.

        :param ref_attr: attribute to record from each node
        :param allsec: if True, record from all sections, not just nodes
        :return: list of NEURON vectors for recording values during simulation
        """
        if allsec:
            return [h.Vector().record(getattr(sec(0.5), ref_attr)) for sec in self.sections]
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

    def set_save_vm(self: Fiber, allsec: bool = False) -> None:
        """Record membrane voltage (mV) along the axon."""  # noqa: DAR101
        self.vm = self.record_values('_ref_v', allsec=allsec)

    def set_save_im(self: Fiber, allsec: bool = False) -> None:
        """Record membrane current (nA) along the axon."""  # noqa: DAR101
        self.im = self.record_values('_ref_i_membrane', allsec=allsec)

    def set_save_vext(self: Fiber) -> None:
        """Record extracellular potential (mV) along the axon."""
        self.vext = [h.Vector().record(sec(0.5)._ref_vext[0]) for sec in self.sections]

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
        xs = x - self.coordinates[:, 0]
        ys = y - self.coordinates[:, 1]
        zs = z - self.coordinates[:, 2]
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

    def measure_cv(self: Fiber, start: float = 0.25, end: float = 0.75, tolerance: float = 0.005) -> float:
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
        coords = [self.longitudinal_coordinates[i] for i, section in enumerate(self.sections) if section in self.nodes]
        distance = np.abs(coords[start_ind] - coords[end_ind])
        time = np.abs(aptimes[-1] - aptimes[0])
        distance *= 1e-6  # convert to meters
        time *= 1e-3  # convert to seconds

        return distance / time

    def add_intrinsic_activity(
        self: Fiber,
        loc: float = 0.1,
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
        self.nc.delay = 0  # no delay

    @staticmethod
    def calculate_periaxonal_current(from_sec: h.Section, to_sec: h.Section, vext_from: float, vext_to: float) -> float:
        """Calculate the periaxonal current between two compartments.

        :param from_sec: the section from which the current is flowing
        :param to_sec: the section to which the current is flowing
        :param vext_from: the periaxonal potential at the from_sec [mV]
        :param vext_to: the periaxonal potential at the to_sec [mV]
        :return: the periaxonal current between the two compartments [mA]
        """
        length_from = 1e-4 * from_sec.L  # [cm]
        xraxial_from = from_sec.xraxial[0]  # [megaOhm/cm]
        length_to = 1e-4 * to_sec.L  # [cm]
        xraxial_to = to_sec.xraxial[0]  # [megaOhm/cm]
        r_periaxonal = 1e6 * (xraxial_to * length_to / 2 + xraxial_from * length_from / 2)  # [Ohm]
        return (vext_from - vext_to) / r_periaxonal  # I = V/R [mA]

    def membrane_currents(self: Fiber, downsample: int = 1) -> np.ndarray:
        """Calculate membrane currents, including periaxonal currents for myelinated fibers.

        :param downsample: the downsample rate for the time vector
        :return: membrane current matrix, consisting of membrane currents for all sections for every time point
        """
        assert (
            self.im is not None
        ), "Membrane currents not saved. Call set_save_im(allsec=True) before running the simulation."
        assert (
            self.vext is not None
        ), "Extracellular potentials not saved. Call set_save_vext() before running the simulation."
        assert len(self.im) == len(
            self.sections
        ), "Membrane currents not saved for all sections, call set_save_im(allsec=True) before running the simulation."

        time_vector = np.array(self.time)
        downsampled_time = time_vector[::downsample]
        downsample_idx = np.arange(0, len(time_vector), downsample)

        # Precompute lengths and diameters for all sections
        sections_length = np.array([1e-4 * sec.L for sec in self.sections])  # [cm]
        sections_diameter = np.array([1e-4 * sec.diam for sec in self.sections])  # [cm]

        i_membrane_matrix = np.zeros((len(downsampled_time), len(self.sections)))

        for time_idx in downsample_idx:
            downsampled_idx = time_idx // downsample
            specific_i_membrane = np.array(
                [self.im[sec_idx][time_idx] for sec_idx in range(len(self.sections))]
            )  # [mA/cm^2]
            i_membrane = np.pi * sections_length * sections_diameter * specific_i_membrane  # [mA]
            v_ext = np.array([self.vext[sec_idx][time_idx] for sec_idx in range(len(self.sections))])  # [mV]

            if self.myelinated:
                # Initialize periaxonal currents
                peri_i_left = np.zeros(len(self.sections))
                peri_i_right = np.zeros(len(self.sections))

                # Calculate periaxonal current from left compartment
                peri_i_left[1:] = np.array(  # TODO perhaps the problem is here? Go back to previous code.
                    [
                        self.calculate_periaxonal_current(
                            self.sections[sec_idx - 1], self.sections[sec_idx], v_ext[sec_idx - 1], v_ext[sec_idx]
                        )
                        for sec_idx in range(1, len(self.sections))
                    ]
                )
                # Calculate periaxonal current from right compartment
                peri_i_right[:-1] = np.array(
                    [
                        self.calculate_periaxonal_current(
                            self.sections[sec_idx + 1], self.sections[sec_idx], v_ext[sec_idx + 1], v_ext[sec_idx]
                        )
                        for sec_idx in range(len(self.sections) - 1)
                    ]
                )

                # Calculate net current
                net_i_extra = i_membrane + peri_i_left + peri_i_right  # [mA]
            else:
                net_i_extra = i_membrane  # [mA]

            # Add current to matrix
            i_membrane_matrix[downsampled_idx] = net_i_extra

        return i_membrane_matrix, downsampled_time

    @staticmethod
    def sfap(current_matrix: np.ndarray, potentials: np.ndarray) -> np.ndarray:
        """Calculate SFAP from a membrane current matrix and recording potentials.

        :param current_matrix: rows are time points, columns are sections
        :param potentials: the extracellular potentials at all fiber coordinates
        :return: the single fiber action potential across all time points
        """
        assert (
            len(potentials) == current_matrix.shape[1]
        ), "Potentials and current matrix columns must have the same length"
        return 1e3 * np.dot(current_matrix, potentials)  # [uV]

    def record_sfap(self: Fiber, rec_potentials: list | ndarray, downsample: int = 1) -> np.ndarray:
        """Potential over time from fiber at a recording electrode.

        :param rec_potentials: the extracellular potentials from recording electrode
        :param downsample: the downsample rate for the time vector
        :return: the single fiber action potential across all time points
        """
        membrane_currents, downsampled_time = self.membrane_currents(downsample)
        return self.sfap(membrane_currents, rec_potentials), downsampled_time

    def set_xyz(self: Fiber, x: float = 0, y: float = 0, z: float = 0) -> None:
        """Set the x and y coordinates of the fiber.

        This assumes that the fiber is straight and centered at the xy origin.
        Do not use with 3D fibers (provided path_coordinates at initialization)

        :param x: x-coordinate of the fiber [um]
        :param y: y-coordinate of the fiber [um]
        :param z: amount to shift the z-coordinate of the fiber [um]
        """
        assert not self.__is_3d, "set_xyz() is not compatible with 3D fibers"
        # warn if the x coordinates are not all the same or the y coordinates are not all the same
        if not np.allclose(self.coordinates[:, 0], x) or not np.allclose(self.coordinates[:, 1], y):
            warnings.warn("X or Y coordinates vary, you may be running this on a fiber with a 3D path", stacklevel=2)
        self.coordinates[:, 0] = x
        self.coordinates[:, 1] = y
        self.coordinates[:, 2] += z

    # 3D Fiber functionality #

    def resample_potentials_3d(
        self: Fiber,
        potentials: np.ndarray,
        potential_coords: np.ndarray,
        center: bool = False,
        inplace: bool = False,
    ) -> np.ndarray:
        """Use linear interpolation to resample the high-res potentials to the proper fiber coordinates.

        :param potentials: high-res potentials
        :param potential_coords: x, y, z coordinates of high-res potentials. Must be along the fiber path
        :param center: if True, center the potentials around the fiber midpoint
        :param inplace: if True, replace the potentials in the fiber with the resampled potentials
        :return: resampled potentials
        """
        assert self.__is_3d, "resample_potentials_3d() is only compatible with 3D fibers"

        potential_coords, potentials = np.array(potential_coords), np.array(potentials)

        # if shape of potential coords is 2D, calculate arc lengths and replace
        assert len(potential_coords.shape) == 2, (
            "Potential coordinates must be a 2D array. " "If using arc lengths, use resample_potentials() instead."
        )
        assert potential_coords.shape[1] == 3, "Must provide exactly 3 coordinates for x,y,z"

        # TODO need some way to ensure that the potential_coords are along the fiber path

        return self.resample_potentials(
            potentials=potentials,
            potential_coords=nd_line(potential_coords).cumul,
            center=center,
            inplace=inplace,
        )

    # Heterogeneous and Homogeneous Fiber Classes #


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
        self: _HomogeneousFiber, modelfunc: Callable, n_nodes: int, n_sections: int, length: float, *args, **kwargs
    ) -> Fiber:
        """Build fiber model sections with NEURON.

        :param n_nodes: number of nodes in the fiber
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
        self.nodecount = int(n_nodes or n_sections or math.floor(length / self.delta_z))

        # Create fiber sections
        self.sectionbuilder(modelfunc, *args, **kwargs)

        # use section attribute L to generate fiber coordinates for NEURON calculations
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
        self: HeterogeneousFiber,
        function_list: list[Callable],
        n_nodes: int = None,
        n_sections: int = None,
        length: float = None,
    ) -> HeterogeneousFiber:
        """Build fiber model sections with NEURON.

        :param n_nodes: number of nodes in the fiber
        :param n_sections: number of sections to comprise fiber
        :param length: length of the fiber
        :param function_list: list of functions to generate fiber sections. Each function should generate a section.
            The function list should start with the node of Ranvier and end with the section before the next node.
            Example: function_list = [node_func, myelin_func]
        :return: generated instance of fiber model class
        """
        # Determine geometrical parameters for fiber based on fiber model
        if n_nodes is not None:
            n_sections = (n_nodes - 1) * len(function_list) + 1
        elif length is not None:
            n_sections = math.floor(length / self.delta_z) * len(function_list) + 1
        else:
            assert n_sections is not None
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
