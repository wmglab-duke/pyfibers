"""The copyrights of this software are owned by Duke University."""

from __future__ import annotations

import math
import typing
import warnings
from enum import Enum, unique
from typing import Callable

import numpy as np
from neuron import h
from numpy import ndarray

h.load_file('stdrun.hoc')


@unique
class FiberModel(Enum):
    """Fiber models."""

    MRG_INTERPOLATION = 0
    MRG_DISCRETE = 1
    SUNDT = 2
    TIGERHOLM = 3
    RATTAY = 4
    SCHILD97 = 5
    SCHILD94 = 6


def build_fiber(
    fiber_model: FiberModel, diameter: float, n_sections: int = None, length: float = None, **kwargs
) -> Fiber:
    """Generate a fiber model in NEURON.

    :param fiber_model: fiber model to use
    :param diameter: fiber diameter [um]
    :param n_sections: number of fiber coordinates to use
    :param length: length of the fiber
    :param kwargs: keyword arguments to pass to the fiber model class
    :raises ValueError: if the fiber model is not supported
    :return: generated instance of fiber model class
    """
    assert (length is not None) or (n_sections is not None), "Must specify either length or n_sections"
    assert (length is None) or (n_sections is None), "Can't specify both length and n_sections"

    # todo: maybe stop passing model, find a cleaner way to implement this factory
    # https://codereview.stackexchange.com/questions/269572/factory-pattern-using-enum
    fiberclass: MRGFiber | RattayFiber | TigerholmFiber | SundtFiber | SchildFiber
    if fiber_model in [FiberModel.MRG_DISCRETE, FiberModel.MRG_INTERPOLATION]:
        from pyfibers.models.mrg import MRGFiber

        fiberclass = MRGFiber(fiber_model, diameter, **kwargs)
    elif fiber_model == FiberModel.RATTAY:
        from pyfibers.models.rattay import RattayFiber

        fiberclass = RattayFiber(fiber_model, diameter, **kwargs)
    elif fiber_model == FiberModel.TIGERHOLM:
        from pyfibers.models.tigerholm import TigerholmFiber

        fiberclass = TigerholmFiber(fiber_model, diameter, **kwargs)
    elif fiber_model == FiberModel.SUNDT:
        from pyfibers.models.sundt import SundtFiber

        fiberclass = SundtFiber(fiber_model, diameter, **kwargs)
    elif fiber_model in [FiberModel.SCHILD94, FiberModel.SCHILD97]:
        from pyfibers.models.schild import SchildFiber

        fiberclass = SchildFiber(fiber_model, diameter, **kwargs)
    else:
        raise ValueError("Fiber Model not valid")

    fiberclass.generate(n_sections, length)

    fiberclass.potentials = np.zeros(len(fiberclass.coordinates))

    assert len(fiberclass) == fiberclass.nodecount, "Node count does not match number of nodes"

    return fiberclass


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
        self.gating: dict[str, h.Vector] = None
        self.gating_variables: dict[str, str] = {}
        self.vm: list = None
        self.apc: list = None
        self.diameter = diameter
        self.fiber_model = fiber_model
        self.temperature = temperature
        self.passive_end_nodes = (
            passive_end_nodes  # TODO error if longer than half the fiber length, warning if more than a quarter
        )
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

    def set_save_vm(self: Fiber) -> None:
        """Record membrane voltage (mV) along the axon."""
        if self.passive_end_nodes:
            self.vm = (
                [None] * self.passive_end_nodes
                + [
                    h.Vector().record(node(0.5)._ref_v)
                    for node in self.nodes[self.passive_end_nodes : -self.passive_end_nodes]
                ]
                + [None] * self.passive_end_nodes
            )
        else:
            self.vm = [h.Vector().record(node(0.5)._ref_v) for node in self]

    def set_save_gating(self: Fiber) -> None:
        """Record gating parameters for axon nodes."""
        assert self.gating_variables, "Gating variables not defined for this fiber type"

        nodelist = (
            self.nodes if not self.passive_end_nodes else self.nodes[self.passive_end_nodes : -self.passive_end_nodes]
        )

        self.gating = {}
        for name, var in self.gating_variables.items():
            self.gating[name] = []
            for node in nodelist:
                self.gating[name].append(h.Vector().record(getattr(node(0.5), f"_ref_{var}")))
            if self.passive_end_nodes:
                self.gating[name] = (
                    [None] * self.passive_end_nodes + self.gating[name] + [None] * self.passive_end_nodes
                )

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


class _HomogeneousFiber(Fiber):
    """Initialize Homogeneous (all sections are identical) class."""

    def __init__(
        self: _HomogeneousFiber, fiber_model: FiberModel, diameter: float, delta_z: float = 8.333, **kwargs
    ) -> None:
        """Initialize UnmyelinatedFiber class.

        :param fiber_model: name of fiber model type
        :param diameter: fiber diameter [microns]
        :param delta_z: length of each node/section
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, diameter=diameter, **kwargs)
        self.delta_z = delta_z
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
        self.coordinates = (
            np.cumsum([section.L for section in self.sections]) - self.sections[0].L / 2
        )  # end of each section

        self.length = np.sum([section.L for section in self.sections])  # actual length of fiber

        assert np.isclose(
            self.coordinates[-1] - self.coordinates[0],  # center to center length
            self.delta_z * (self.nodecount - 1),  # expected length of fiber
        ), "Fiber length is not correct."

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
        for i in range(self.nodecount - 1):
            self.sections[i + 1].connect(self.sections[i])

        self.nodes = self.sections
