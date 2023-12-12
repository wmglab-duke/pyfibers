"""The copyrights of this software are owned by Duke University."""

from __future__ import annotations

import math
import typing
import warnings
from typing import Callable, TypedDict

import numpy as np
from neuron import h
from numpy import ndarray

from wmglab_neuron import FiberModel

h.load_file('stdrun.hoc')


def build_fiber(
    fiber_model: FiberModel, diameter: float, n_sections: int = None, length: float = None, **kwargs
) -> _Fiber:
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
        fiberclass = MRGFiber(fiber_model, diameter, **kwargs)
    elif fiber_model == FiberModel.RATTAY:
        fiberclass = RattayFiber(fiber_model, diameter, **kwargs)
    elif fiber_model == FiberModel.TIGERHOLM:
        fiberclass = TigerholmFiber(fiber_model, diameter, **kwargs)
    elif fiber_model == FiberModel.SUNDT:
        fiberclass = SundtFiber(fiber_model, diameter, **kwargs)
    elif fiber_model in [FiberModel.SCHILD94, FiberModel.SCHILD97]:
        fiberclass = SchildFiber(fiber_model, diameter, **kwargs)
    else:
        raise ValueError("Fiber Model not valid")

    fiberclass.generate(n_sections, length)

    fiberclass.potentials = np.zeros(len(fiberclass.coordinates))

    assert len(fiberclass) == fiberclass.nodecount, "Node count does not match number of nodes"

    return fiberclass


class _Fiber:
    def __init__(  # TODO update tests
        self: _Fiber,
        fiber_model: FiberModel,
        diameter: float,
        temperature: float = 37,
        passive_end_nodes: bool = True,
    ) -> None:
        """Initialize Fiber class.

        :param diameter: fiber diameter [um]
        :param fiber_model: name of fiber model type
        :param temperature: temperature of model [degrees celsius]
        :param passive_end_nodes: if True, set passive properties at the end nodes
        """
        self.gating: dict[str, h.Vector] = {}
        self.gating_variables: dict[str, str] = {}
        self.vm: list = []
        self.apc: list = []
        self.diameter = diameter
        self.fiber_model = fiber_model
        self.temperature = temperature
        self.passive_end_nodes = passive_end_nodes
        self.nodecount: int = None
        self.delta_z: float = None
        self.sections: list = []
        self.nodes: list = []
        self.coordinates: ndarray = np.array([])
        self.length: float = None
        self.potentials: ndarray = np.array([])

    def __str__(self: _Fiber) -> str:
        """Return a string representation of the fiber."""  # noqa: DAR201
        return (
            f"{self.fiber_model.name} fiber of diameter {self.diameter} μm and length {self.length:.2f} μm "
            f"\n\tnode count: {len(self)}, section count: {len(self.sections)}"
        )

    def __repr__(self: _Fiber) -> str:
        """Return a string representation of the fiber."""  # noqa: DAR201
        # TODO: make this more informative for developers
        return self.__str__()

    def __len__(self: _Fiber) -> int:
        """Return the number of nodes in the fiber."""  # noqa: DAR201
        assert self.nodecount == len(self.nodes), "Node count does not match number of nodes"
        return len(self.nodes)

    def __getitem__(self: _Fiber, item: int) -> h.Section:
        """Return the node at the given index."""  # noqa: DAR201, DAR101
        return self.nodes[item]

    def __iter__(self: _Fiber) -> typing.Iterator[h.Section]:
        """Return an iterator over the nodes in the fiber."""  # noqa: DAR201
        return iter(self.nodes)

    def __contains__(self: _Fiber, item: h.Section) -> bool:
        """Return True if the section is in the fiber."""  # noqa: DAR201, DAR101
        return item in self.sections

    def loc(self: _Fiber, loc: float) -> h.Section:
        """Return the node at the given location (Using the same convention as NEURON).

        :param loc: location in the fiber (from 0 to 1)
        :return: node at the given location
        """
        return self.nodes[self.loc_index(loc)]

    def loc_index(self: _Fiber, loc: float) -> int:
        """Return the index of the node at the given location (Using the same convention as NEURON).

        :param loc: location in the fiber (from 0 to 1)
        :return: index of the node at the given location
        """
        return int(loc * (len(self) - 1))

    def resample_potentials(
        self: _Fiber,
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

    def apcounts(self: _Fiber, thresh: float = -30) -> None:
        """Create a list of NEURON APCount objects at all nodes along the axon.

        :param thresh: the threshold value for Vm to pass for an AP to be detected [mV]
        """
        self.apc = [h.APCount(node(0.5)) for node in self]
        for apc in self.apc:
            apc.thresh = thresh

    def set_save_vm(self: _Fiber) -> None:
        """Record membrane voltage (mV) along the axon."""
        if self.passive_end_nodes:
            self.vm = [None] + [h.Vector().record(node(0.5)._ref_v) for node in self.nodes[1:-1]] + [None]
        else:
            self.vm = [h.Vector().record(node(0.5)._ref_v) for node in self]

    def set_save_gating(self: _Fiber) -> None:
        """Record gating parameters for axon nodes."""
        assert self.gating_variables, "Gating variables not defined for this fiber type"

        if self.passive_end_nodes:
            nodelist = self.nodes[1:-1]
        else:
            nodelist = self.nodes

        self.gating = {}
        for name, var in self.gating_variables.items():
            self.gating[name] = []
            for node in nodelist:
                self.gating[name].append(h.Vector().record(getattr(node(0.5), f"_ref_{var}")))
            if self.passive_end_nodes:
                self.gating[name] = [None] + self.gating[name] + [None]

    def point_source_potentials(
        self: _Fiber,
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
                4
                * np.pi
                * np.sqrt(sigma_y * sigma_z * xs**2 + sigma_x * sigma_z * ys**2 + sigma_x * sigma_y * zs**2)
            )

        if inplace:
            self.potentials = potentials

        return potentials


class MRGFiber(_Fiber):
    """Implementation of the MRG fiber model."""

    def __init__(self: MRGFiber, fiber_model: FiberModel, diameter: float, **kwargs) -> None:
        """Initialize MRGFiber class.

        :param fiber_model: name of fiber model type
        :param diameter: fiber diameter [microns]
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, diameter=diameter, **kwargs)
        self.gating_variables = {
            "h": "h_axnode_myel",
            "m": "m_axnode_myel",
            "mp": "mp_axnode_myel",
            "s": "s_axnode_myel",
        }
        self.myelinated = True
        self.v_rest = -80  # millivolts

    def generate(self: MRGFiber, n_sections: int, length: float) -> _Fiber:
        """Build fiber model sections with NEURON.

        :param n_sections: number of fiber coordinates
        :param length: desired length of fiber [um] (mutually exclusive with n_sections)
        :return: Fiber object
        """
        # Determine geometrical parameters for fiber based on fiber model
        axon_diam, nl, node_diam, paranodal_length_2 = self.get_mrg_params()

        if length is not None:
            n_sections = math.floor(length / self.delta_z) * 11 + 1
        else:
            assert (n_sections - 1) % 11 == 0, (
                "n_sections must be 1 + 11n where n is an integer one less than the " "number of nodes of Ranvier."
            )

        # Determine number of nodecount
        self.nodecount = int(1 + (n_sections - 1) / 11)

        # Create fiber sections
        self.create_sections(
            axon_diam,
            node_diam,
            node_diam,
            axon_diam,
            paranodal_length_2,
            nl,
        )
        # use section attribute L to generate coordinates
        # get the coordinates at the center of each section, each section is a different length
        start_coords = np.array([0] + [section.L for section in self.sections[:-1]])  # start of each section
        end_coords = np.array([section.L for section in self.sections])  # end of each section
        self.coordinates: np.ndarray = np.cumsum((start_coords + end_coords) / 2)  # center of each section

        self.length = self.coordinates[-1] - self.coordinates[0]  # actual length of fiber

        expected = self.delta_z * (self.nodecount - 1)  # expected length of fiber

        assert np.isclose(
            self.length, expected
        ), f"Fiber length is not correct. Expected {expected} but got {self.length}"

        return self

    def get_mrg_params(self: MRGFiber) -> tuple[float, float, float, float]:
        """Get geometrical parameters for MRG fiber model.

        :raises ValueError: if an invalid fiber diameter is passed in
        :return: geometrical parameters for MRG fiber model
        """

        class MRGDiscreteParameters(TypedDict):
            node_length: float
            paranodal_length_1: float
            diameters: list[float]
            delta_zs: list[int]
            paranodal_length_2s: list[int]
            gs: list[None | float]
            axon_diams: list[float]
            node_diams: list[float]
            nls: list[int]

        class MRGInterpolationParameters(TypedDict):
            node_length: float
            paranodal_length_1: float
            paranodal_length_2: Callable[[float], float]
            delta_z: Callable[[float], float]
            nl: Callable[[float], float]
            node_diam: Callable[[float], float]
            axon_diam: Callable[[float], float]

        class FiberParameters(TypedDict):
            MRG_DISCRETE: MRGDiscreteParameters
            MRG_INTERPOLATION: MRGInterpolationParameters

        fiber_parameters_all: FiberParameters = {  # TODO: needs comments
            "MRG_DISCRETE": {
                "node_length": 1.0,
                "paranodal_length_1": 3.0,
                "diameters": [1.0, 2.0, 5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0],
                "delta_zs": [100, 200, 500, 750, 1000, 1150, 1250, 1350, 1400, 1450, 1500],
                "paranodal_length_2s": [5, 10, 35, 38, 40, 46, 50, 54, 56, 58, 60],
                "gs": [None, None, 0.605, 0.63, 0.661, 0.69, 0.7, 0.719, 0.739, 0.767, 0.791],
                "axon_diams": [0.8, 1.6, 3.4, 4.6, 5.8, 6.9, 8.1, 9.2, 10.4, 11.5, 12.7],
                "node_diams": [0.7, 1.4, 1.9, 2.4, 2.8, 3.3, 3.7, 4.2, 4.7, 5.0, 5.5],
                "nls": [15, 30, 80, 100, 110, 120, 130, 135, 140, 145, 150],
            },
            "MRG_INTERPOLATION": {
                "node_length": 1.0,
                "paranodal_length_1": 3.0,
                "paranodal_length_2": lambda d: -0.1652 * d**2 + 6.354 * d - 0.2862,
                "delta_z": lambda d: -8.215 * d**2 + 272.4 * d - 780.2 if d >= 5.643 else 81.08 * d + 37.84,
                "nl": lambda d: -0.4749 * d**2 + 16.85 * d - 0.7648,
                "node_diam": lambda d: 0.01093 * d**2 + 0.1008 * d + 1.099,
                "axon_diam": lambda d: 0.02361 * d**2 + 0.3673 * d + 0.7122,
            },
        }
        paranodal_length_2: float | int
        nl: float | int

        if self.fiber_model == FiberModel.MRG_DISCRETE:
            fiber_param_discrete = fiber_parameters_all["MRG_DISCRETE"]
            try:
                diameter_index = fiber_param_discrete['diameters'].index(self.diameter)
            except IndexError:
                raise ValueError(
                    "Diameter chosen not valid for FiberModel.MRG_DISCRETE. "
                    "Choose from {fiber_parameters['diameters']}"
                )
            paranodal_length_2 = fiber_param_discrete['paranodal_length_2s'][diameter_index]
            axon_diam = fiber_param_discrete['axon_diams'][diameter_index]
            node_diam = fiber_param_discrete['node_diams'][diameter_index]
            nl = fiber_param_discrete['nls'][diameter_index]
            self.delta_z = fiber_param_discrete['delta_zs'][diameter_index]

        elif self.fiber_model == FiberModel.MRG_INTERPOLATION:
            fiber_param_interp = fiber_parameters_all["MRG_INTERPOLATION"]
            paranodal_length_2 = fiber_param_interp['paranodal_length_2'](self.diameter)
            nl = fiber_param_interp['nl'](self.diameter)
            node_diam = fiber_param_interp['node_diam'](self.diameter)
            axon_diam = fiber_param_interp['axon_diam'](self.diameter)
            self.delta_z = fiber_param_interp['delta_z'](self.diameter)
            if self.diameter < 2 or self.diameter > 16:
                raise ValueError("Diameter for FiberModel.MRG_INTERPOLATION must be between 2 and 16 um (inclusive)")
        return axon_diam, nl, node_diam, paranodal_length_2

    def create_sections(
        self: MRGFiber,
        stin_diam: float,
        node_diam: float,
        mysa_diam: float,
        flut_diam: float,
        flut_length: float,
        nl: float,
    ) -> MRGFiber:
        """Create and connect NEURON sections for a myelinated fiber type.

        :param stin_diam: diameter of internodal fiber segment (STIN) [um]
        :param node_diam: diameter of node of Ranvier fiber segment [um]
        :param mysa_diam: diameter of myelin attachment section of fiber segment (MYSA) [um]
        :param flut_diam: diameter of main section of paranode fiber segment (FLUT) [um]
        :param flut_length: length of main section of paranode fiber segment (FLUT) [um]
        :param nl: number of myelin lemella
        :return: Fiber object
        """
        # Electrical parameters
        rhoa = 0.7e6  # [ohm-um]
        mycm = 0.1  # lamella membrane; [uF/cm2]
        mygm = 0.001  # lamella membrane; [S/cm2]

        nodelength = 1.0  # Length of nodes of Ranvier [um]
        paralength1 = 3  # Length of MYSA [um]

        interlength = (self.delta_z - nodelength - (2 * paralength1) - (2 * flut_length)) / 6

        # Create the axon sections
        nsegments = self.nodecount + 2 * (self.nodecount - 1) + 2 * (self.nodecount - 1) + 6 * (self.nodecount - 1)
        for ind in range(1, nsegments + 1):
            if ind % 11 == 1:
                section = self.create_node(
                    ind,
                    node_diam,
                    nodelength,
                    rhoa,
                    mycm,
                    mygm,
                    nl,
                )
                self.nodes.append(section)
            elif ind % 11 == 2 or ind % 11 == 0:
                section = self.create_mysa(ind, paralength1, rhoa, mysa_diam, mycm, mygm, nl)
            elif ind % 11 == 3 or ind % 11 == 10:
                section = self.create_flut(ind, flut_length, rhoa, flut_diam, mycm, mygm, nl)
            else:
                section = self.create_stin(ind, interlength, rhoa, stin_diam, mycm, mygm, nl)
            self.sections.append(section)

        # Connect the axon sections
        for i in range(nsegments - 1):
            self.sections[i + 1].connect(self.sections[i])

        return self

    def create_mysa(
        self: MRGFiber,
        i: int,
        paralength1: float,
        rhoa: float,
        para_diam_1: float,
        mycm: float,
        mygm: float,
        nl: float,
    ) -> h.Section:
        """Create a single MYSA segment for MRG_DISCRETE fiber type.

        :param i: index of fiber segment
        :param paralength1: length of myelin attachment section of fiber segment (MYSA) [um]
        :param rhoa: intracellular resistivity [Ohm-um]
        :param para_diam_1: diameter of myelin attachment section of fiber segment (MYSA) [um]
        :param mycm: lamella membrane capacitance [uF/cm2]
        :param mygm: lamella membrane conductance [uF/cm2]
        :param nl: number of myelin lemella
        :return: nrn.h.Section
        """
        space_p1 = 0.002  # Thickness of periaxonal space in MYSA sections [um]
        # periaxonal space resistivity for MYSA segment [Mohms/cm]
        rpn1 = (rhoa * 0.01) / (math.pi * ((((para_diam_1 / 2) + space_p1) ** 2) - ((para_diam_1 / 2) ** 2)))

        mysa = h.Section(name='mysa ' + str(i))
        mysa.nseg = 1
        mysa.diam = self.diameter
        mysa.L = paralength1
        mysa.Ra = rhoa * (1 / (para_diam_1 / self.diameter) ** 2) / 10000
        mysa.cm = 2 * para_diam_1 / self.diameter
        mysa.insert('pas')
        mysa.g_pas = 0.001 * para_diam_1 / self.diameter
        mysa.e_pas = self.v_rest

        mysa.insert('extracellular')
        mysa.xraxial[0] = rpn1
        mysa.xc[0] = mycm / (nl * 2)  # short circuit
        mysa.xg[0] = mygm / (nl * 2)  # short circuit

        return mysa

    def create_flut(
        self: MRGFiber,
        i: int,
        paralength2: float,
        rhoa: float,
        para_diam_2: float,
        mycm: float,
        mygm: float,
        nl: float,
    ) -> h.Section:
        """Create a single FLUT segment for MRG_DISCRETE fiber type.

        :param i: index of fiber segment
        :param paralength2: length of main section of paranode fiber segment (FLUT) [um]
        :param rhoa: intracellular resistivity [Ohm-um]
        :param para_diam_2: diameter of main section of paranode fiber segment (FLUT) [um]
        :param mycm: lamella membrane capacitance [uF/cm2]
        :param mygm: lamella membrane conductance [uF/cm2]
        :param nl: number of myelin lemella
        :return: nrn.h.Section
        """
        space_p2 = 0.004  # Thickness of periaxonal space in FLUT sections [um]
        # periaxonal space resistivity for of paranode fiber segment (FLUT) [Mohms/cm]
        rpn2 = (rhoa * 0.01) / (math.pi * ((((para_diam_2 / 2) + space_p2) ** 2) - ((para_diam_2 / 2) ** 2)))

        flut = h.Section(name='flut ' + str(i))
        flut.nseg = 1
        flut.diam = self.diameter
        flut.L = paralength2
        flut.Ra = rhoa * (1 / (para_diam_2 / self.diameter) ** 2) / 10000
        flut.cm = 2 * para_diam_2 / self.diameter
        flut.insert('pas')
        flut.g_pas = 0.0001 * para_diam_2 / self.diameter
        flut.e_pas = self.v_rest

        flut.insert('extracellular')
        flut.xraxial[0] = rpn2
        flut.xc[0] = mycm / (nl * 2)  # short circuit
        flut.xg[0] = mygm / (nl * 2)  # short circuit

        return flut

    def create_stin(
        self: MRGFiber,
        i: int,
        interlength: float,
        rhoa: float,
        axon_diam: float,
        mycm: float,
        mygm: float,
        nl: float,
    ) -> h.Section:
        """Create a STIN segment for MRG_DISCRETE fiber type.

        :param i: index of fiber segment
        :param interlength: length of internodal fiber segment (STIN) [um]
        :param rhoa: intracellular resistivity [Ohm-um]
        :param axon_diam: diameter of internodal fiber segment (STIN) [um]
        :param mycm: lamella membrane capacitance [uF/cm2]
        :param mygm: lamella membrane conductance [uF/cm2]
        :param nl: number of myelin lemella
        :return: nrn.h.Section
        """
        space_i = 0.004  # Thickness of periaxonal space in STIN sections [um]
        # periaxonal space resistivity for internodal fiber segment (STIN) [Mohms/cm]
        rpx = (rhoa * 0.01) / (math.pi * ((((axon_diam / 2) + space_i) ** 2) - ((axon_diam / 2) ** 2)))

        stin = h.Section(name='stin ' + str(i))
        stin.nseg = 1
        stin.diam = self.diameter
        stin.L = interlength
        stin.Ra = rhoa * (1 / (axon_diam / self.diameter) ** 2) / 10000
        stin.cm = 2 * axon_diam / self.diameter
        stin.insert('pas')
        stin.g_pas = 0.0001 * axon_diam / self.diameter
        stin.e_pas = self.v_rest

        stin.insert('extracellular')
        stin.xraxial[0] = rpx
        stin.xc[0] = mycm / (nl * 2)  # short circuit
        stin.xg[0] = mygm / (nl * 2)  # short circuit

        return stin

    def create_node(
        self: MRGFiber,
        index: int,
        node_diam: float,
        nodelength: float,
        rhoa: float,
        mycm: float,
        mygm: float,
        nl: float,
    ) -> h.Section:
        """Create a node of Ranvier for MRG_DISCRETE fiber type.

        :param index: index of fiber segment
        :param node_diam: diameter of node of Ranvier fiber segment [um]
        :param nodelength: Length of nodes of Ranvier [um]
        :param rhoa: intracellular resistivity [Ohm-um]
        :param mycm: lamella membrane capacitance [uF/cm2]
        :param mygm: lamella membrane conductance [uF/cm2]
        :param nl: number of myelin lemella
        :return: nrn.h.Section
        """
        space_p1 = 0.002  # Thickness of periaxonal space in MYSA sections [um]
        # periaxonal space resistivity for node of Ranvier fiber segment [Mohms/cm]
        rpn0 = (rhoa * 0.01) / (math.pi * ((((node_diam / 2) + space_p1) ** 2) - ((node_diam / 2) ** 2)))

        node = h.Section(name='node ' + str(index))
        node.nseg = 1
        node.diam = node_diam
        node.L = nodelength
        node.Ra = rhoa / 10000

        if self.passive_end_nodes and (index == 1 or index == (self.nodecount - 1) * 11 + 1):
            node.cm = 2
            node.insert('pas')
            node.g_pas = 0.0001
            node.e_pas = self.v_rest
            node.insert('extracellular')
            node.xc[0] = mycm / (nl * 2)  # short circuit
            node.xg[0] = mygm / (nl * 2)  # short circuit

        else:
            node.cm = 2
            node.insert('axnode_myel')
            node.insert('extracellular')
            node.xraxial[0] = rpn0
            node.xc[0] = 0  # short circuit
            node.xg[0] = 1e10  # short circuit

        return node


class _HomogeneousFiber(_Fiber):
    """Initialize Homogeneous (all sections are identical) class."""

    def __init__(self: _HomogeneousFiber, fiber_model: FiberModel, diameter: float, **kwargs) -> None:
        """Initialize UnmyelinatedFiber class.

        :param fiber_model: name of fiber model type
        :param diameter: fiber diameter [microns]
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, diameter=diameter, **kwargs)
        self.v_rest: int = None

    def generate_homogeneous(
        self: _HomogeneousFiber, n_sections: int, length: float, modelfunc: Callable, *args, **kwargs
    ) -> _Fiber:
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

        self.length = self.coordinates[-1] - self.coordinates[0]  # actual length of fiber

        expected = self.delta_z * (self.nodecount - 1)  # expected length of fiber

        assert np.isclose(
            self.length, expected
        ), f"Fiber length is not correct. Expected {expected} but got {self.length}"

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
            name = f"node {i}"
            if self.passive_end_nodes and (i == 0 or i == self.nodecount - 1):
                self.nodebuilder(self.passive_node)(self.v_rest, name=name)
            else:
                self.nodebuilder(modelnodefunc)(*args, name=name, **kwargs)
        for i in range(self.nodecount - 1):
            self.sections[i + 1].connect(self.sections[i])

        self.nodes = self.sections


class RattayFiber(_HomogeneousFiber):
    """Rattay fiber model."""

    def __init__(self: RattayFiber, fiber_model: FiberModel, diameter: float, **kwargs) -> None:
        """Initialize UnmyelinatedFiber class.

        :param fiber_model: name of fiber model type
        :param diameter: fiber diameter [microns]
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, diameter=diameter, **kwargs)
        self.myelinated = False
        self.delta_z = 8.333  # microns
        self.v_rest = -70  # millivolts

    def generate(self: RattayFiber, n_sections: int, length: float) -> _Fiber:  # noqa D102
        return self.generate_homogeneous(n_sections, length, self.create_rattay)

    @staticmethod
    def create_rattay(node: h.Section) -> None:
        """Create a RATTAY node.

        :param node: NEURON section
        """
        node.insert('RattayAberham')

        node.Ra = 100  # required for propagation; less than 100 does not propagate
        node.cm = 1
        node.ena = 45
        node.ek = -82


class TigerholmFiber(_HomogeneousFiber):
    """Tigerholm Fiber model."""

    def __init__(self: TigerholmFiber, fiber_model: FiberModel, diameter: float, **kwargs) -> None:
        """Initialize UnmyelinatedFiber class.

        :param fiber_model: name of fiber model type
        :param diameter: fiber diameter [microns]
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, diameter=diameter, **kwargs)
        self.myelinated = False
        self.delta_z = 8.333  # microns
        self.v_rest = -55  # millivolts

        if self.passive_end_nodes:
            warnings.warn('Ignoring passive_end_nodes for Tigerholm fiber', UserWarning, stacklevel=2)
            self.passive_end_nodes = False

    def generate(self: TigerholmFiber, n_sections: int, length: float) -> _Fiber:  # noqa D102
        return self.generate_homogeneous(n_sections, length, self.create_tigerholm, celsius=self.temperature)

    @staticmethod
    def create_tigerholm(node: h.Section, celsius: int) -> None:
        """Create a TIGERHOLM node.

        :param celsius: model temperature [celsius]
        :param node: NEURON section
        """
        node.insert('ks')
        node.insert('kf')
        node.insert('h')
        node.insert('nattxs')
        node.insert('nav1p8')
        node.insert('nav1p9')
        node.insert('nakpump')
        node.insert('kdrTiger')
        node.insert('kna')
        node.insert('naoiTiger')
        node.insert('koiTiger')
        node.insert('leak')
        node.insert('extrapump')

        node.gbar_kna = 0.00042
        node.Ra = 35.4
        node.cm = 1
        node.celsiusT_ks = celsius
        node.celsiusT_kf = celsius
        node.celsiusT_h = celsius
        node.celsiusT_nattxs = celsius
        node.celsiusT_nav1p8 = celsius
        node.celsiusT_nav1p9 = celsius
        node.celsiusT_nakpump = celsius
        node.celsiusT_kdrTiger = celsius
        node.gbar_ks = 0.0069733
        node.gbar_kf = 0.012756
        node.gbar_h = 0.0025377
        node.gbar_nattxs = 0.10664
        node.gbar_nav1p8 = 0.24271
        node.gbar_nav1p9 = 9.4779e-05
        node.smalla_nakpump = -0.0047891
        node.gbar_kdrTiger = 0.018002

    def balance(self: TigerholmFiber) -> None:
        """Balance membrane currents for Tigerholm model."""
        v_rest = self.v_rest
        for s in self.sections:
            if (-(s.ina_nattxs + s.ina_nav1p9 + s.ina_nav1p8 + s.ina_h + s.ina_nakpump) / (v_rest - s.ena)) < 0:
                s.pumpina_extrapump = -(s.ina_nattxs + s.ina_nav1p9 + s.ina_nav1p8 + s.ina_h + s.ina_nakpump)
            else:
                s.gnaleak_leak = -(s.ina_nattxs + s.ina_nav1p9 + s.ina_nav1p8 + s.ina_h + s.ina_nakpump) / (
                    v_rest - s.ena
                )

            if (-(s.ik_ks + s.ik_kf + s.ik_h + s.ik_kdrTiger + s.ik_nakpump + s.ik_kna) / (v_rest - s.ek)) < 0:
                s.pumpik_extrapump = -(s.ik_ks + s.ik_kf + s.ik_h + s.ik_kdrTiger + s.ik_nakpump + s.ik_kna)
            else:
                s.gkleak_leak = -(s.ik_ks + s.ik_kf + s.ik_h + s.ik_kdrTiger + s.ik_nakpump + s.ik_kna) / (
                    v_rest - s.ek
                )


class SundtFiber(_HomogeneousFiber):
    """Sundt fiber model."""

    def __init__(self: SundtFiber, fiber_model: FiberModel, diameter: float, **kwargs) -> None:
        """Initialize UnmyelinatedFiber class.

        :param fiber_model: enum of fiber model type
        :param diameter: fiber diameter [microns]
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, diameter=diameter, **kwargs)
        self.myelinated = False
        self.delta_z = 8.333  # microns
        self.v_rest = -60  # millivolts

    def generate(self: SundtFiber, n_sections: int, length: float) -> _Fiber:  # noqa D102
        return self.generate_homogeneous(n_sections, length, self.create_sundt)

    @staticmethod
    def create_sundt(node: h.Section) -> None:
        """Create a SUNDT node.

        :param node: NEURON section
        """
        node.insert('nahh')
        node.insert('borgkdr')  # insert delayed rectified K channels
        node.insert('pas')  # insert leak channels

        node.gnabar_nahh = 0.04
        node.mshift_nahh = -6  # NaV1.7/1.8 channelshift
        node.hshift_nahh = 6  # NaV1.7/1.8 channelshift
        node.gkdrbar_borgkdr = 0.04  # density of K channels
        node.ek = -90  # K equilibrium potential
        node.g_pas = 1 / 10000  # set Rm = 10000 ohms-cm2
        node.Ra = 100  # intracellular resistance
        node.e_pas = node.v + (node.ina + node.ik) / node.g_pas  # calculate leak equilibrium potential


class SchildFiber(_HomogeneousFiber):
    """Schild fiber model."""

    def __init__(self: SchildFiber, fiber_model: FiberModel, diameter: float, **kwargs) -> None:
        """Initialize UnmyelinatedFiber class.

        :param fiber_model: name of fiber model type
        :param diameter: fiber diameter [microns]
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, diameter=diameter, **kwargs)
        self.myelinated = False
        self.delta_z = 8.333  # microns

        self.v_rest = -48
        self.model97 = fiber_model == FiberModel.SCHILD97

    def generate(self, n_sections: int, length: float) -> _Fiber:  # noqa D102
        return self.generate_homogeneous(
            n_sections,
            length,
            self.create_schild,
            celsius=self.temperature,
            model97=self.model97,
            ca_l=self.delta_z,
            ca_nseg=1,
        )

    @staticmethod
    def create_schild(
        node: h.Section, celsius: float, model97: bool = False, ca_l: float = 8.3333, ca_nseg: int = 1
    ) -> None:
        """Create a SCHILD node.

        :param celsius: model temperature [celsius]
        :param node: NEURON section
        :param model97: True for Schild 1997 model, False for Schild 1994 model
        :param ca_l: length of node for calculating conductances
        :param ca_nseg: number of node segments for calculating conductances
        """
        R = 8314  # noqa: N806 # molar gas constant
        F = 96500  # noqa: N806 # faraday's constant
        node.insert('leakSchild')  # All mechanisms from Schild 1994 inserted into model
        node.insert('kd')
        node.insert('ka')
        node.insert('can')
        node.insert('cat')
        node.insert('kds')
        node.insert('kca')
        node.insert('caextscale')
        node.insert('caintscale')
        node.insert('CaPump')
        node.insert('NaCaPump')
        node.insert('NaKpumpSchild')
        if model97:
            node.insert('naf97mean')
            node.insert('nas97mean')
        else:
            node.insert('naf')
            node.insert('nas')

        # Ionic concentrations
        h.cao0_ca_ion = 2.0  # [mM] Initial Cao Concentration
        h.cai0_ca_ion = 0.000117  # [mM] Initial Cai Concentrations
        node.ko = 5.4  # [mM] External K Concentration
        node.ki = 145.0  # [mM] Internal K Concentration
        h.ion_style("k_ion", 1, 2, 0, 0, 0, sec=node)  # Allows ek to be calculated manually
        node.ek = ((R * (celsius + 273.15)) / F) * np.log(
            node.ko / node.ki
        )  # Manual Calculation of ek in order to use Schild F and R values
        node.nao = 154  # [mM] External Na Concentration
        node.nai = 8.9  # [mM] Internal Na Concentration
        h.ion_style("na_ion", 1, 2, 0, 0, 0, sec=node)  # Allows ena to be calculated manually
        node.ena = ((R * (celsius + 273.15)) / F) * np.log(
            node.nao / node.nai
        )  # Manual Calculation of ena in order to use Schild F and R values
        if model97:
            node.gbar_naf97mean = (
                0.022434928  # [S/cm^2] This block sets the conductance to the conductances in Schild 1997
            )
            node.gbar_nas97mean = 0.022434928
            node.gbar_kd = 0.001956534
            node.gbar_ka = 0.001304356
            node.gbar_kds = 0.000782614
            node.gbar_kca = 0.000913049
            node.gbar_can = 0.000521743
            node.gbar_cat = 0.00018261
            node.gbna_leakSchild = 1.8261e-05
            node.gbca_leakSchild = 9.13049e-06
        node.Ra = 100
        node.cm = 1.326291192

        node.L_caintscale = ca_l
        node.nseg_caintscale = ca_nseg
        node.L_caextscale = ca_l
        node.nseg_caextscale = ca_nseg
