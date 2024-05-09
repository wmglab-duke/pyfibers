"""The copyrights of this software are owned by Duke University."""

from __future__ import annotations

import math
from typing import Callable, TypedDict

import numpy as np
from neuron import h

from pyfibers import FiberModel
from pyfibers.fiber import _Fiber

h.load_file('stdrun.hoc')


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

        self.length = np.sum([section.L for section in self.sections])  # actual length of fiber

        assert np.isclose(
            self.coordinates[-1] - self.coordinates[0],  # center to center length
            self.delta_z * (self.nodecount - 1),  # expected length of fiber
        ), "Fiber length is not correct."

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
            self.delta_z: float = fiber_param_discrete['delta_zs'][diameter_index]

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
        # check if this is a passive node
        self.passive = self.passive_end_nodes and (
            (index - 1) / 11 < self.passive_end_nodes
            or (self.nodecount - 1) - (index - 1) / 11 < self.passive_end_nodes
        )

        space_p1 = 0.002  # Thickness of periaxonal space in MYSA sections [um]
        # periaxonal space resistivity for node of Ranvier fiber segment [Mohms/cm]
        rpn0 = (rhoa * 0.01) / (math.pi * ((((node_diam / 2) + space_p1) ** 2) - ((node_diam / 2) ** 2)))

        name = f'active node {index}' if not self.passive else f'passive node {index}'
        node = h.Section(name=name)
        node.nseg = 1
        node.diam = node_diam
        node.L = nodelength
        node.Ra = rhoa / 10000

        if self.passive:
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
