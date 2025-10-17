"""Implementation of the MRG fiber model.

Based on the following papers:
Original implementation: https://doi.org/10.1152/jn.00353.2001
Extension to 2 um diameter: https://doi.org/10.1152/jn.00989.2003
Extension to 1 um diameter: https://doi.org/10.1088/1741-2552/aa6a5f
Implementation of interpolation model: https://doi.org/10.1371/journal.pcbi.1009285
Implementation of small interpolation model: https://doi.org/10.1371/journal.pcbi.1011833
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable
from typing import TypedDict

from neuron import h

from pyfibers.fiber import Fiber

h.load_file("stdrun.hoc")

# Set up module-level logger
logger = logging.getLogger(__name__)


# Classes to enable type hinting for dictionaries
class MRGDiscreteParameters(TypedDict):  # noqa: D101
    node_length: list[float]
    paranodal_length_1: list[float]
    diameters: list[float]
    delta_z: list[int]
    paranodal_length_2: list[int]
    axon_diam: list[float]
    node_diam: list[float]
    nl: list[int]
    rhoa: list[float]
    mycm: list[float]
    mygm: list[float]


class MRGInterpolationParameters(TypedDict):  # noqa: D101
    node_length: Callable[[float], float]
    paranodal_length_1: Callable[[float], float]
    paranodal_length_2: Callable[[float], float]
    delta_z: Callable[[float], float]
    nl: Callable[[float], float]
    node_diam: Callable[[float], float]
    axon_diam: Callable[[float], float]
    rhoa: Callable[[float], float]
    mycm: Callable[[float], float]
    mygm: Callable[[float], float]


class FiberParameters(TypedDict):  # noqa: D101
    MRG_DISCRETE: MRGDiscreteParameters
    MRG_INTERPOLATION: MRGInterpolationParameters
    PENA: MRGInterpolationParameters


# Parameters that define each fiber model variant
fiber_parameters_all: FiberParameters = {
    "MRG_DISCRETE": MRGDiscreteParameters(
        node_length=[1.0] * 11,
        paranodal_length_1=[3.0] * 11,
        rhoa=[0.7e6] * 11,
        mycm=[0.1] * 11,
        mygm=[0.001] * 11,
        diameters=[1.0, 2.0, 5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0],
        delta_z=[100, 200, 500, 750, 1000, 1150, 1250, 1350, 1400, 1450, 1500],
        paranodal_length_2=[5, 10, 35, 38, 40, 46, 50, 54, 56, 58, 60],
        axon_diam=[0.8, 1.6, 3.4, 4.6, 5.8, 6.9, 8.1, 9.2, 10.4, 11.5, 12.7],
        node_diam=[0.7, 1.4, 1.9, 2.4, 2.8, 3.3, 3.7, 4.2, 4.7, 5.0, 5.5],
        nl=[15, 30, 80, 100, 110, 120, 130, 135, 140, 145, 150],
    ),
    "MRG_INTERPOLATION": MRGInterpolationParameters(
        node_length=lambda d: 1.0,
        paranodal_length_1=lambda d: 3.0,
        rhoa=lambda d: 0.7e6,
        mycm=lambda d: 0.1,
        mygm=lambda d: 0.001,
        paranodal_length_2=lambda d: -0.1652 * d**2 + 6.354 * d - 0.2862,
        delta_z=lambda d: -8.215 * d**2 + 272.4 * d - 780.2 if d >= 5.643 else 81.08 * d + 37.84,
        nl=lambda d: -0.4749 * d**2 + 16.85 * d - 0.7648,
        node_diam=lambda d: 0.01093 * d**2 + 0.1008 * d + 1.099,
        axon_diam=lambda d: 0.02361 * d**2 + 0.3673 * d + 0.7122,
    ),
    "PENA": MRGInterpolationParameters(
        node_length=lambda d: 1.0,
        paranodal_length_1=lambda d: 3.0,
        rhoa=lambda d: 0.7e6,
        mycm=lambda d: 0.1,
        mygm=lambda d: 0.001,
        paranodal_length_2=lambda d: -0.171 * d**2 + 6.48 * d - 0.935,
        delta_z=lambda d: -3.22 * d**2 + 148 * d - 128,
        nl=lambda d: int(17.4 * (0.553 * d - 0.024) - 1.74),
        node_diam=lambda d: 0.321 * (0.553 * d - 0.024) + 0.37,
        axon_diam=lambda d: 0.553 * d - 0.024,
    ),
}


class MRGFiber(Fiber):
    """Implementation of the MRG fiber model."""

    submodels = ['MRG_DISCRETE', 'MRG_INTERPOLATION', 'SMALL_MRG_INTERPOLATION', 'PENA']

    def __init__(self: MRGFiber, diameter: float, **kwargs) -> None:
        """Initialize MRGFiber class.

        :param diameter: Fiber diameter [microns].
        :param kwargs: Keyword arguments to pass to the base class.
        :raises ValueError: If delta_z is specified in kwargs.
        """
        self.mrg_params: dict = None
        if "delta_z" in kwargs:
            raise ValueError("Cannot specify delta_z for MRG Fiber")
        super().__init__(diameter=diameter, **kwargs)
        self.gating_variables = {
            "h": "h_axnode_myel",
            "m": "m_axnode_myel",
            "mp": "mp_axnode_myel",
            "s": "s_axnode_myel",
        }
        self.myelinated = True
        self.v_rest = -80  # millivolts
        self.get_mrg_params()

    def generate(self: MRGFiber, **kwargs) -> Fiber:
        """Build fiber model sections with NEURON.

        :param kwargs: passed to superclass generate method
        :return: Fiber object
        """
        # Determine geometrical parameters for fiber based on fiber model
        self.get_mrg_params()
        # Function list for section order
        function_list = [
            self.create_node,
            self.create_mysa,
            self.create_flut,
            self.create_stin,
            self.create_stin,
            self.create_stin,
            self.create_stin,
            self.create_stin,
            self.create_stin,
            self.create_flut,
            self.create_mysa,
        ]

        return super().generate(function_list, **kwargs)

    def get_mrg_params(self: MRGFiber) -> None:
        """Get geometrical parameters for MRG fiber model and save to self.

        :raises ValueError: If an invalid fiber diameter is passed in.
        """
        if self.fiber_model.name == "MRG_DISCRETE":
            fiber_param_discrete = fiber_parameters_all["MRG_DISCRETE"]
            try:
                diameter_index = fiber_param_discrete["diameters"].index(self.diameter)
            except ValueError:
                raise ValueError(
                    "Diameter chosen not valid for MRG_DISCRETE. "
                    "Choose from {fiber_parameters_all['MRG_DISCRETE']['diameters']}"
                )
            self.mrg_params = {
                param: fiber_param_discrete[param][diameter_index]  # type: ignore
                for param in fiber_param_discrete.keys()
            }
        elif self.fiber_model.name == "MRG_INTERPOLATION":
            fiber_param_interp = fiber_parameters_all["MRG_INTERPOLATION"]
            self.mrg_params = {
                param: fiber_param_interp[param](self.diameter) for param in fiber_param_interp.keys()  # type: ignore
            }
            if self.diameter < 2 or self.diameter > 16:
                raise ValueError("Diameter for MRG_INTERPOLATION must be between 2 and 16 um (inclusive)")
        elif self.fiber_model.name in ["SMALL_MRG_INTERPOLATION", "PENA"]:
            # Show deprecation warning for old name
            if self.fiber_model.name == "SMALL_MRG_INTERPOLATION":
                warnings.warn(
                    "SMALL_MRG_INTERPOLATION is deprecated and will be removed in a future version. "
                    "Use PENA instead.",
                    FutureWarning,
                    stacklevel=2,
                )

            # Use PENA logic for both names
            fiber_param_interp = fiber_parameters_all["PENA"]
            self.mrg_params = {
                param: fiber_param_interp[param](self.diameter) for param in fiber_param_interp.keys()  # type: ignore
            }
            if self.diameter < 1.011 or self.diameter > 16:
                raise ValueError("Diameter for PENA must be between 1.011 and 16 um (inclusive)")
            if self.diameter > 5.7:
                logger.warning("%s fiber model is not recommended for fiber diameters above 5.7 um", self.fiber_model)
        self.delta_z = self.mrg_params["delta_z"]

    def create_mysa(self: MRGFiber, i: int) -> h.Section:
        """Create a single MYSA segment for MRG fiber type.

        :param i: index of fiber segment
        :return: nrn.h.Section
        """
        rhoa = self.mrg_params["rhoa"]  # intracellular resistivity [Ohm-um]
        mycm = self.mrg_params["mycm"]  # lamella membrane capacitance [uF/cm2]
        mygm = self.mrg_params["mygm"]  # lamella membrane conductance [uF/cm2]
        nl = self.mrg_params["nl"]  # number of myelin lemella
        mysa_diam = self.mrg_params["node_diam"]  # diameter of myelin attachment section of fiber segment (MYSA) [um]
        paralength1 = self.mrg_params["paranodal_length_1"]  # Length of MYSA [um]

        space_p1 = 0.002  # Thickness of periaxonal space in MYSA sections [um]
        # periaxonal space resistivity for MYSA segment [Mohms/cm]
        rpn1 = (rhoa * 0.01) / (math.pi * ((((mysa_diam / 2) + space_p1) ** 2) - ((mysa_diam / 2) ** 2)))

        mysa = h.Section(name="mysa " + str(i))
        mysa.nseg = 1
        mysa.diam = self.diameter
        mysa.L = paralength1
        mysa.Ra = rhoa * (1 / (mysa_diam / self.diameter) ** 2) / 10000
        mysa.cm = 2 * mysa_diam / self.diameter
        mysa.insert("pas")
        mysa.g_pas = 0.001 * mysa_diam / self.diameter
        mysa.e_pas = self.v_rest

        mysa.insert("extracellular")
        mysa.xraxial[0] = rpn1
        mysa.xc[0] = mycm / (nl * 2)
        mysa.xg[0] = mygm / (nl * 2)

        return mysa

    def create_flut(self: MRGFiber, i: int) -> h.Section:
        """Create a single FLUT segment for MRG fiber type.

        :param i: index of fiber segment
        :return: nrn.h.Section
        """
        rhoa = self.mrg_params["rhoa"]  # intracellular resistivity [Ohm-um]
        mycm = self.mrg_params["mycm"]  # lamella membrane capacitance [uF/cm2]
        mygm = self.mrg_params["mygm"]  # lamella membrane conductance [uF/cm2]
        nl = self.mrg_params["nl"]  # number of myelin lemella
        axon_diam = self.mrg_params["axon_diam"]  # diameter of main section of paranode fiber segment (FLUT) [um]
        flut_length = self.mrg_params[
            "paranodal_length_2"
        ]  # Length of main section of paranode fiber segment (FLUT) [um]

        space_p2 = 0.004  # Thickness of periaxonal space in FLUT sections [um]
        # periaxonal space resistivity for of paranode fiber segment (FLUT) [Mohms/cm]
        rpn2 = (rhoa * 0.01) / (math.pi * ((((axon_diam / 2) + space_p2) ** 2) - ((axon_diam / 2) ** 2)))

        flut = h.Section(name="flut " + str(i))
        flut.nseg = 1
        flut.diam = self.diameter
        flut.L = flut_length
        flut.Ra = rhoa * (1 / (axon_diam / self.diameter) ** 2) / 10000
        flut.cm = 2 * axon_diam / self.diameter
        flut.insert("pas")
        flut.g_pas = 0.0001 * axon_diam / self.diameter
        flut.e_pas = self.v_rest

        flut.insert("extracellular")
        flut.xraxial[0] = rpn2
        flut.xc[0] = mycm / (nl * 2)
        flut.xg[0] = mygm / (nl * 2)

        return flut

    def create_stin(self: MRGFiber, i: int) -> h.Section:
        """Create a STIN segment for MRG fiber type.

        :param i: index of fiber segment
        :return: nrn.h.Section
        """
        rhoa = self.mrg_params["rhoa"]  # intracellular resistivity [Ohm-um]
        mycm = self.mrg_params["mycm"]  # lamella membrane capacitance [uF/cm2]
        mygm = self.mrg_params["mygm"]  # lamella membrane conductance [uF/cm2]
        nl = self.mrg_params["nl"]  # number of myelin lemella
        axon_diam = self.mrg_params["axon_diam"]  # diameter of internodal fiber segment (STIN) [um]
        flut_length = self.mrg_params[
            "paranodal_length_2"
        ]  # Length of main section of paranode fiber segment (FLUT) [um]
        nodelength = self.mrg_params["node_length"]  # Length of nodes of Ranvier [um]
        paralength1 = self.mrg_params["paranodal_length_1"]  # Length of MYSA [um]
        interlength = (
            self.delta_z - nodelength - (2 * paralength1) - (2 * flut_length)
        ) / 6  # Length of internodal fiber segment (STIN) [um]

        space_i = 0.004  # Thickness of periaxonal space in STIN sections [um]
        # periaxonal space resistivity for internodal fiber segment (STIN) [Mohms/cm]
        rpx = (rhoa * 0.01) / (math.pi * ((((axon_diam / 2) + space_i) ** 2) - ((axon_diam / 2) ** 2)))

        stin = h.Section(name="stin " + str(i))
        stin.nseg = 1
        stin.diam = self.diameter
        stin.L = interlength
        stin.Ra = rhoa * (1 / (axon_diam / self.diameter) ** 2) / 10000
        stin.cm = 2 * axon_diam / self.diameter
        stin.insert("pas")
        stin.g_pas = 0.0001 * axon_diam / self.diameter
        stin.e_pas = self.v_rest

        stin.insert("extracellular")
        stin.xraxial[0] = rpx
        stin.xc[0] = mycm / (nl * 2)
        stin.xg[0] = mygm / (nl * 2)

        return stin

    def create_node(self: MRGFiber, index: int, node_type: str) -> h.Section:
        """Create a node of Ranvier for MRG_DISCRETE fiber type.

        :param index: index of fiber segment
        :param node_type: type of node (active or passive)
        :return: nrn.h.Section
        """
        rhoa = self.mrg_params["rhoa"]  # intracellular resistivity [Ohm-um]
        node_diam = self.mrg_params["node_diam"]  # diameter of node of Ranvier fiber segment [um]
        nodelength = self.mrg_params["node_length"]  # Length of nodes of Ranvier [um]

        name = f"{node_type} node {index}"
        node = h.Section(name=name)
        node.nseg = 1
        node.diam = node_diam
        node.L = nodelength
        node.Ra = rhoa / 10000
        node.cm = 2

        space_p1 = 0.002  # Thickness of periaxonal space in MYSA sections [um]
        # periaxonal space resistivity for node of Ranvier fiber segment [Mohms/cm]
        rpn0 = (rhoa * 0.01) / (math.pi * ((((node_diam / 2) + space_p1) ** 2) - ((node_diam / 2) ** 2)))

        node.insert("axnode_myel")
        node.insert("extracellular")
        node.xraxial[0] = rpn0
        node.xc[0] = 0  # short circuit
        node.xg[0] = 1e10  # short circuit

        # adjust conductances (PENA only)
        if self.fiber_model.name == "PENA":
            node.gnabar_axnode_myel = 2.333333
            node.gkbar_axnode_myel = 0.115556

        return node
