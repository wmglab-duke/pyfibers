"""The copyrights of this software are owned by Duke University.

Please refer to the LICENSE and README.md files for licensing instructions.
The source code can be found on the following GitHub repository: https://github.com/wmglab-duke/ascent
"""

import math
import warnings

import numpy as np
from neuron import h

from wmglab_neuron import FiberModel, FiberTypeParameters

Section = h.Section
SectionList = h.SectionList
ion_style = h.ion_style

h.load_file('stdrun.hoc')


class FiberBuilder:
    """Builds a fiber model in NEURON.

    Used to select the correct class for the fiber model.
    """

    @staticmethod
    def generate(fiber_model: FiberModel, *args, n_sections: int = None, length: float = None, **kwargs):
        """Generate a fiber model in NEURON.

        :param fiber_model: fiber model to use
        :param args: arguments to pass to the fiber model class
        :param n_sections: number of fiber coordinates to use
        :param length: length of the fiber
        :param kwargs: keyword arguments to pass to the fiber model class
        :raises ValueError: if the fiber model is not supported
        :return: generated instance of fiber model class
        """
        assert (length is not None) or (n_sections is not None), "Must specify either length or n_sections"
        assert (length is None) or (n_sections is None), "Can't specify both length and n_sections"
        # todo: maybe stop passing model, find a cleaner way to implement this factory
        if fiber_model in [FiberModel.MRG_DISCRETE, FiberModel.MRG_INTERPOLATION]:
            fiberclass = MRGFiber(fiber_model, *args, **kwargs)
        elif fiber_model == FiberModel.RATTAY:
            fiberclass = RattayFiber(fiber_model, *args, **kwargs)
        elif fiber_model == FiberModel.TIGERHOLM:
            fiberclass = TigerholmFiber(fiber_model, *args, **kwargs)
        elif fiber_model == FiberModel.SUNDT:
            fiberclass = SundtFiber(fiber_model, *args, **kwargs)
        elif fiber_model in [FiberModel.SCHILD94, FiberModel.SCHILD97]:
            fiberclass = SchildFiber(fiber_model, *args, **kwargs)
        else:
            raise ValueError("Fiber Model not valid")

        return fiberclass.generate(n_sections, length)


class _Fiber:
    def __init__(
        self, diameter: float, fiber_model: FiberModel, temperature: float = 37, passive_end_nodes: bool = True
    ):
        """Initialize Fiber class.

        :param diameter: fiber diameter [um]
        :param fiber_model: name of fiber model type
        :param temperature: temperature of model [degrees celsius]
        :param passive_end_nodes: if True, set passive properties at the end nodes
        """
        self.diameter = diameter
        self.fiber_model = fiber_model
        self.temperature = temperature
        self.passive_end_nodes = passive_end_nodes
        self.nodecount = None
        self.delta_z = None
        self.sections = []
        self.nodes = []
        self.coordinates = None
        self.length = None

        self.fiber_parameters = FiberTypeParameters[fiber_model]
        self.v_rest = self.fiber_parameters['v_rest']
        self.myelinated = self.fiber_parameters['myelinated']

    def __str__(self):
        """Return a string representation of the fiber."""  # noqa: DAR201
        return (
            f"{self.fiber_model.name} fiber with diameter {self.diameter} μm and length {self.length:.2f} μm "
            f"\n\tnodecount: {self.nodecount}, section count: {len(self.sections)}"
        )

    def __repr__(self):
        """Return a string representation of the fiber."""  # noqa: DAR201
        # TODO: make this more informative for developers
        return self.__str__()

    def __len__(self):
        """Return the number of nodes in the fiber."""  # noqa: DAR201
        return self.nodecount

    def __getitem__(self, item):
        """Return the node at the given index."""  # noqa: DAR201, DAR101
        return self.nodes[item]

    def __iter__(self):
        """Return an iterator over the nodes in the fiber."""  # noqa: DAR201
        return iter(self.nodes)

    def __contains__(self, item):
        """Return True if the section is in the fiber."""  # noqa: DAR201, DAR101
        return item in self.sections

    def loc(self, loc):
        """Return the node at the given location (Using the same convention as NEURON).

        :param loc: location in the fiber (from 0 to 1)
        :return: node at the given location
        """
        return self.nodes[int(loc * (self.nodecount - 1))]

    def resample_potentials(self, potentials, potential_coords, center: bool = False):
        """Use linear interpolation to resample the high-res potentials to the proper fiber coordinates.

        :param potentials: high-res potentials
        :param potential_coords: coordinates of high-res potentials
        :param center: if True, center the potentials around the fiber midpoint
        :return: resampled potentials
        """
        potential_coords, potentials = np.array(potential_coords), np.array(potentials)
        if not center:
            potential_coords = potential_coords - potential_coords[0]
            target_coords = self.coordinates
        else:
            target_coords = self.coordinates - (self.coordinates[0] + self.coordinates[-1]) / 2
            potential_coords = potential_coords - (potential_coords[0] + potential_coords[-1]) / 2
        return np.interp(target_coords, potential_coords, potentials)

    def apcounts(self, thresh: float = -30):
        """Create a list of NEURON APCount objects at all nodes along the axon.

        :param thresh: the threshold value for Vm to pass for an AP to be detected [mV]
        """
        self.apc = [h.APCount(node(0.5)) for node in self.nodes]
        for apc in self.apc:
            apc.thresh = thresh

    def set_save_vm(self):
        """Record membrane voltage (mV) along the axon."""
        if self.passive_end_nodes:
            self.vm = [None] + [h.Vector().record(node(0.5)._ref_v) for node in self.nodes[1:-1]] + [None]
        else:
            self.vm = [h.Vector().record(node(0.5)._ref_v) for node in self.nodes]


class MRGFiber(_Fiber):
    """Implementation of the MRG fiber model."""

    def __init__(self, fiber_model: FiberModel, *args, **kwargs):
        """Initialize MRGFiber class.

        :param fiber_model: name of fiber model type
        :param args: arguments to pass to the base class
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, *args, **kwargs)

    def generate(self, n_sections: int, length: float):
        """Build fiber model sections with NEURON.

        :param n_sections: number of fiber coordinates
        :param length: desired length of fiber [um] (mutually exclusive with n_sections)
        :raises ValueError: if an invalid fiber diameter is passed in
        :return: Fiber object
        """
        fiber_parameters = self.fiber_parameters
        # Determine geometrical parameters for fiber based on fiber model
        if self.fiber_model == FiberModel.MRG_DISCRETE:
            try:
                diameter_index = fiber_parameters['diameters'].index(self.diameter)
            except IndexError:
                raise ValueError(
                    "Diameter chosen not valid for FiberModel.MRG_DISCRETE. "
                    "Choose from {fiber_parameters['diameters']}"
                )
            paranodal_length_2 = fiber_parameters['paranodal_length_2s'][diameter_index]
            axon_diam = fiber_parameters['axonDs'][diameter_index]
            node_diam = fiber_parameters['nodeDs'][diameter_index]
            nl = fiber_parameters['nls'][diameter_index]
            self.delta_z = fiber_parameters['delta_zs'][diameter_index]

        elif self.fiber_model == FiberModel.MRG_INTERPOLATION:
            paranodal_length_2 = fiber_parameters['paranodal_length_2'](self.diameter)
            nl = fiber_parameters['nl'](self.diameter)
            node_diam = fiber_parameters['nodeD'](self.diameter)
            axon_diam = fiber_parameters['axonD'](self.diameter)
            self.delta_z = fiber_parameters['delta_z'](self.diameter)
            if self.diameter < 2 or self.diameter > 16:
                raise ValueError("Diameter for FiberModel.MRG_INTERPOLATION must be between 2 and 16 um (inclusive)")

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
        self.coordinates = np.cumsum((start_coords + end_coords) / 2)  # center of each section

        self.length = self.coordinates[-1] - self.coordinates[0]  # actual length of fiber

        expected = self.delta_z * (self.nodecount - 1)  # expected length of fiber

        assert np.isclose(
            self.length, expected
        ), f"Fiber length is not correct. Expected {expected} but got {self.length}"

        return self

    def create_sections(
        self,
        stin_diam: float,
        node_diam: float,
        mysa_diam: float,
        flut_diam: float,
        flut_length: float,
        nl: int,
    ):
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
        for i in range(0, nsegments - 1):
            self.sections[i + 1].connect(self.sections[i])

        return self

    def create_mysa(
        self,
        i: int,
        paralength1: float,
        rhoa: float,
        para_diam_1: float,
        mycm: float,
        mygm: float,
        nl: int,
    ):
        """Create a single MYSA segment for MRG_DISCRETE fiber type.

        :param i: index of fiber segment
        :param paralength1: length of myelin attachment section of fiber segment (MYSA) [um]
        :param rhoa: intracellular resistivity [Ohm-um]
        :param para_diam_1: diameter of myelin attachment section of fiber segment (MYSA) [um]
        :param mycm: lamella membrane capacitance [uF/cm2]
        :param mygm: lamella membrane conductance [uF/cm2]
        :param nl: number of myelin lemella
        :return: nrn.Section
        """
        space_p1 = 0.002  # Thickness of periaxonal space in MYSA sections [um]
        # periaxonal space resistivity for MYSA segment [Mohms/cm]
        rpn1 = (rhoa * 0.01) / (math.pi * ((((para_diam_1 / 2) + space_p1) ** 2) - ((para_diam_1 / 2) ** 2)))

        mysa = Section(name='mysa ' + str(i))
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
        self,
        i: int,
        paralength2: float,
        rhoa: float,
        para_diam_2: float,
        mycm: float,
        mygm: float,
        nl: float,
    ):
        """Create a single FLUT segment for MRG_DISCRETE fiber type.

        :param i: index of fiber segment
        :param paralength2: length of main section of paranode fiber segment (FLUT) [um]
        :param rhoa: intracellular resistivity [Ohm-um]
        :param para_diam_2: diameter of main section of paranode fiber segment (FLUT) [um]
        :param mycm: lamella membrane capacitance [uF/cm2]
        :param mygm: lamella membrane conductance [uF/cm2]
        :param nl: number of myelin lemella
        :return: nrn.Section
        """
        space_p2 = 0.004  # Thickness of periaxonal space in FLUT sections [um]
        # periaxonal space resistivity for of paranode fiber segment (FLUT) [Mohms/cm]
        rpn2 = (rhoa * 0.01) / (math.pi * ((((para_diam_2 / 2) + space_p2) ** 2) - ((para_diam_2 / 2) ** 2)))

        flut = Section(name='flut ' + str(i))
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
        self,
        i: int,
        interlength: float,
        rhoa: float,
        axon_diam: float,
        mycm: float,
        mygm: float,
        nl: int,
    ):
        """Create a STIN segment for MRG_DISCRETE fiber type.

        :param i: index of fiber segment
        :param interlength: length of internodal fiber segment (STIN) [um]
        :param rhoa: intracellular resistivity [Ohm-um]
        :param axon_diam: diameter of internodal fiber segment (STIN) [um]
        :param mycm: lamella membrane capacitance [uF/cm2]
        :param mygm: lamella membrane conductance [uF/cm2]
        :param nl: number of myelin lemella
        :return: nrn.Section
        """
        space_i = 0.004  # Thickness of periaxonal space in STIN sections [um]
        # periaxonal space resistivity for internodal fiber segment (STIN) [Mohms/cm]
        rpx = (rhoa * 0.01) / (math.pi * ((((axon_diam / 2) + space_i) ** 2) - ((axon_diam / 2) ** 2)))

        stin = Section(name='stin ' + str(i))
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
        self,
        index: int,
        node_diam: float,
        nodelength: float,
        rhoa: float,
        mycm: float,
        mygm: float,
        nl: int,
    ):
        """Create a node of Ranvier for MRG_DISCRETE fiber type.

        :param index: index of fiber segment
        :param node_diam: diameter of node of Ranvier fiber segment [um]
        :param nodelength: Length of nodes of Ranvier [um]
        :param rhoa: intracellular resistivity [Ohm-um]
        :param mycm: lamella membrane capacitance [uF/cm2]
        :param mygm: lamella membrane conductance [uF/cm2]
        :param nl: number of myelin lemella
        :return: nrn.Section
        """
        space_p1 = 0.002  # Thickness of periaxonal space in MYSA sections [um]
        # periaxonal space resistivity for node of Ranvier fiber segment [Mohms/cm]
        rpn0 = (rhoa * 0.01) / (math.pi * ((((node_diam / 2) + space_p1) ** 2) - ((node_diam / 2) ** 2)))

        node = Section(name='node ' + str(index))
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

    def set_save_gating(self):
        """Record gating parameters (h, m, mp, s) for myelinated fiber types."""
        self.gating = {"h": [], "m": [], "mp": [], "s": []}
        if self.passive_end_nodes:
            nodelist = self.nodes[1:-1]
        else:
            nodelist = self.nodes
        for node in nodelist:
            h_node = h.Vector().record(node(0.5)._ref_h_inf_axnode_myel)
            m_node = h.Vector().record(node(0.5)._ref_m_inf_axnode_myel)
            mp_node = h.Vector().record(node(0.5)._ref_mp_inf_axnode_myel)
            s_node = h.Vector().record(node(0.5)._ref_s_inf_axnode_myel)
            self.gating['h'].append(h_node)
            self.gating['m'].append(m_node)
            self.gating['mp'].append(mp_node)
            self.gating['s'].append(s_node)
        if self.passive_end_nodes:
            self.gating['h'] = [None] + self.gating['h'] + [None]
            self.gating['m'] = [None] + self.gating['m'] + [None]
            self.gating['mp'] = [None] + self.gating['mp'] + [None]
            self.gating['s'] = [None] + self.gating['s'] + [None]


class _HomogeneousFiber(_Fiber):
    """Initialize Homogeneous (all sections are identical) class.

    :param fiber_model: name of fiber model type
    :param args: arguments to pass to the base class
    :param kwargs: keyword arguments to pass to the base class
    """

    def __init__(self, fiber_model: FiberModel, *args, **kwargs):
        """Initialize UnmyelinatedFiber class.

        :param fiber_model: name of fiber model type
        :param args: arguments to pass to the base class
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, *args, **kwargs)

    def generate_homogeneous(self, n_sections: int, length: float, modelfunc, *args, **kwargs):
        """Build fiber model sections with NEURON.

        :param n_sections: number of fiber coordinates from COMSOL
        :param length: length of fiber [um] (mutually exclusive with n_sections)
        :param modelfunc: function to generate fiber model (mechanisms and attributes)
        :param args: arguments to pass to modelfunc
        :param kwargs: keyword arguments to pass to modelfunc
        :return: Fiber object
        """
        # Determine geometrical parameters for fiber based on fiber model
        self.delta_z = self.fiber_parameters['delta_zs']  # TODO: ability to specify section length

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

    def nodebuilder(self, nodefunc):
        """Generate a node and apply the specific model described by nodefunc.

        :param nodefunc: function to build node
        :return: nrn.Section
        """

        def wrapper(*args, name='node', **kwargs):
            node = Section(name=name)
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

    def sectionbuilder(self, modelnodefunc, *args, **kwargs):
        """Create and connect NEURON sections for an unmyelinated fiber.

        :param modelnodefunc: function to build node mechanisms and attributes
        :param args: arguments to pass to modelnodefunc
        :param kwargs: keyword arguments to pass to modelnodefunc
        """
        self.sections = []
        for i in range(0, self.nodecount):
            name = f"node {i}"
            if self.passive_end_nodes and (i == 0 or i == self.nodecount - 1):
                self.nodebuilder(self.passive_node)(self.v_rest, name=name)
            else:
                self.nodebuilder(modelnodefunc)(*args, name=name, **kwargs)
        for i in range(0, self.nodecount - 1):
            self.sections[i + 1].connect(self.sections[i])

        self.nodes = self.sections

    @staticmethod
    def passive_node(node, v_rest):
        node.insert('pas')
        node.g_pas = 0.0001
        node.e_pas = v_rest
        node.Ra = 1e10


class RattayFiber(_HomogeneousFiber):
    """Rattay fiber model."""

    def __init__(self, fiber_model: FiberModel, *args, **kwargs):
        """Initialize UnmyelinatedFiber class.

        :param fiber_model: name of fiber model type
        :param args: arguments to pass to the base class
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, *args, **kwargs)

    def generate(self, n_sections: int, length: float):  # noqa D102
        return self.generate_homogeneous(n_sections, length, self.create_rattay)

    @staticmethod
    def create_rattay(node):
        """Create a RATTAY node.

        :param node: NEURON section
        """
        node.insert('RattayAberham')

        node.Ra = 100  # required for propagation; less than 100 does not propagate
        node.cm = 1
        node.ena = 45
        node.ek = -82


class SchildFiber(_HomogeneousFiber):
    """Schild fiber model."""

    def __init__(self, fiber_model: FiberModel, *args, **kwargs):
        """Initialize UnmyelinatedFiber class.

        :param fiber_model: name of fiber model type
        :param args: arguments to pass to the base class
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, *args, **kwargs)

    def generate(self, n_sections: int, length: float):  # noqa D102
        return self.generate_homogeneous(n_sections, length, self.create_schild, celsius=self.temperature)

    @staticmethod
    def create_schild(node, celsius, model97=False):
        """Create a SCHILD node.

        :param celsius: model temperature [celsius]
        :param node: NEURON section
        :param model97: True for Schild 1997 model, False for Schild 1994 model
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
        ion_style("k_ion", 1, 2, 0, 0, 0, sec=node)  # Allows ek to be calculated manually
        node.ek = ((R * (celsius + 273.15)) / F) * np.log10(
            node.ko / node.ki
        )  # Manual Calculation of ek in order to use Schild F and R values
        node.nao = 154  # [mM] External Na Concentration
        node.nai = 8.9  # [mM] Internal Na Concentration
        ion_style("na_ion", 1, 2, 0, 0, 0, sec=node)  # Allows ena to be calculated manually
        node.ena = ((R * (celsius + 273.15)) / F) * np.log10(
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
            node.gbna_leak = 1.8261e-05
        node.Ra = 100
        node.cm = 1.326291192


class TigerholmFiber(_HomogeneousFiber):
    """Tigerholm Fiber model."""

    def __init__(self, fiber_model: FiberModel, *args, **kwargs):
        """Initialize UnmyelinatedFiber class.

        :param fiber_model: name of fiber model type
        :param args: arguments to pass to the base class
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, *args, **kwargs)
        if self.passive_end_nodes:
            warnings.warn('Ignoring passive_end_nodes for Tigerholm fiber', UserWarning, stacklevel=2)
            self.passive_end_nodes = False

    def generate(self, n_sections: int, length: float):  # noqa D102
        return self.generate_homogeneous(n_sections, length, self.create_tigerholm, celsius=self.temperature)

    @staticmethod
    def create_tigerholm(node, celsius):
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
        node.Ra = 35.5
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

    def balance(self):
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

    def __init__(self, fiber_model: FiberModel, *args, **kwargs):
        """Initialize UnmyelinatedFiber class.

        :param fiber_model: name of fiber model type
        :param args: arguments to pass to the base class
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, *args, **kwargs)

    def generate(self, n_sections: int, length: float):  # noqa D102
        return self.generate_homogeneous(n_sections, length, self.create_sundt)

    @staticmethod
    def create_sundt(node):
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
