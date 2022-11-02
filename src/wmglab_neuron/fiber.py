"""The copyrights of this software are owned by Duke University.

Please refer to the LICENSE and README.md files for licensing instructions.
The source code can be found on the following GitHub repository: https://github.com/wmglab-duke/ascent
"""

import math

import numpy as np
from neuron import h

from src.wmglab_neuron import FiberModel, FiberTypeParameters

Section = h.Section
SectionList = h.SectionList
ion_style = h.ion_style

h.load_file('stdrun.hoc')


class FiberBuilder:
    """Builds a fiber model in NEURON.

    Used to select the correct class for the fiber model.
    """

    @staticmethod
    def generate(fiber_model: FiberModel, *args, n_fiber_coords: int = None, length: float = None, **kwargs):
        """Generate a fiber model in NEURON.

        :param fiber_model: fiber model to use
        :param args: arguments to pass to the fiber model class
        :param n_fiber_coords: number of fiber coordinates to use
        :param length: length of the fiber
        :param kwargs: keyword arguments to pass to the fiber model class
        :return: generated instance of fiber model class
        """
        assert (length is not None) or (n_fiber_coords is not None), "Must specify either length or n_fiber_coords"
        assert (length is None) or (n_fiber_coords is None), "Can't specify both length and n_fiber_coords"

        if fiber_model in [FiberModel.MRG_DISCRETE, FiberModel.MRG_INTERPOLATION]:
            fiberclass = MRGFiber(fiber_model, *args, **kwargs)
        else:
            fiberclass = UnmyelinatedFiber(fiber_model, *args, **kwargs)
        return fiberclass.generate(n_fiber_coords, length)

    # TODO: rework generation to be all insertions followed by all attributes


class _Fiber:
    pass

    def __init__(self, diameter: float, fiber_model: FiberModel, temperature: float):
        """Initialize Fiber class.

        :param diameter: fiber diameter [um]
        :param fiber_model: name of fiber model type
        :param temperature: temperature of model [degrees celsius]
        """
        self.diameter = diameter
        self.fiber_model = fiber_model
        self.temperature = temperature
        self.nodecount = None
        self.delta_z = None
        self.passive_end_nodes = None
        self.sections = []
        self.nodes = []
        self.coordinates = None
        self.length = None

        self.fiber_parameters = FiberTypeParameters[fiber_model]
        self.v_rest = self.fiber_parameters['v_rest']
        self.myelinated = self.fiber_parameters['myelinated']

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


class MRGFiber(_Fiber):
    """Implementation of the MRG fiber model."""

    def __init__(self, fiber_model: FiberModel, *args, **kwargs):
        """Initialize MRGFiber class.

        :param fiber_model: name of fiber model type
        :param args: arguments to pass to the base class
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(fiber_model=fiber_model, *args, **kwargs)

    def generate(self, n_fiber_coords: int, length: float):
        """Build fiber model sections with NEURON.

        :param n_fiber_coords: number of fiber coordinates
        :param length: desired length of fiber [um] (mutually exclusive with n_fiber_coords)
        :return: Fiber object
        """
        fiber_parameters = self.fiber_parameters
        self.passive_end_nodes = fiber_parameters['passive_end_nodes']
        # Determine geometrical parameters for fiber based on fiber model
        if self.fiber_model == FiberModel.MRG_DISCRETE:
            diameter_index = fiber_parameters['diameters'].index(self.diameter)
            paranodal_length_2 = fiber_parameters['paranodal_length_2s'][diameter_index]
            axon_diam = fiber_parameters['axonDs'][diameter_index]
            node_diam = fiber_parameters['nodeDs'][diameter_index]
            para_diam_1 = fiber_parameters['paraD1s'][diameter_index]
            para_diam_2 = fiber_parameters['paraD2s'][diameter_index]
            nl = fiber_parameters['nls'][diameter_index]
            self.delta_z = fiber_parameters['delta_zs'][diameter_index]

        elif self.fiber_model == FiberModel.MRG_INTERPOLATION:
            if self.diameter >= 5.643:
                self.delta_z = -8.215 * self.diameter**2 + 272.4 * self.diameter - 780.2
            else:
                self.delta_z = 81.08 * self.diameter + 37.84
            paranodal_length_2 = -0.1652 * self.diameter**2 + 6.354 * self.diameter - 0.2862
            nl = -0.4749 * self.diameter**2 + 16.85 * self.diameter - 0.7648
            node_diam = 0.01093 * self.diameter**2 + 0.1008 * self.diameter + 1.099
            para_diam_1 = node_diam
            para_diam_2 = 0.02361 * self.diameter**2 + 0.3673 * self.diameter + 0.7122
            axon_diam = para_diam_2

        if length is not None:
            n_fiber_coords = math.floor(length / self.delta_z) * 11 + 1

        # Determine number of nodecount
        self.nodecount = int(1 + (n_fiber_coords - 1) / 11)

        # Create fiber sections
        self.create_sections(
            axon_diam,
            node_diam,
            para_diam_1,
            para_diam_2,
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

        e_pas_vrest = self.v_rest

        # Geometrical parameters [um]
        paranodes1 = 2 * (self.nodecount - 1)  # Number of MYSA paranodes
        paranodes2 = 2 * (self.nodecount - 1)  # Number of FLUT paranodes
        axoninter = 6 * (self.nodecount - 1)  # Number of STIN internodes

        nodelength = 1.0  # Length of nodes of Ranvier [um]
        paralength1 = 3  # Length of MYSA [um]
        space_p1 = 0.002  # Thickness of periaxonal space in MYSA sections [um]
        space_p2 = 0.004  # Thickness of periaxonal space in FLUT sections [um]
        space_i = 0.004  # Thickness of periaxonal space in STIN sections [um]

        rpn0 = (rhoa * 0.01) / (math.pi * ((((node_diam / 2) + space_p1) ** 2) - ((node_diam / 2) ** 2)))
        rpn1 = (rhoa * 0.01) / (math.pi * ((((mysa_diam / 2) + space_p1) ** 2) - ((mysa_diam / 2) ** 2)))
        rpn2 = (rhoa * 0.01) / (math.pi * ((((flut_diam / 2) + space_p2) ** 2) - ((flut_diam / 2) ** 2)))
        rpx = (rhoa * 0.01) / (math.pi * ((((stin_diam / 2) + space_i) ** 2) - ((stin_diam / 2) ** 2)))
        interlength = (self.delta_z - nodelength - (2 * paralength1) - (2 * flut_length)) / 6

        # Create the axon sections
        node_ind = mysa_ind = stin_ind = flut_ind = 0
        nsegments = self.nodecount + paranodes1 + paranodes2 + axoninter
        for ind in range(1, nsegments + 1):
            if ind % 11 == 1:
                section = self.create_node(
                    node_ind,
                    node_diam,
                    nodelength,
                    rhoa,
                    mycm,
                    mygm,
                    self.passive_end_nodes,
                    self.nodecount,
                    nl,
                    rpn0,
                )
                node_ind += 1
                self.nodes.append(section)
            elif ind % 11 == 2 or ind % 11 == 0:
                section = self.create_mysa(
                    mysa_ind, self.diameter, paralength1, rhoa, mysa_diam, e_pas_vrest, rpn1, mycm, mygm, nl
                )
                mysa_ind += 1
            elif ind % 11 == 3 or ind % 11 == 10:
                section = self.create_flut(
                    flut_ind, self.diameter, flut_length, rhoa, flut_diam, e_pas_vrest, rpn2, mycm, mygm, nl
                )
                flut_ind += 1
            else:
                section = self.create_stin(
                    stin_ind, self.diameter, interlength, rhoa, stin_diam, e_pas_vrest, rpx, mycm, mygm, nl
                )
                stin_ind += 1
            self.sections.append(section)

        # Connect the axon sections
        for i in range(0, nsegments - 1):
            self.sections[i + 1].connect(self.sections[i])

        return self

    @staticmethod
    def create_mysa(
        i: int,
        fiber_diam: float,
        paralength1: float,
        rhoa: float,
        para_diam_1: float,
        e_pas_vrest: float,
        rpn1: float,
        mycm: float,
        mygm: float,
        nl: int,
    ):
        """Create a single MYSA segment for MRG_DISCRETE fiber type.

        :param i: index of fiber segment
        :param fiber_diam: fiber diameter [um]
        :param paralength1: length of myelin attachment section of fiber segment (MYSA) [um]
        :param rhoa: intracellular resistivity [Ohm-um]
        :param para_diam_1: diameter of myelin attachment section of fiber segment (MYSA) [um]
        :param e_pas_vrest: resting potential of axon [mV]
        :param rpn1: periaxonal space resistivity for MYSA segment [Mohms/cm]
        :param mycm: lamella membrane capacitance [uF/cm2]
        :param mygm: lamella membrane conductance [uF/cm2]
        :param nl: number of myelin lemella
        :return: nrn.Section
        """
        mysa = Section(name='mysa ' + str(i))
        mysa.nseg = 1
        mysa.diam = fiber_diam
        mysa.L = paralength1
        mysa.Ra = rhoa * (1 / (para_diam_1 / fiber_diam) ** 2) / 10000
        mysa.cm = 2 * para_diam_1 / fiber_diam
        mysa.insert('pas')
        mysa.g_pas = 0.001 * para_diam_1 / fiber_diam
        mysa.e_pas = e_pas_vrest

        mysa.insert('extracellular')
        mysa.xraxial[0] = rpn1
        mysa.xc[0] = mycm / (nl * 2)  # short circuit
        mysa.xg[0] = mygm / (nl * 2)  # short circuit

        return mysa

    @staticmethod
    def create_flut(
        i: int,
        fiber_diam: float,
        paralength2: float,
        rhoa: float,
        para_diam_2: float,
        e_pas_vrest: float,
        rpn2: float,
        mycm: float,
        mygm: float,
        nl: float,
    ):
        """Create a single FLUT segment for MRG_DISCRETE fiber type.

        :param i: index of fiber segment
        :param fiber_diam: fiber diameter [um]
        :param paralength2: length of main section of paranode fiber segment (FLUT) [um]
        :param rhoa: intracellular resistivity [Ohm-um]
        :param para_diam_2: diameter of main section of paranode fiber segment (FLUT) [um]
        :param e_pas_vrest: resting potential of axon [mV]
        :param rpn2: periaxonal space resistivity for of paranode fiber segment (FLUT) [Mohms/cm]
        :param mycm: lamella membrane capacitance [uF/cm2]
        :param mygm: lamella membrane conductance [uF/cm2]
        :param nl: number of myelin lemella
        :return: nrn.Section
        """
        flut = Section(name='flut ' + str(i))
        flut.nseg = 1
        flut.diam = fiber_diam
        flut.L = paralength2
        flut.Ra = rhoa * (1 / (para_diam_2 / fiber_diam) ** 2) / 10000
        flut.cm = 2 * para_diam_2 / fiber_diam
        flut.insert('pas')
        flut.g_pas = 0.0001 * para_diam_2 / fiber_diam
        flut.e_pas = e_pas_vrest

        flut.insert('extracellular')
        flut.xraxial[0] = rpn2
        flut.xc[0] = mycm / (nl * 2)  # short circuit
        flut.xg[0] = mygm / (nl * 2)  # short circuit

        return flut

    @staticmethod
    def create_stin(
        i: int,
        fiber_diam: float,
        interlength: float,
        rhoa: float,
        axon_diam: float,
        e_pas_vrest: float,
        rpx: float,
        mycm: float,
        mygm: float,
        nl: int,
    ):
        """Create a STIN segment for MRG_DISCRETE fiber type.

        :param i: index of fiber segment
        :param fiber_diam: fiber diameter [um]
        :param interlength: length of internodal fiber segment (STIN) [um]
        :param rhoa: intracellular resistivity [Ohm-um]
        :param axon_diam: diameter of internodal fiber segment (STIN) [um]
        :param e_pas_vrest: resting potential of axon [mV]
        :param rpx: periaxonal space resistivity for internodal fiber segment (STIN) [Mohms/cm]
        :param mycm: lamella membrane capacitance [uF/cm2]
        :param mygm: lamella membrane conductance [uF/cm2]
        :param nl: number of myelin lemella
        :return: nrn.Section
        """
        stin = Section(name='stin ' + str(i))
        stin.nseg = 1
        stin.diam = fiber_diam
        stin.L = interlength
        stin.Ra = rhoa * (1 / (axon_diam / fiber_diam) ** 2) / 10000
        stin.cm = 2 * axon_diam / fiber_diam
        stin.insert('pas')
        stin.g_pas = 0.0001 * axon_diam / fiber_diam
        stin.e_pas = e_pas_vrest

        stin.insert('extracellular')
        stin.xraxial[0] = rpx
        stin.xc[0] = mycm / (nl * 2)  # short circuit
        stin.xg[0] = mygm / (nl * 2)  # short circuit

        return stin

    @staticmethod
    def create_node(
        index: int,
        node_diam: float,
        nodelength: float,
        rhoa: float,
        mycm: float,
        mygm: float,
        passive: float,
        axonnodes: float,
        nl: int,
        rpn0: float,
    ):
        """Create a node of Ranvier for MRG_DISCRETE fiber type.

        :param index: index of fiber segment
        :param node_diam: diameter of node of Ranvier fiber segment [um]
        :param nodelength: Length of nodes of Ranvier [um]
        :param rhoa: intracellular resistivity [Ohm-um]
        :param mycm: lamella membrane capacitance [uF/cm2]
        :param mygm: lamella membrane conductance [uF/cm2]
        :param passive: true for passive end node strategy, false otherwise
        :param axonnodes: number of node of Ranvier segments
        :param nl: number of myelin lemella
        :param rpn0: periaxonal space resistivity for node of Ranvier fiber segment [Mohms/cm]
        :return: nrn.Section
        """
        node = Section(name='node ' + str(index))
        node.nseg = 1
        node.diam = node_diam
        node.L = nodelength
        node.Ra = rhoa / 10000

        if passive and (index == 0 or index == axonnodes - 1):
            node.cm = 2
            node.insert('pas')
            node.g_pas = 0.0001
            node.e_pas = -70
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


class UnmyelinatedFiber(_Fiber):
    """Initialize MRGFiber class.

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

    def generate(self, n_fiber_coords: int, length: float):
        """Build fiber model sections with NEURON.

        :param n_fiber_coords: number of fiber coordinates from COMSOL
        :param length: length of fiber [um] (mutually exclusive with n_fiber_coords)
        :return: Fiber object
        """
        # Determine geometrical parameters for fiber based on fiber model
        self.delta_z = self.fiber_parameters['delta_zs']
        self.passive_end_nodes = self.fiber_parameters['passive_end_nodes']  # todo make arg

        # Determine number of nodecount
        self.nodecount = int(n_fiber_coords) if length is None else math.floor(length / self.delta_z)

        # Create fiber sections
        self.create_sections()

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

    def create_sections(self):
        """Create and connect NEURON sections for an unmyelinated fiber."""
        length = self.delta_z * self.nodecount
        nsegments = int(length / self.delta_z)

        self.sections = []
        for i in range(0, nsegments):
            node = Section(name='node ' + str(i))
            self.sections.append(node)

            node.diam = self.diameter
            node.nseg = 1
            node.L = self.delta_z
            if self.passive_end_nodes and (i == 0 or i == nsegments - 1):
                node.insert('pas')
                node.g_pas = 0.0001
                node.e_pas = self.v_rest
                node.insert('extracellular')
                node.xc[0] = 0  # short circuit
                node.xg[0] = 1e10  # short circuit

                node.Ra = 1e10

            else:
                if self.fiber_model == FiberModel.SUNDT:  # Sundt model
                    self.create_sundt(node, self.v_rest)
                elif self.fiber_model == FiberModel.TIGERHOLM:  # Tigerholm model
                    self.create_tigerholm(node, self.temperature, self.v_rest)
                elif self.fiber_model == FiberModel.RATTAY:
                    # Rattay and Aberham model -- adjusted for a resting potential of -70mV
                    self.create_rattay(node, self.v_rest)
                elif self.fiber_model in [FiberModel.SCHILD94, FiberModel.SCHILD97]:  # Schild model
                    self.create_schild(node, self.temperature, self.v_rest, self.fiber_model == FiberModel.SCHILD97)

                node.insert('extracellular')
                node.xc[0] = 0  # short circuit
                node.xg[0] = 1e10  # short circuit

        for i in range(0, nsegments - 1):
            self.sections[i + 1].connect(self.sections[i])

        self.nodes = self.sections

    @staticmethod
    def create_schild(node, celsius, v_rest, model97=False):
        """Create a SCHILD node.

        :param celsius: model temperature [celsius]
        :param node: NEURON section
        :param v_rest: resting potential [mV]
        :param model97: True for Schild 1997 model, False for Schild 1994 model
        """
        node.R = 8314
        node.F = 96500
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
        node.cao0_ca_ion = 2.0  # [mM] Initial Cao Concentration
        node.cai0_ca_ion = 0.000117  # [mM] Initial Cai Concentrations
        node.ko = 5.4  # [mM] External K Concentration
        node.ki = 145.0  # [mM] Internal K Concentration
        node.kstyle = ion_style("k_ion", 1, 2, 0, 0, 0)  # Allows ek to be calculated manually
        node.ek = ((node.R * (celsius + 273.15)) / node.F) * np.log10(
            node.ko / node.ki
        )  # Manual Calculation of ek in order to use Schild F and R values
        node.nao = 154  # [mM] External Na Concentration
        node.nai = 8.9  # [mM] Internal Na Concentration
        node.nastyle = ion_style("na_ion", 1, 2, 0, 0, 0)  # Allows ena to be calculated manually
        node.ena = ((node.R * (celsius + 273.15)) / node.F) * np.log10(
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
        node.v = v_rest

    @staticmethod
    def create_rattay(node, v_rest):
        """Create a RATTAY node.

        :param node: NEURON section
        :param v_rest: resting potential [mV]
        """
        node.insert('RattayAberham')
        node.Ra = 100  # required for propagation; less than 100 does not propagate
        node.cm = 1
        node.v = v_rest
        node.ena = 45
        node.ek = -82

    @staticmethod
    def create_tigerholm(node, celsius, v_rest):
        """Create a TIGERHOLM node.

        :param celsius: model temperature [celsius]
        :param node: NEURON section
        :param v_rest: resting potential [mV]
        """
        node.insert('ks')
        node.gbar_ks = 0.0069733
        node.insert('kf')
        node.gbar_kf = 0.012756
        node.insert('h')
        node.gbar_h = 0.0025377
        node.insert('nattxs')
        node.gbar_nattxs = 0.10664
        node.insert('nav1p8')
        node.gbar_nav1p8 = 0.24271
        node.insert('nav1p9')
        node.gbar_nav1p9 = 9.4779e-05
        node.insert('nakpump')
        node.smalla_nakpump = -0.0047891
        node.insert('kdrTiger')
        node.gbar_kdrTiger = 0.018002
        node.insert('kna')
        node.gbar_kna = 0.00042
        node.insert('naoiTiger')
        node.insert('koiTiger')
        node.insert('leak')
        node.insert('extrapump')
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
        node.v = v_rest

    @staticmethod
    def create_sundt(node, v_rest):
        """Create a SUNDT node.

        :param node: NEURON section
        :param v_rest: resting potential [mV]
        """
        node.insert('nahh')
        node.gnabar_nahh = 0.04
        node.mshift_nahh = -6  # NaV1.7/1.8 channelshift
        node.hshift_nahh = 6  # NaV1.7/1.8 channelshift
        node.insert('borgkdr')  # insert delayed rectified K channels
        node.gkdrbar_borgkdr = 0.04  # density of K channels
        node.ek = -90  # K equilibrium potential
        node.insert('pas')  # insert leak channels
        node.g_pas = 1 / 10000  # set Rm = 10000 ohms-cm2
        node.Ra = 100  # intracellular resistance
        node.v = v_rest
        node.e_pas = node.v + (node.ina + node.ik) / node.g_pas  # calculate leak equilibrium potential

    def balance(self):
        """Balance membrane currents for Tigerholm model.

        :raises ValueError: if the model is not Tigerholm
        """
        if not self.fiber_model == FiberModel.TIGERHOLM:
            raise ValueError('balance() is only valid for Tigerholm model')
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
