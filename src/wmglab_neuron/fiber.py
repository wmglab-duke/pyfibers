"""The copyrights of this software are owned by Duke University.

Please refer to the LICENSE and README.md files for licensing instructions.
The source code can be found on the following GitHub repository: https://github.com/wmglab-duke/ascent
"""

import math

import numpy as np
from neuron import h

from src.wmglab_neuron import FiberTypeParameters

Section = h.Section
SectionList = h.SectionList
ion_style = h.ion_style

h.load_file('stdrun.hoc')


# TODO: every variable with flag in the name needs to be changed to something more descriptive


class Fiber:
    """Create a fiber model from NEURON sections."""

    def __init__(
        self, diameter: float, fiber_mode: str, temperature: float, n_fiber_coords: int, potentials: list[float]
    ):
        """Initialize Fiber class.

        :param diameter: fiber diameter [um]
        :param fiber_mode: name of fiber model type
        :param temperature: temperature of model [degrees celsius]
        :param n_fiber_coords: number of fiber coordinates
        :param potentials: list of membrane potentials [mV]
        """
        # TODO: need to think about making sure this is built so it is easy to add new fiber models
        self.potentials = []
        self.diameter = diameter
        self.fiber_model = fiber_mode
        self.temperature = temperature
        if fiber_mode not in ['MRG_DISCRETE', 'MRG_INTERPOLATION']:
            self.myelinated = False
        else:
            self.myelinated = True
        self.axonnodes = None
        self.delta_z = None
        self.passive_end_nodes = None
        self.sections = []
        self.n_aps = None
        self.v_init = None
        self.potentials = potentials
        assert len(potentials) == n_fiber_coords, 'Number of fiber coordinates does not match number of potentials'

        self._generate(n_fiber_coords)

        # TODO: add function for interpolating fiber coordinates/potentials.
        # TODO: make potentials at the fiber level,
        #  so you can run the same simulation class on multiple fibers, or many different simulation on same fiber

    def _generate(self, n_fiber_coords: int):
        """Build fiber model sections with NEURON.

        #TODO: split this into a unmyel and a myel function for separate subclasses of fiber superclass

        :param n_fiber_coords: number of fiber coordinates from COMSOL
        :return: Fiber object
        """
        # todo: need to save fiber coordinates
        fiber_parameters = FiberTypeParameters[self.fiber_model]
        # Determine geometrical parameters for fiber based on fiber model
        if self.fiber_model != 'MRG_DISCRETE' and self.fiber_model != 'MRG_INTERPOLATION':
            fiber_type = fiber_parameters['fiber_type']
            neuron_flag = fiber_parameters['neuron_flag']
            node_channels = fiber_parameters['node_channels']
            self.delta_z = fiber_parameters['delta_zs']
            self.passive_end_nodes = fiber_parameters['passive_end_nodes']
            channels_type = fiber_parameters['channels_type']

        elif self.fiber_model == 'MRG_DISCRETE':
            diameters, my_delta_zs, paranodal_length_2s, axon_diams, node_diams, para_diam_1, para_diam_2, nls = (
                fiber_parameters[key]
                for key in (
                    'diameters',
                    'delta_zs',
                    'paranodal_length_2s',
                    'axonDs',
                    'nodeDs',
                    'paraD1s',
                    'paraD2s',
                    'nls',
                )
            )
            diameter_index = diameters.index(self.diameter)
            neuron_flag = fiber_parameters['neuron_flag']
            node_channels = fiber_parameters['node_channels']
            fiber_type = fiber_parameters['fiber_type']
            paranodal_length_2 = fiber_parameters['paranodal_length_2s'][diameter_index]
            axon_diam = fiber_parameters['axonDs'][diameter_index]
            node_diam = fiber_parameters['nodeDs'][diameter_index]
            para_diam_1 = fiber_parameters['paraD1s'][diameter_index]
            para_diam_2 = fiber_parameters['paraD2s'][diameter_index]
            nl = fiber_parameters['nls'][diameter_index]
            self.delta_z = fiber_parameters['delta_zs'][diameter_index]
            self.passive_end_nodes = fiber_parameters['passive_end_nodes']

        elif self.fiber_model == 'MRG_INTERPOLATION':
            diameter = self.diameter
            neuron_flag = fiber_parameters['neuron_flag']
            node_channels = fiber_parameters['node_channels']
            fiber_type = fiber_parameters['fiber_type']
            self.passive_end_nodes = fiber_parameters['passive_end_nodes']
            if self.diameter >= 5.643:
                self.delta_z = -8.215 * diameter**2 + 272.4 * diameter - 780.2
            else:
                self.delta_z = 81.08 * diameter + 37.84
            paranodal_length_2 = -0.1652 * diameter**2 + 6.354 * diameter - 0.2862
            nl = -0.4749 * self.diameter**2 + 16.85 * self.diameter - 0.7648
            node_diam = 0.01093 * self.diameter**2 + 0.1008 * self.diameter + 1.099
            para_diam_1 = node_diam
            para_diam_2 = 0.02361 * self.diameter**2 + 0.3673 * self.diameter + 0.7122
            axon_diam = para_diam_2

        # Determine number of axonnodes
        if neuron_flag == 2:
            self.axonnodes = int(1 + (n_fiber_coords - 1) / 11)
        elif neuron_flag == 3:
            self.axonnodes = int(n_fiber_coords)
            length = self.delta_z * self.axonnodes

        # Determine starting voltage of system
        if fiber_type == 1:
            self.v_init = -88.3
        elif fiber_type == 2:
            self.v_init = -80
        elif fiber_type == 3:
            v_init_c_fibers = [
                -60,
                -55,
                -82,
                -48,
            ]  # v_rests for Sundt, Tigerholm, Rattay/Aberham, and Schild C-Fiber models, respectively
            self.v_init = v_init_c_fibers[channels_type - 1]

        # Create fiber sections
        if self.myelinated:
            self.create_myelinated_fiber(
                node_channels,
                self.axonnodes,
                self.diameter,
                axon_diam,
                node_diam,
                para_diam_1,
                para_diam_2,
                self.delta_z,
                paranodal_length_2,
                nl,
                self.passive_end_nodes,
            )
        elif not self.myelinated:
            self.create_unmyelinated_fiber(
                self.diameter,
                length,
                c_fiber_model_type=channels_type,
                celsius=self.temperature,
                delta_z=self.delta_z,
                passive_end_nodes=self.passive_end_nodes,
            )
        return self

    def create_myelinated_fiber(
        self,
        node_channels: bool,
        axonnodes: int,
        fiber_diam: float,
        axon_diam: float,
        node_diam: float,
        para_diam_1: float,
        para_diam_2: float,
        deltaz: float,
        paralength2: float,
        nl: int,
        passive_end_nodes: bool,
    ):
        """Create and connect NEURON sections for a myelinated fiber type.

        :param node_channels: true for Schild fiber models mechanisms, false otherwise
        :param axonnodes: number of node of Ranvier segments
        :param fiber_diam: fiber diameter [um]
        :param axon_diam: diameter of internodal fiber segment (STIN) [um]
        :param node_diam: diameter of node of Ranvier fiber segment [um]
        :param para_diam_1: diameter of myelin attachment section of fiber segment (MYSA) [um]
        :param para_diam_2: diameter of main section of paranode fiber segment (FLUT) [um]
        :param deltaz: node-node separation [um]
        :param paralength2: length of main section of paranode fiber segment (FLUT) [um]
        :param nl: number of myelin lemella
        :param passive_end_nodes: true for passive end node strategy, false otherwise
        :return: Fiber object
        """
        # Electrical parameters
        rhoa = 0.7e6  # [ohm-um]
        mycm = 0.1  # lamella membrane; [uF/cm2]
        mygm = 0.001  # lamella membrane; [S/cm2]

        if node_channels == 0:
            e_pas_vrest = -80
        elif node_channels == 1:
            e_pas_vrest = -57

        # Geometrical parameters [um]
        paranodes1 = 2 * (axonnodes - 1)  # Number of MYSA paranodes
        paranodes2 = 2 * (axonnodes - 1)  # Number of FLUT paranodes
        axoninter = 6 * (axonnodes - 1)  # Number of STIN internodes

        nodelength = 1.0  # Length of nodes of Ranvier [um]
        paralength1 = 3  # Length of MYSA [um]
        space_p1 = 0.002  # Thickness of periaxonal space in MYSA sections [um]
        space_p2 = 0.004  # Thickness of periaxonal space in FLUT sections [um]
        space_i = 0.004  # Thickness of periaxonal space in STIN sections [um]

        rpn0 = (rhoa * 0.01) / (math.pi * ((((node_diam / 2) + space_p1) ** 2) - ((node_diam / 2) ** 2)))
        rpn1 = (rhoa * 0.01) / (math.pi * ((((para_diam_1 / 2) + space_p1) ** 2) - ((para_diam_1 / 2) ** 2)))
        rpn2 = (rhoa * 0.01) / (math.pi * ((((para_diam_2 / 2) + space_p2) ** 2) - ((para_diam_2 / 2) ** 2)))
        rpx = (rhoa * 0.01) / (math.pi * ((((axon_diam / 2) + space_i) ** 2) - ((axon_diam / 2) ** 2)))
        interlength = (deltaz - nodelength - (2 * paralength1) - (2 * paralength2)) / 6

        # Create the axon sections
        node_ind, mysa_ind, stin_ind, flut_ind = 0, 0, 0, 0
        nsegments = axonnodes + paranodes1 + paranodes2 + axoninter
        for ind in range(1, nsegments + 1):
            if ind % 11 == 1:
                section = self.create_node(
                    node_ind,
                    node_diam,
                    nodelength,
                    rhoa,
                    mycm,
                    mygm,
                    passive_end_nodes,
                    axonnodes,
                    node_channels,
                    nl,
                    rpn0,
                )
                node_ind += 1
            elif ind % 11 == 2 or ind % 11 == 0:
                section = self.create_mysa(
                    mysa_ind, fiber_diam, paralength1, rhoa, para_diam_1, e_pas_vrest, rpn1, mycm, mygm, nl
                )
                mysa_ind += 1
            elif ind % 11 == 3 or ind % 11 == 10:
                section = self.create_flut(
                    flut_ind, fiber_diam, paralength2, rhoa, para_diam_2, e_pas_vrest, rpn2, mycm, mygm, nl
                )
                flut_ind += 1
            else:
                section = self.create_stin(
                    stin_ind, fiber_diam, interlength, rhoa, axon_diam, e_pas_vrest, rpx, mycm, mygm, nl
                )
                stin_ind += 1
            self.sections.append(section)

        # Connect the axon sections
        for i in range(0, nsegments - 1):
            self.sections[i + 1].connect(self.sections[i])

        return self

    def create_unmyelinated_fiber(
        self,
        fiber_diam: float = 6,
        length: float = 21,
        c_fiber_model_type: int = 1,
        celsius: float = 37,
        delta_z: float = 50 / 6,
        insert97na: bool = 0,
        conductances97: bool = 0,
        passive_end_nodes: bool = 0,
    ):
        """Create and connect NEURON sections for an unmyelinated fiber.

        :param fiber_diam: fiber diameter [um]
        :param length: fiber length [um]
        :param c_fiber_model_type: fiber model type (1=Sundt, 2=Tigerholm, 3=Rattay, 4=Schild94/Schild97)
        :param celsius: model temperature [celsius]
        :param delta_z: node-node separation [um]
        :param insert97na: controls sodium channel mechanisms. True if Schild97 fiber model, false otherwise
        :param conductances97: controls conductance density. True if Schild97 fiber model, false otherwise
        :param passive_end_nodes: true for passive end node strategy, false otherwise
        :return: instance of Fiber class
        """
        nsegments = int(length / delta_z)

        self.sections = []
        for i in range(0, nsegments):
            node = Section(name='node ' + str(i))
            self.sections.append(node)

            node.diam = fiber_diam
            node.nseg = 1
            node.L = delta_z
            if passive_end_nodes and (i == 0 or i == nsegments - 1):
                node.insert('pas')
                node.g_pas = 0.0001
                if c_fiber_model_type == 1:
                    node.e_pas = -60  # Sundt model equilibrium potential
                elif c_fiber_model_type == 2:
                    node.e_pas = -55  # Tigerholm model equilibrium potential
                elif c_fiber_model_type == 3:
                    node.e_pas = -70  # Rattay model equilibrium potential
                elif c_fiber_model_type == 4:
                    node.e_pas = -48  # Schild model equilibrium potential
                else:
                    node.e_pas = -70

                node.insert('extracellular')
                node.xc[0] = 0  # short circuit
                node.xg[0] = 1e10  # short circuit

                node.Ra = 1e10

            else:
                if c_fiber_model_type == 1:  # Sundt model
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
                    node.v = -60
                    node.e_pas = node.v + (node.ina + node.ik) / node.g_pas  # calculate leak equilibrium potential
                elif c_fiber_model_type == 2:  # Tigerholm model
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
                    node.v = -55
                elif c_fiber_model_type == 3:  # Rattay and Aberham model -- adjusted for a resting potential of -70mV
                    node.insert('RattayAberham')
                    node.Ra = 100  # required for propagation; less than 100 does not propagate
                    node.cm = 1
                    node.v = -70
                    node.ena = 45
                    node.ek = -82
                elif c_fiber_model_type == 4:  # Schild model
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
                    if insert97na:
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
                    if conductances97:
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
                    node.v = -48

                node.insert('extracellular')
                node.xc[0] = 0  # short circuit
                node.xg[0] = 1e10  # short circuit

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
        node_channels: float,
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
        :param node_channels: true for Schild fiber models mechanisms, false otherwise
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
            if node_channels == 0:
                node.cm = 2
                node.insert('axnode_myel')
            elif node_channels == 1:
                print('WARNING: Custom fiber models not yet implemented')
                pass

            node.insert('extracellular')
            node.xraxial[0] = rpn0
            node.xc[0] = 0  # short circuit
            node.xg[0] = 1e10  # short circuit

        return node
