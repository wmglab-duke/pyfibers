"""Implementation of Schild 1994 and 1997 fiber models.

As described in:
1994 implementation: http://dx.doi.org/10.1152/jn.1994.71.6.2338
1997 implementation: http://dx.doi.org/10.1152/jn.1997.78.6.3198
Updated by: http://dx.doi.org/10.1152/jn.00315.2020
"""

from __future__ import annotations

import numpy as np
from neuron import h

from pyfibers.fiber import Fiber

h.load_file('stdrun.hoc')


class SchildFiber(Fiber):
    """Schild fiber model."""

    submodels = ['SCHILD97', 'SCHILD94']

    def __init__(self: SchildFiber, diameter: float, delta_z: float = 8.333, **kwargs) -> None:
        """Initialize SchildFiber class.

        :param diameter: fiber diameter [microns]
        :param delta_z: node spacing [microns]
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(diameter=diameter, **kwargs)
        self.myelinated = False
        self.gating_variables = {
            "d_cat": "d_cat",
            "f_cat": "f_cat",
            "d_can": "d_can",
            "fn1_can": "f1_can",
            "fn2_can": "f2_can",
            "m_naf": "m_naf",
            "h_naf": "h_naf",
            "j_naf": "l_naf",
            "m_nas": "m_nas",
            "h_nas": "h_nas",
            "n": "n_kd",
            "p": "p_ka",
            "q": "q_ka",
            "x": "x_kds",
            "y": "y1_kds",
            "c": "c_kca",
        }
        self.v_rest = -46.5  # [mV] Resting membrane potential
        self.delta_z = delta_z
        # update gating variables for Schild 1997 model
        if self.fiber_model.name == 'SCHILD97':
            self.gating_variables["m_naf"] = "m_naf97mean"
            self.gating_variables["m_nas"] = "m_nas97mean"
            self.gating_variables["h_naf"] = "h_naf97mean"
            self.gating_variables["h_nas"] = "h_nas97mean"
            self.gating_variables.pop("j_naf")

    def generate(self: SchildFiber, **kwargs) -> Fiber:  # noqa: D102
        return super().generate([self.create_schild], **kwargs)

    def create_schild(self: SchildFiber, ind: int, node_type: str) -> h.Section:
        """Create a SCHILD node.

        :param ind: node index in the fiber
        :param node_type: node type ('active' or 'passive')
        :return: created node with SCHILD mechanisms
        """
        R = 8314  # noqa: N806 # molar gas constant
        F = 96500  # noqa: N806 # faraday's constant
        node = self.nodebuilder(ind, node_type)
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
        if self.fiber_model.name == 'SCHILD97':
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
        node.ek = ((R * (self.temperature + 273.15)) / F) * np.log(
            node.ko / node.ki
        )  # Manual Calculation of ek in order to use Schild F and R values
        node.nao = 154  # [mM] External Na Concentration
        node.nai = 8.9  # [mM] Internal Na Concentration
        h.ion_style("na_ion", 1, 2, 0, 0, 0, sec=node)  # Allows ena to be calculated manually
        node.ena = ((R * (self.temperature + 273.15)) / F) * np.log(
            node.nao / node.nai
        )  # Manual Calculation of ena in order to use Schild F and R values
        if self.fiber_model.name == 'SCHILD97':
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

        node.L_caintscale = node.L_caextscale = self.delta_z  # length of node for calculating conductances
        node.nseg_caintscale = node.nseg_caextscale = 1  # number of node segments for calculating conductances

        # Adjustments to maintain identical function with Thio .mod modifications
        node.fhspace_caextscale = 1  # [um] Thickness of shell

        return node
