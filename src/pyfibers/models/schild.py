"""The copyrights of this software are owned by Duke University."""

from __future__ import annotations

import numpy as np
from neuron import h

from pyfibers.fiber import Fiber, _HomogeneousFiber

h.load_file('stdrun.hoc')


class SchildFiber(_HomogeneousFiber):
    """Schild fiber model."""

    submodels = ['SCHILD97', 'SCHILD94']

    def __init__(self: SchildFiber, diameter: float, **kwargs) -> None:
        """Initialize SchildFiber class.

        :param diameter: fiber diameter [microns]
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(diameter=diameter, **kwargs)
        self.myelinated = False
        self.delta_z = 8.333  # microns
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
        self.v_rest = -48
        # update gating variables for Schild 1997 model
        if self.fiber_model.name == 'SCHILD97':
            self.gating_variables["m_naf"] = "m_naf97mean"
            self.gating_variables["m_nas"] = "m_nas97mean"
            self.gating_variables["h_naf"] = "h_naf97mean"
            self.gating_variables["h_nas"] = "h_nas97mean"
            self.gating_variables.pop("j_naf")

    def generate(self, n_sections: int, length: float) -> Fiber:  # noqa D102
        return self.generate_homogeneous(
            n_sections,
            length,
            self.create_schild,
            model_type=self.fiber_model.name,
            celsius=self.temperature,
            ca_l=self.delta_z,
            ca_nseg=1,
        )

    @staticmethod
    def create_schild(
        node: h.Section, celsius: float, model_type: str = 'SCHILD94', ca_l: float = 8.3333, ca_nseg: int = 1
    ) -> None:
        """Create a SCHILD node.

        :param celsius: model temperature [celsius]
        :param node: NEURON section
        :param model_type: model type string, either 'SCHILD97' or 'SCHILD94'
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
        if model_type == 'SCHILD97':
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
        if model_type == 'SCHILD97':
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
