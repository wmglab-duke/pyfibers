"""The copyrights of this software are owned by Duke University."""

from __future__ import annotations

import warnings

import numpy as np
from neuron import h

from pyfibers.fiber import Fiber

h.load_file("stdrun.hoc")


class ThioFiber(Fiber):
    """Thio fiber model."""

    submodels = ['THIO_AUTONOMIC', 'THIO_CUTANEOUS']

    # Parameters later assigned to NEURON machanisms
    thio_params = {
        'THIO_AUTONOMIC': [
            0.036813,
            0.075747,
            0.000376,
            0.000156,
            0.000004,
            0.009546,
            0.002864,
            0.002789,
            0.005337,
            0.000289,
            0.000024,
            0.000006,
            0.000210,
            0.056316,
            23.117539,
        ],
        'THIO_CUTANEOUS': [
            0.035663,
            0.115643,
            0.000504,
            0.002016,
            0.000188,
            0.000361,
            0.000003,
            0.000106,
            0.327196,
            0.001786,
            0.000044,
            0.000755,
            0.009242,
            0.000456,
            27.513088,
        ],
    }

    def __init__(self: ThioFiber, diameter: float, delta_z: float = 8.333, **kwargs) -> None:
        """Initialize fiber class.

        :param diameter: fiber diameter [microns]
        :param delta_z: node spacing [microns]
        :param kwargs: keyword arguments to pass to the base class
        """
        super().__init__(diameter=diameter, **kwargs)
        self.myelinated = False
        self.delta_z = delta_z  # microns
        self.v_rest = -58.5  # mV

        # pulled from state variable/name of mechanism
        self.gating_variables = {
            "m1.7": "m_nav7",
            "h1.7": "h_nav7",
            "h1.8": "h_newnav8",
            "m1.8": "m_newnav8",
            "s1.8": "s_newnav8",
            "h1.9": "h_nav9",
            "m1.9": "m_nav9",
            "s1.9": "s_nav9",
            "m_bk": "m_bk",
            "h_bk": "h_bk",
            "m_cav12": "m_cav12",  # Titus - L-Type Voltage Dependent Calcium Channel
            "h_cav12": "h_cav12",
            "m_cav22": "m_cav22",  # Titus - N-Type Voltage Dependent Calcium Channel
            "h_cav22": "h_cav22",
            "s_cav22": "s_cav22",
            "n_km": "n_km",
            "m_km": "m_km",
            "m_hcn": "m_hcn",
            "n_hcn": "n_hcn",
            "h_kv21": "h_kv21",
            "m_kv21": "m_kv21",
            "h_ka34": "h_ka34",
            "m_ka34": "m_ka34",
            "h_ka14": "h_ka14",
            "s_ka14": "s_ka14",
            "m_ka14": "m_ka14",
            "n_sk": "n_sk",
        }

        if self.passive_end_nodes:
            warnings.warn("Ignoring passive_end_nodes for Thio fiber", UserWarning, stacklevel=2)
            self.passive_end_nodes: bool = False

    def generate(self: ThioFiber, **kwargs) -> Fiber:  # noqa: D102
        return super().generate([self.create_thio], **kwargs)

    def create_thio(self: ThioFiber, ind: int, node_type: str) -> None:
        """Create a THIO node.

        :param ind: node index in the fiber
        :param node_type: node type ('active' or 'passive')
        :returns: NEURON section with THIO node mechanisms
        """
        node = self.nodebuilder(ind, node_type)

        R = 8314  # noqa: N806 # molar gas constant
        F = 96485.3329  # noqa: N806 # faraday's constant
        node.insert("nav7")
        node.insert("newnav8")
        node.insert("nav9")
        node.insert("bk")
        node.insert("cav12")
        node.insert("cav22")
        node.insert("caextscale")
        node.insert("caintscale")
        node.insert("km")
        node.insert("hcn")
        node.insert("kv21")
        node.insert("ka34")
        node.insert("ka14")
        node.insert("sk")
        node.insert("nacx")
        node.insert("naoi")
        node.insert("koi")
        node.insert("NaKpumpSchild")
        node.insert("leak")
        node.insert("extrapump")

        # Ionic concentrations
        h.cao0_ca_ion = 2.0  # [mM] Initial Cao Concentration
        h.cai0_ca_ion = 0.000117  # [mM] Initial Cai Concentrations
        node.ko = 5.4  # [mM] External K Concentration
        node.ki = 145.0  # [mM] Internal K Concentration
        # CHECK: kstyle=ion_style("k_ion",3,2,1,1,0) 		// Allows ek to be calculated manually
        h.ion_style("k_ion", 3, 2, 1, 1, 0, sec=node)  # Allows ek to be calculated manually
        node.ek = ((R * (self.temperature + 273.15)) / F) * np.log(
            node.ko / node.ki
        )  # Manual Calculation of ek in order to use Thio F and R values
        node.nao = 154  # [mM] External Na Concentration
        node.nai = 8.9  # [mM] Internal Na Concentration
        # CHECK: nastyle=ion_style("na_ion",3,2,1,1,0) 		// Allows ena to be calculated manually
        h.ion_style("na_ion", 3, 2, 1, 1, 0, sec=node)  # Allows ena to be calculated manually
        node.ena = ((R * (self.temperature + 273.15)) / F) * np.log(
            node.nao / node.nai
        )  # Manual Calculation of ena in order to use Thio F and R values

        node.cm = 1.326291192

        (
            node.gbar_nav7,
            node.gbar_newnav8,
            node.gbar_nav9,
            node.gbar_bk,
            node.gbar_cav12,
            node.gbar_cav22,
            node.gbar_km,
            node.gbar_hcn,
            node.gbar_kv21,
            node.gbar_ka34,
            node.gbar_ka14,
            node.gbar_sk,
            node.gbar_nacx,
            node.INaKmax22_NaKpumpSchild,
            node.Ra,
        ) = self.thio_params[self.fiber_model.name]

        node.L_caintscale = node.L_caextscale = self.delta_z
        node.nseg_caintscale = node.nseg_caextscale = 1

        return node

    def balance(self: ThioFiber) -> None:
        """Balance membrane currents."""
        v_rest = self.v_rest
        for s in self.sections:
            if (-(s.ina) / (v_rest - s.ena)) < 0:
                s.pumpina_extrapump = -(s.ina)
                s.gnaleak_leak = 0
            else:
                s.gnaleak_leak = -(s.ina) / (v_rest - s.ena)
                s.pumpina_extrapump = 0

            if (-(s.ik) / (v_rest - s.ek)) < 0:
                s.pumpik_extrapump = -(s.ik)
                s.gkleak_leak = 0
            else:
                s.gkleak_leak = -(s.ik) / (v_rest - s.ek)
                s.pumpik_extrapump = 0

            if (-(s.ica) / (v_rest - s.eca)) < 0:
                s.pumpica_extrapump = -(s.ica)
            else:
                s.gcaleak_leak = -(s.ica) / (v_rest - s.eca)
