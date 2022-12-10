"""Defines Recording class.

The copyrights of this software are owned by Duke University. Please
refer to the LICENSE and README.md files for licensing instructions. The
source code can be found on the following GitHub repository:
https://github.com/wmglab-duke/ascent
"""
import pandas as pd
from neuron import h

from src.wmglab_neuron import _Fiber

h.load_file('stdrun.hoc')


class Recording:
    """Manage recording parameters for NEURON simulations."""

    # TODO: either make this attach to an instance of stimulation or make all these methods of simulation
    # TODO: make extensible to allow user to save additional data?
    def __init__(self, fiber: _Fiber):
        """Initialize Recording class.

        :param fiber: instance of fiber class
        """
        self.save_vm = False
        self.save_gating = False
        self.save_istim = False
        self.vm = []

        self.gating_inds = list(range(0, fiber.nodecount))
        if fiber.passive_end_nodes:
            del self.gating_inds[0]
            del self.gating_inds[-1]

        self.gating_h = []
        self.gating_m = []
        self.gating_mp = []
        self.gating_s = []
        self.gating = [self.gating_h, self.gating_m, self.gating_mp, self.gating_s]

        self.istim = []

        self.apc = []

    def reset(self):
        """Reset recording attributes in order to be used for subsequent runs."""
        self.vm = []

        self.gating_h = []
        self.gating_m = []
        self.gating_mp = []
        self.gating_s = []
        self.gating = [self.gating_h, self.gating_m, self.gating_mp, self.gating_s]

        self.istim = []

        self.apc = []

    def record_ap(self, fiber: _Fiber, thresh: float = -30):
        """Create a list of NEURON APCount objects at all nodes along the axon.

        :param fiber: instance of fiber class
        :param thresh: the threshold value for Vm to pass for an AP to be detected [mV]
        """
        # TODO: add check for end excitation
        self.apc = [h.APCount(node(0.5)) for node in fiber.nodes]
        for apc in self.apc:
            apc.thresh = thresh

    def record_vm(self, fiber: _Fiber):
        """Record membrane voltage (mV) along the axon.

        :param fiber: instance of fiber class
        """
        self.vm = [h.Vector().record(node(0.5)._ref_v) for node in fiber.nodes]

    def record_istim(self, istim):  # todo: remove all "object" type hints
        """Record applied intracellular stimulation (nA).

        :param istim: instance of intracellular stimulation object
        """
        self.istim = h.Vector().record(istim._ref_i)

    def record_gating(self, fiber: _Fiber, fix_passive: bool = False):
        """Record gating parameters (h, m, mp, s) for myelinated fiber types.

        :param fiber: instance of fiber class
        :param fix_passive: true if fiber has passive end nodes, false otherwise
        """
        assert fiber.myelinated, "Cannot record gating parameters for unmyelinated _Fibers."
        # Set up recording vectors for h, m, mp, and s gating parameters all along the axon
        for node_ind in self.gating_inds:
            h_node = h.Vector().record(fiber.sections[node_ind * 11](0.5)._ref_h_inf_axnode_myel)
            m_node = h.Vector().record(fiber.sections[node_ind * 11](0.5)._ref_m_inf_axnode_myel)
            mp_node = h.Vector().record(fiber.sections[node_ind * 11](0.5)._ref_mp_inf_axnode_myel)
            s_node = h.Vector().record(fiber.sections[node_ind * 11](0.5)._ref_s_inf_axnode_myel)
            self.gating_h.append(h_node)
            self.gating_m.append(m_node)
            self.gating_mp.append(mp_node)
            self.gating_s.append(s_node)

    def ap_checker(
        self,
        fiber: _Fiber,
        ap_detect_location: float = 0.9,
    ) -> int:
        """Check to see if an action potential occurred at the end of a run.

        # remove this function and check in the respective functions

        :param fiber: instance of fiber class
        :param ap_detect_location: is the location (decimal % of fiber length) where APs are detected for threshold
        :return: number of action potentials that occurred
        """
        # Determine user-specified location along axon to check for action potential
        node_index = int((fiber.nodecount - 1) * ap_detect_location)
        return self.apc[node_index].n

    def threshold_checker(
        self,
        fiber: _Fiber,
        block: bool = False,
        ap_detect_location: float = 0.9,
        istim_delay: float = 0,
    ) -> int:
        """Check if stimulation was above or below threshold.

        :param fiber: instance of fiber class
        :param block: true if BLOCK_THRESHOLD protocol, false otherwise
        :param ap_detect_location: is the location (decimal % of fiber length) where APs are detected for threshold
        :param istim_delay: the delay from the simulation start to the onset of the intracellular stimulation [ms]
        :return: True if stim was supra-threshold, False if sub-threshold
        """
        # Determine user-specified location along axon to check for action potential
        node_index = int((fiber.nodecount - 1) * ap_detect_location)
        if block:
            return self.apc[node_index].time <= istim_delay
        else:
            return bool(self.apc[node_index].n)

    def get_variables(self, fiber: _Fiber):  # noqa: C901
        """Return recorded variables from a simulation.

        :param fiber: instance of fiber class
        :return: time, space, vm, gating, istim, apc
        """
        # Put all recorded data into pandas DataFrame
        # TODO: add column and index labels?
        vm_data = pd.DataFrame(self.vm)
        all_gating_data = {
            param: pd.DataFrame(gating_vector) for param, gating_vector in zip(['h', 'm', 'mp', 's'], self.gating)
        }
        istim_data = pd.DataFrame(self.istim)
        ap_loctime = [self.apc[loc_node_ind].time for loc_node_ind in range(0, fiber.nodecount)]
        ap_counts = [self.apc[loc_node_ind].n for loc_node_ind in range(0, fiber.nodecount)]
        return vm_data, all_gating_data, istim_data, ap_loctime, ap_counts

    def set_save(self, vm=False, gating=False, istim=False):
        """Set which variables to save.

        :param vm: true if membrane voltage should be saved, false otherwise
        :param gating: true if gating parameters should be saved, false otherwise
        :param istim: true if intracellular stimulation should be saved, false otherwise
        """
        self.save_vm = vm
        self.save_gating = gating
        self.save_istim = istim
