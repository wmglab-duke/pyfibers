"""Defines Recording class.

The copyrights of this software are owned by Duke University. Please
refer to the LICENSE and README.md files for licensing instructions. The
source code can be found on the following GitHub repository:
https://github.com/wmglab-duke/ascent
"""

from neuron import h

h.load_file('stdrun.hoc')


class Recording:
    """Manage recording parameters for NEURON simulations."""

    def __init__(self, fiber: object):
        """Initialize Recording class.

        :param fiber: instance of Fiber class
        """
        self.time = h.Vector().record(h._ref_t)
        self.space = list(range(0, fiber.axonnodes))
        self.vm = []

        self.gating_inds = list(range(0, fiber.axonnodes))
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
        self.ap_end_count = []
        self.ap_end_times = []

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
        self.ap_end_count = []
        self.ap_end_times = []

    def record_ap(self, fiber: object, thresh: float = -30):
        # TODO: consider merging with record_ap_end_times
        """Create a list of NEURON APCount objects at all nodes along the axon.

        :param fiber: instance of Fiber class
        :param thresh: the threshold value for Vm to pass for an AP to be detected [mV]
        """
        for i in range(0, fiber.axonnodes):
            if fiber.myelination:
                ind = i * 11
            else:
                ind = i
            self.apc.append(h.APCount(fiber.sections[ind](0.5)))
            self.apc[i].thresh = thresh

    def record_ap_end_times(self, fiber: object, ap_end_inds: list, ap_end_thresh: float):
        """Record when action potential occurs at specified indices. For 'end_ap_times' in sim.json.

        :param fiber: instance of Fiber class
        :param ap_end_inds: list of user-specified indices to record APs
        :param ap_end_thresh: threshold value for action potentials
        """
        # Create vectors to save ap times to
        self.ap_end_times = [h.Vector(), h.Vector()]
        for ap_end_vector, ap_end_ind in zip(self.ap_end_times, ap_end_inds):
            if fiber.myelination:
                # if myelinated, create APCount at node of Ranvier
                ap_count = h.APCount(fiber.sections[ap_end_ind * 11](0.5))
            else:
                # if unmyelinated, create APCount at axon segment
                ap_count = h.APCount(fiber.sections[ap_end_ind](0.5))
            ap_count.thresh = ap_end_thresh
            ap_count.record(ap_end_vector)  # save AP times detected by APCount to vector
            self.ap_end_count.append(ap_count)

    def record_vm(self, fiber):
        """Record membrane voltage (mV) along the axon.

        :param fiber: instance of Fiber class
        """
        for ind in range(0, fiber.axonnodes):
            if fiber.myelination:
                v_node = h.Vector().record(fiber.sections[ind * 11](0.5)._ref_v)
            else:
                v_node = h.Vector().record(fiber.sections[ind](0.5)._ref_v)
            self.vm.append(v_node)
        return

    def record_istim(self, istim: object):
        """Record applied intracellular stimulation (nA).

        :param istim: instance of intracellular stimulation object
        """
        self.istim = h.Vector().record(istim._ref_i)

    def record_gating(self, fiber: object, fix_passive: bool = False):
        """Record gating parameters (h, m, mp, s) for myelinated fiber types.

        :param fiber: instance of Fiber class
        :param fix_passive: true if fiber has passive end nodes, false otherwise
        """
        if fix_passive is False:
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

        # If fiber has passive end nodes, then insert vectors of 0's to output
        elif fix_passive and fiber.passive_end_nodes:
            for gating_vectors in self.gating:
                size = gating_vectors[0].size()
                passive_node = h.Vector(size, 0)
                gating_vectors.insert(0, passive_node)
                gating_vectors.append(passive_node)

    def ap_checker(
        self,
        fiber: object,
        find_block_thresh: bool = False,
        ap_detect_location: float = 0.9,
        istim_delay: float = 0,
    ) -> int:
        """Check to see if an action potential occurred at the end of a run.

        :param fiber: instance of Fiber class
        :param find_block_thresh: true if BLOCK_THRESHOLD protocol, false otherwise
        :param ap_detect_location: is the location (decimal % of fiber length) where APs are detected for threshold
        :param istim_delay: the delay from the simulation start to the onset of the intracellular stimulation [ms]
        :return: number of action potentials that occurred
        """
        # Determine user-specified location along axon to check for action potential
        node_index = int((fiber.axonnodes - 1) * ap_detect_location)

        if find_block_thresh:
            if self.apc[node_index].time > istim_delay:
                n_aps = 0  # False - block did not occur
            else:
                n_aps = 1  # True - block did occur
        else:
            n_aps = self.apc[node_index].n
        return n_aps
