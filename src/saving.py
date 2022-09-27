"""Defines Saving class.

The copyrights of this software are owned by Duke University. Please
refer to the LICENSE and README.md files for licensing instructions. The
source code can be found on the following GitHub repository:
https://github.com/wmglab-duke/ascent
"""


import os

import pandas as pd


class Saving:
    """Manage saving parameters to file for NEURON simulations."""

    def __init__(self):
        """Initialize Saving class."""

        self.space_vm = None
        self.space_gating = None
        self.time_inds = []
        self.time_vm = None
        self.time_gating = None
        self.istim = None
        self.locs = []
        self.node_inds = []
        self.ap_end_times = None
        self.ap_end_inds = []
        self.ap_end_thresh = None
        self.runtime = None
        self.output_path = None
        return

    def inherit(self,
                sim_path: str,
                dt: float,
                fiber: object,
                space_vm: bool = False,
                space_gating: bool = False,
                space_times: list[float] = [],
                time_vm: bool = False,
                time_gating: bool = False,
                istim: bool = False,
                locs: list[float] = [],
                end_ap_times: bool = False,
                loc_min: float = 0.1,
                loc_max: float = 0.9,
                ap_end_thresh: float = -30,
                ap_loctime: bool = False,
                runtime: bool = False,
                ):
        """Assign values to all Saving instance attributes.

        :param sim_path: path to n_sim directory
        :param dt: user-specified time step for simulation
        :param fiber: instance of Fiber class
        :param space_vm: save transmembrane at all sections at the time stamps defined in 'space_times'
        :param space_gating: save channel gating parameters at all sections at the time stamps defined in 'space_times'
        :param space_times: times in the simulation at which to save the values of state variables [ms]
        :param time_vm: save the transmembrane potential at each time step at the locations defined in 'locs'
        :param time_gating: save the channel gating parameters at each time step at the locations defined in 'locs'
        :param istim: save the applied intracellular stimulation at each time step
        :param locs: locations (decimal percentages of fiber length) at which to save state variables at all time steps
        :param end_ap_times: record when action potential occurs at specified indices
        :param loc_min: if end_ap_times, decimal % of fiber length at which to save times when AP is triggered
        :param loc_max: if end_ap_times, decimal % of fiber length at which to save times when Vm returns to threshold
        :param ap_end_thresh: if end_ap_times, the threshold value for Vm to pass for an AP to be detected [mV]
        :param ap_loctime: save, for each fiber node, the last time an AP passed over that node
        :param runtime: save the simulation runtime
        :return: Saving object
        """
        self.space_vm = space_vm
        self.space_gating = space_gating
        sim_times = space_times
        self.time_inds = [int(t / dt) for t in sim_times]  # divide by dt to avoid indexing error
        self.time_inds.sort()
        self.time_vm = time_vm
        self.time_gating = time_gating
        self.istim = istim
        self.locs = locs
        if self.locs != 'all':
            self.node_inds = [int((fiber.axonnodes - 1) * loc) for loc in self.locs]
        elif self.locs == 'all':
            self.node_inds = list(range(0, fiber.axonnodes))
        self.node_inds.sort()
        if end_ap_times:
            self.ap_end_times = True
            node_ind_min = int((fiber.axonnodes - 1) * loc_min)
            node_ind_max = int((fiber.axonnodes - 1) * loc_max)
            self.ap_end_inds = [node_ind_min, node_ind_max]
            self.ap_end_thresh = ap_end_thresh
        self.ap_loctime = ap_loctime
        self.runtime = runtime
        self.output_path = os.path.join(sim_path, 'data', 'outputs')
        return

    def save_thresh(self, fiber: object, thresh: float):
        """Save threshold from NEURON simulation to file.

        :param fiber: instance of Fiber class
        :param thresh: activation threshold from NEURON simulation
        """
        # save threshold to submit/n_sims/#/data/outputs/thresh_inner#_fiber#.dat
        thresh_path = os.path.join(self.output_path, f'thresh_inner{fiber.inner_ind}_fiber{fiber.fiber_ind}.dat')
        with open(thresh_path, 'w') as thresh_file:
            thresh_file.write(f"{thresh:.6f} mA")

    def save_runtime(self, fiber: object, runtime: float, amp_ind: int = 0):
        """Save NEURON simulation runtime to file.

        :param fiber: instance of Fiber class
        :param runtime: runtime of NEURON simulation
        :param amp_ind: index of amplitude in list of amplitudes for finite_amplitude protocol
        """
        if self.runtime:
            # save runtime to submit/n_sims/#/data/outputs/runtime_inner#_fiber#_amp#.dat
            runtimes_path = os.path.join(
                self.output_path, f'runtime_inner{fiber.inner_ind}_fiber{fiber.fiber_ind}_amp{amp_ind}.dat'
            )
            with open(runtimes_path, 'w') as runtime_file:
                runtime_file.write(f'{runtime:.3f}s')

    def save_activation(self, fiber, amp_ind):
        """Save the number of action potentials that occurred at the location specified in sim config to file.

        :param fiber: instance of Fiber class
        :param amp_ind: index of amplitude in list of amplitudes for finite_amplitude protocol
        """
        # save number of action potentials to submit/n_sims/#/data/outputs/activation_inner#_fiber#_amp#.dat
        output_file_path = os.path.join(
            self.output_path, f'activation_inner{fiber.inner_ind}_fiber{fiber.fiber_ind}_amp{amp_ind}.dat'
        )
        with open(output_file_path, 'w') as activation_file:
            activation_file.write(f'{fiber.n_aps:.3f}')

    def save_variables(self, fiber: object, recording: object, dt: float, amp_ind: int = 0):  # noqa: C901
        # TODO: reduce complexity
        """Write user-specified variables to file.

        :param fiber: instance of Fiber class
        :param recording: instance of Recording class
        :param dt: user-specified time step for simulation
        :param amp_ind: index of amplitude if protocol is FINITE_AMPLITUDES
        """

        def create_header(save_type: str, var_type: str, units: str = None):
            """Create a header for text file.

            :param save_type: function of variable to be saved, can be function of time ('time') or space ('space')
            :param var_type: type of variable to be saved (Vm, h, mp, m, s)
            :param units: units of variable (mV, nA)
            :return: list of column headers
            """
            header = []
            if save_type == 'space':  # F(x) - function of space
                header.append('Node#')
                for time in self.time_inds:
                    if units is not None:
                        header.append(f'{var_type}_time{int(time * dt)}ms({units})')
                    else:
                        header.append(f'{var_type}_time{int(time * dt)}ms')
            elif save_type == 'time':  # F(t) - function of time
                header.append('Time(ms)')
                if var_type != 'istim':
                    for node in self.node_inds:
                        if units is not None:
                            header.append(f'{var_type}_node{node + 1}({units})')
                        else:
                            header.append(f'{var_type}_node{node + 1}')
                elif var_type == 'istim':
                    header.append(f'Istim({units})')
            return header

        # Put all recorded data into pandas DataFrame
        vm_data = pd.DataFrame(recording.vm)
        all_gating_data = [pd.DataFrame(gating_vector) for gating_vector in recording.gating]
        istim_data = pd.DataFrame(recording.istim)

        if self.space_vm:
            vm_space_path = os.path.join(
                self.output_path, f'Vm_space_inner{fiber.inner_ind}_fiber{fiber.fiber_ind}_amp{amp_ind}.dat'
            )
            vm_space_data = vm_data[self.time_inds]  # save data only at user-specified times
            vm_space_data.insert(0, 'Node#', recording.space)
            vm_space_header = create_header('space', 'vm', 'mV')
            vm_space_data.to_csv(vm_space_path, header=vm_space_header, sep='\t', float_format='%.6f', index=False)

        if self.space_gating:
            gating_params = ['h', 'm', 'mp', 's']
            for gating_param, gating_data in zip(gating_params, all_gating_data):
                gating_space_path = os.path.join(
                    self.output_path,
                    f'gating_{gating_param}_space_inner{fiber.inner_ind}_fiber{fiber.fiber_ind}_amp{amp_ind}.dat',
                )
                gating_space_data = gating_data[self.time_inds]  # save data only at user-specified times
                gating_space_data.insert(0, 'Node#', recording.space)
                gating_space_header = create_header('space', gating_param)
                gating_space_data.to_csv(
                    gating_space_path, header=gating_space_header, sep='\t', float_format='%.6f', index=False
                )

        if self.time_vm:
            vm_time_path = os.path.join(
                self.output_path, f'Vm_time_inner{fiber.inner_ind}_fiber{fiber.fiber_ind}_amp{amp_ind}.dat'
            )
            vm_time_data = vm_data.T[self.node_inds]  # save data only at user-specified locations
            vm_time_data.insert(0, 'Time', recording.time)
            vm_time_header = create_header('time', 'vm', 'mV')
            vm_time_data.to_csv(vm_time_path, header=vm_time_header, sep='\t', float_format='%.6f', index=False)

        if self.time_gating:
            gating_params = ['h', 'm', 'mp', 's']
            for gating_param, gating_data in zip(gating_params, all_gating_data):
                gating_time_path = os.path.join(
                    self.output_path,
                    f'gating_{gating_param}_time_inner{fiber.inner_ind}_fiber{fiber.fiber_ind}_amp{amp_ind}.dat',
                )
                gating_time_data = gating_data.T[self.node_inds]  # save data only at user-specified locations
                gating_time_data.insert(0, 'Time', recording.time)
                gating_time_header = create_header('time', gating_param)
                gating_time_data.to_csv(
                    gating_time_path, header=gating_time_header, sep='\t', float_format='%.6f', index=False
                )

        if self.istim:
            istim_path = os.path.join(
                self.output_path, f'Istim_inner{fiber.inner_ind}_fiber{fiber.fiber_ind}_amp{amp_ind}.dat'
            )
            istim_data.insert(0, 'Time', recording.time)
            istim_header = create_header('time', 'istim', 'nA')
            istim_data.to_csv(istim_path, header=istim_header, sep='\t', float_format='%.6f', index=False)

        if self.ap_end_times:
            ap_end_times_path = os.path.join(
                self.output_path, f'Aptimes_inner{fiber.inner_ind}_fiber{fiber.fiber_ind}_amp{amp_ind}.dat'
            )
            with open(ap_end_times_path, 'w') as ap_end_file:
                ap_end_file.write(f'Node{self.ap_end_inds[0] + 1} \t Node{self.ap_end_inds[1] + 1} \n')

                min_size, max_size = recording.ap_end_times[0].size(), recording.ap_end_times[1].size()
                n_rows = max(min_size, max_size)
                for i in range(0, n_rows):
                    if i < min_size and i < max_size:
                        ap_end_file.write(
                            f'{recording.ap_end_times[0][i]:.6f} \t {recording.ap_end_times[1][i]:.6f} \n'
                        )
                    elif min_size > i >= max_size:
                        ap_end_file.write(f'{recording.ap_end_times[0][i]:.6f} \t  nan \n')
                    elif min_size <= i < max_size:
                        ap_end_file.write(f'nan \t {recording.ap_end_times[1][i]:.6f} \n')

        if self.ap_loctime:
            ap_loctime_path = os.path.join(
                self.output_path, f'ap_loctime_inner{fiber.inner_ind}_fiber{fiber.fiber_ind}_amp{amp_ind}.dat'
            )
            with open(ap_loctime_path, 'w') as ap_loctime_file:
                for loc_node_ind in range(0, fiber.axonnodes):
                    ap_loctime_file.write(f'{recording.apc[loc_node_ind].time}\n')
