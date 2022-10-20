"""Defines Stimulation class.

The copyrights of this software are owned by Duke University. Please
refer to the LICENSE and README.md files for licensing instructions. The
source code can be found on the following GitHub repository:
https://github.com/wmglab-duke/ascent
"""
import time

from neuron import h

from src.wmglab_neuron import Fiber, Recording, Saving

h.load_file('stdrun.hoc')


class Stimulation:
    """Manage stimulation of NEURON simulations."""

    def __init__(
        self, fiber: Fiber, potentials_list: list[float], waveform_list: list[int], dt: float = 0.001, tstop: float = 50
    ):
        """Initialize Stimulation class.

        :param fiber: instance of Fiber class
        :param potentials_list: list of extracellular potentials Ve(x)
        :param waveform_list: list of amplitudes at each time step of the simulation
        :param dt: time step for simulation [seconds]
        :param tstop: time step for simulation [seconds]
        """
        self.fiber = fiber
        self.potentials = potentials_list
        self.waveform = waveform_list
        self.dt = dt
        self.tstop = tstop
        self.istim = None

    def apply_intracellular(
        self,
        delay: float = 0,
        pw: float = 0,
        dur: float = 0,
        freq: float = 0,
        amp: float = 0,
        ind: int = 0,
    ):
        """Create instance of trainIClamp for intracellular stimulation.

        :param delay: the delay from the start of the simulation to the onset of the intracellular stimulation [ms]
        :param pw: the pulse duration of the intracellular stimulation [ms]
        :param dur: the duration from the start of the simulation to the end of the intracellular stimulation [ms]
        :param freq: the intracellular stimulation frequency [Hz]
        :param amp: the intracellular stimulation amplitude [nA]
        :param ind: the section index (unmyelinated) or node of Ranvier number (myelinated) receiving stimulation

        :return: instance of Stimulation class
        """
        if self.fiber.myelinated:
            intrastim_pulsetrain_ind = ind * 11
        else:
            intrastim_pulsetrain_ind = ind
        intracellular_stim = h.trainIClamp(self.fiber.sections[intrastim_pulsetrain_ind](0.5))
        intracellular_stim.delay = delay
        intracellular_stim.PW = pw
        intracellular_stim.train = dur
        intracellular_stim.freq = freq
        intracellular_stim.amp = amp
        self.istim = intracellular_stim
        return self

    def initialize_extracellular(self):
        """Set extracellular stimulation values to zero along entire fiber."""
        for section in self.fiber.sections:
            section(0.5).e_extracellular = 0

    def update_extracellular(self, e_stims: list[float]):
        """Update the applied extracellular stimulation all along the fiber length.

        :param e_stims: list of extracellular stimulations to apply along fiber length
        """
        for x, section in enumerate(self.fiber.sections):
            section(0.5).e_extracellular = e_stims[x]

    def finite_amplitudes(
        self,
        saving: Saving,
        recording: Recording,
        start_time: float,
        amps: list,
        t_init_ss: float = -200,
        dt_init_ss: float = 10,
    ):
        """Submit runs for FINITE_AMPLITUDES protocol.

        :param saving: instance of Saving class
        :param recording: instance of Recording class
        :param start_time: time at the very beginning of simulation
        :param amps: list of amplitudes to simulate
        :param t_init_ss: the time (<=0ms) for the system to reach steady state before starting the simulation [ms]
        :param dt_init_ss: the time step used to reach steady state [ms]
        """
        time_total = 0
        amps = amps if amps is not None else []
        for amp_ind, amp in enumerate(amps):
            print(f'Running amp {amp_ind} of {len(amps)}: {amp} mA')

            self.run_sim(amp, recording, saving=saving, t_init_ss=t_init_ss, dt_init_ss=dt_init_ss)
            time_individual = time.time() - start_time - time_total
            # todo: remove saving class
            # saving.save_variables(selfy, recording, stimulation.dt, amp_ind)  # Save user-specified variables
            # saving.save_activation(selfy, amp_ind)  # Save number of APs triggered
            # saving.save_runtime(selfy, time_individual, amp_ind)  # Save runtime of inidividual run

            time_total += time_individual
            recording.reset()  # Reset recording vectors to be used again

    def find_threshold(  # noqa: C901
        self,
        saving: Saving,
        recording: Recording,
        find_block_thresh: bool = False,
        bounds_search_mode: str = 'PERCENT_INCREMENT',
        step: float = 10,
        termination_mode: str = 'PERCENT_DIFFERENCE',
        termination_percent: float = 1,
        termination_tolerance: float = 1,  # todo: come up with default value for this
        stimamp_top: float = -1,
        stimamp_bottom: float = -0.01,
        t_init_ss: float = -200,
        dt_init_ss: float = 10,
    ):
        """Binary search to find threshold amplitudes.

        :param saving: instance of Saving class
        :param recording: instance of Recording class
        :param find_block_thresh: true if BLOCK_THRESHOLD protocol, false otherwise
        :param bounds_search_mode: indicates how to change upper and lower bounds for the binary search
        :param step: the incremental increase/decrease of the upper/lower bound in the binary search
        :param termination_mode: indicates when upper and lower bounds converge on a solution of appropriate precision
        :param termination_percent: percent difference between upper/lower bounds for finding threshold (e.g., 1 is 1%)
        :param termination_tolerance: the absolute difference between upper/lower bounds for finding threshold [mA]
        :param stimamp_top: the upper-bound stimulation amplitude first tested in a binary search for thresholds
        :param stimamp_bottom: the lower-bound stimulation amplitude first tested in a binary search for thresholds
        :param t_init_ss: the time (<=0ms) for the system to reach steady state before starting the simulation [ms]
        :param dt_init_ss: the time step used to reach steady state [ms]
        """
        # Determine searching parameters for binary search bounds
        if (
            bounds_search_mode == 'PERCENT_INCREMENT'
        ):  # relative increment (increase bound by a certain percentage of the previous value)
            increment_flag = 1  # flag is true for PERCENT_INCREMENT
            rel_increment = round(step / 100, 4)
        elif (
            bounds_search_mode == 'ABSOLUTE_INCREMENT'
        ):  # absolute increment (increase bound by a a certain amount + previous value)
            increment_flag = 0  # flag is false for ABSOLUTE_INCREMENT
            abs_increment = round(step, 4)

        if termination_mode == 'ABSOLUTE_DIFFERENCE':
            termination_flag = 1
            abs_thresh_resoln = round(termination_tolerance, 4)
        elif termination_mode == 'PERCENT_DIFFERENCE':
            termination_flag = 0
            rel_thresh_resoln = round(termination_percent / 100, 4)

        check_top_flag = 0  # 0 for upper-bound not yet found, value changes to 1 when the upper-bound is found
        check_bottom_flag = 0  # 0 for lower-bound not yet found, value changes to 1 when the lower-bound is found
        # enter binary search when both are found

        # Determine upper- and lower-bounds for simulation
        iterations = 1
        while True:
            if check_top_flag == 0:
                # Check to see if upper-bound triggers action potential
                print(f'Running stimamp_top = {stimamp_top:.6f}')
                self.run_sim(
                    stimamp_top,
                    recording,
                    find_block_thresh=find_block_thresh,
                    t_init_ss=t_init_ss,
                    dt_init_ss=dt_init_ss,
                )

                if self.n_aps == 0:
                    if find_block_thresh == 0:
                        print(
                            'ERROR: Initial stimamp_top value does not elicit an AP - '
                            'need to increase its magnitude and/or increase tstop to detect evoked AP'
                        )
                    else:
                        print(
                            'WARNING: Initial stimamp_top value does not block - '
                            'need to increase its magnitude and/or increase tstop to block test pulse evoked AP'
                        )
                    if increment_flag == 0:
                        stimamp_top = stimamp_top + abs_increment
                    elif increment_flag == 1:
                        stimamp_top = stimamp_top * (1 + rel_increment)
                else:
                    check_top_flag = 1

            if check_bottom_flag == 0:
                # Check to see if lower-bound does not trigger action potential
                print(f'Running stimamp_bottom = {stimamp_bottom:.6f}')
                self.run_sim(
                    stimamp_bottom,
                    recording,
                    find_block_thresh=find_block_thresh,
                    t_init_ss=t_init_ss,
                    dt_init_ss=dt_init_ss,
                )

                if self.n_aps != 0:
                    if find_block_thresh == 0:
                        print(
                            'ERROR: Initial stimamp_bottom value elicits an AP - '
                            'need to decrease its magnitude and/or increase tstop to detect block test pulses'
                        )
                    else:
                        print(
                            'WARNING: Initial stimamp_bottom value blocks - '
                            'need to decrease its magnitude and/or increase tstop to detect test pulse evoked AP'
                        )
                    if increment_flag == 0:
                        stimamp_bottom = stimamp_bottom - abs_increment
                    elif increment_flag == 1:
                        stimamp_bottom = stimamp_bottom * (1 - rel_increment)
                else:
                    check_bottom_flag = 1

            if check_bottom_flag == 1 and check_top_flag == 1:
                print('Bounds set - entering binary search')
                break

            iterations += 1

            if iterations >= 100:
                print('maximum number of bounds searching steps reached. breaking.')
                quit()

        # Enter binary search
        while True:
            stimamp_prev = stimamp_top

            stimamp = (stimamp_bottom + stimamp_top) / 2
            print(f'stimamp_bottom = {stimamp_bottom:.6f}      stimamp_top = {stimamp_top:.6f}')
            print(f'Running stimamp: {stimamp:.6f}')
            self.run_sim(
                stimamp,
                recording,
                find_block_thresh=find_block_thresh,
                t_init_ss=t_init_ss,
                dt_init_ss=dt_init_ss,
            )

            if termination_flag == 0:
                thresh_resoln = abs(rel_thresh_resoln)
                tolerance = abs((stimamp_bottom - stimamp_top) / stimamp_top)
            elif termination_flag == 1:
                thresh_resoln = abs(abs_thresh_resoln)
                tolerance = abs(stimamp_bottom - stimamp_top)

            # Check to see if stimamp is at threshold
            if tolerance < thresh_resoln:
                if self.n_aps < 1:
                    stimamp = stimamp_prev
                print(f'Done searching! stimamp: {stimamp:.6f} mA for extracellular\n')

                # Run one more time at threshold to save user-specified variables
                self.run_sim(
                    stimamp,
                    recording,
                    find_block_thresh=find_block_thresh,
                    saving=saving,
                    t_init_ss=t_init_ss,
                    dt_init_ss=dt_init_ss,
                )
                # todo: remove saving class
                # saving.save_thresh(selfy, stimamp)  # Save threshold value to file
                break
            elif self.n_aps >= 1:
                stimamp_top = stimamp
            elif self.n_aps < 1:
                stimamp_bottom = stimamp
        return

    def run_protocol(
        self,
        saving: Saving,
        recording: Recording,
        start_time: float,
        protocol_mode: str = 'ACTIVATION_THRESHOLD',
        amps: list = None,
        t_init_ss: float = -200,
        dt_init_ss: float = 10,
    ):
        """Determine protocol and submit runs for simulation.

        :param saving: instance of Saving class
        :param recording: instance of Recording class
        :param start_time: time at the very beginning of simulation
        :param protocol_mode: protocol for simulation ('FINITE_AMPLITUDES', 'BLOCK_THRESHOLD', 'ACTIVATION_THRESHOLD')
        :param amps: list of amplitudes to simulate
        :param t_init_ss: the time (<=0ms) for the system to reach steady state before starting the simulation [ms]
        :param dt_init_ss: the time step used to reach steady state [ms]
        """
        amps = amps or []
        # Determine protocol
        if protocol_mode != 'FINITE_AMPLITUDES':
            find_thresh = True
            if protocol_mode == 'BLOCK_THRESHOLD':
                find_block_thresh = 1
            elif protocol_mode == 'ACTIVATION_THRESHOLD':
                find_block_thresh = 0
        elif protocol_mode == 'FINITE_AMPLITUDES':
            find_thresh = False
            find_block_thresh = False

        if find_thresh:  # Protocol is BLOCK_THRESHOLD or ACTIVATION_THRESHOLD
            self.find_threshold(saving, recording, find_block_thresh)
            # todo: remove saving class
            # saving.save_variables(selfy, recording, stimulation.dt)  # Save user-specified variables
            # saving.save_runtime(selfy, time_individual)  # Save runtime of simulation

        else:  # Protocol is FINITE_AMPLITUDES
            self.finite_amplitudes(saving, recording, start_time, amps, t_init_ss, dt_init_ss)

    def run_sim(
        self,
        stimamp: float,
        recording: Recording,
        find_block_thresh: bool = False,
        saving: Saving = None,
        t_init_ss: float = -200,
        dt_init_ss: float = 10,
    ):
        """Run a simulation for a single stimulation amplitude.

        :param stimamp: amplitude to be applied to extracellular stimulation
        :param recording: instance of Recording class
        :param find_block_thresh: true if BLOCK_THRESHOLD protocol, false otherwise
        :param saving: instance of Saving class
        :param t_init_ss: the time (<=0ms) for the system to reach steady state before starting the simulation [ms]
        :param dt_init_ss: the time step used to reach steady state [ms]
        :return: Fiber object
        """

        def balance():
            """Balance membrane currents for Tigerholm model."""
            v_rest = -55
            for s in self.fiber.sections:
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

        def steady_state(sim_dt: float, t_init_ss: float, dt_init_ss: float):
            """Allow system to reach steady-state by using a large dt before simulation.

            :param sim_dt: user-specified time step for simulation
            :param t_init_ss: the time (<=0ms) for the system to reach steady state before starting the simulation [ms]
            :param dt_init_ss: the time step used to reach steady state [ms]
            """
            h.t = t_init_ss  # Start before t=0
            h.dt = dt_init_ss  # Large dt
            while h.t <= -dt_init_ss:
                h.fadvance()
            h.dt = sim_dt  # Set simulation time step to user-specified time step
            h.t = 0  # Probably redundant, reset simulation time to zero
            h.fcurrent()
            h.frecord_init()

        # If saving variables, record variables
        if saving is not None:
            if saving.time_vm or saving.space_vm:
                recording.record_vm(self.fiber)
            if saving.time_gating or saving.space_gating:
                recording.record_gating(self.fiber)
            if saving.istim:
                recording.record_istim(self.istim)
            if saving.ap_end_times:
                recording.record_ap_end_times(self.fiber, saving.ap_end_inds, saving.ap_end_thresh)

        h.finitialize(self.fiber.v_init)  # Initialize the simulation
        if self.fiber.fiber_model == 'TIGERHOLM':  # Balance membrane currents if Tigerholm
            balance()

        self.initialize_extracellular()  # Set extracellular stimulation at each segment to zero
        steady_state(self.dt, t_init_ss, dt_init_ss)  # Allow system to reach steady-state before simulation
        h.celsius = self.fiber.temperature  # Set simulation temperature

        # Set up APcount
        recording.record_ap(self.fiber)

        # Begin simulation
        n_tsteps = len(self.waveform)
        for i in range(0, n_tsteps):
            if h.t > self.tstop:
                break
            amp = self.waveform[i]
            scaled_stim = [stimamp * amp * x for x in self.potentials]
            self.update_extracellular(scaled_stim)

            h.fadvance()
        # Done with simulation

        # Insert vectors of 0's for gating parameters at passive end nodes
        if saving is not None and (saving.time_gating or saving.space_gating):
            recording.record_gating(self.fiber, fix_passive=True)

        # Check if APs occurred
        self.n_aps = recording.ap_checker(self.fiber, find_block_thresh)

        if saving is None:
            print(f'{int(self.n_aps)} AP(s) detected')
        return self
