"""Defines Stimulation class.

The copyrights of this software are owned by Duke University. Please
refer to the LICENSE and README.md files for licensing instructions. The
source code can be found on the following GitHub repository:
https://github.com/wmglab-duke/ascent
"""

from neuron import h

from src.wmglab_neuron import Fiber, Recording

h.load_file('stdrun.hoc')


class Stimulation:
    """Manage stimulation of NEURON simulations."""

    def __init__(
        self, fiber: Fiber, potentials: list[float], waveform: list[int], dt: float = 0.001, tstop: float = 50
    ):
        """Initialize Stimulation class.

        :param fiber: instance of Fiber class
        :param potentials: list of extracellular potentials Ve(x)
        :param waveform: list of amplitudes at each time step of the simulation
        :param dt: time step for simulation [seconds]
        :param tstop: time step for simulation [seconds]
        """
        # TODO: need to think about making this extensible so that users could add custom simulations
        self.fiber = fiber
        self.potentials = potentials
        self.waveform = waveform
        self.dt = dt
        self.tstop = tstop
        self.istim = None
        # todo: error if len(potentials) != len(fiber coords)
        # todo: possibly add the ability to run on a list of fiber objects
        # this could allow massive parallelization
        # also then potentials need to be stored in the fiber

    def add_intracellular_stim(
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
        # TODO: need to init with a zero istim if none is provided
        # or skip recording istim if none is provided because recording istim errors
        # right now if this function isnt run
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

    def find_threshold(  # noqa: C901
        self,
        recording: Recording,
        condition: str = "activation",
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

        :param recording: instance of Recording class
        :param condition: condition to search for threshold (activation or block)
        :param bounds_search_mode: indicates how to change upper and lower bounds for the binary search
        :param step: the incremental increase/decrease of the upper/lower bound in the binary search
        :param termination_mode: indicates when upper and lower bounds converge on a solution of appropriate precision
        :param termination_percent: percent difference between upper/lower bounds for finding threshold (e.g., 1 is 1%)
        :param termination_tolerance: the absolute difference between upper/lower bounds for finding threshold [mA]
        :param stimamp_top: the upper-bound stimulation amplitude first tested in a binary search for thresholds
        :param stimamp_bottom: the lower-bound stimulation amplitude first tested in a binary search for thresholds
        :param t_init_ss: the time (<=0ms) for the system to reach steady state before starting the simulation [ms]
        :param dt_init_ss: the time step used to reach steady state [ms]
        :return: the threshold amplitude for the given condition, and the number of detected aps
        """
        # TODO: only record data on threshold? or maybe once search bounds are set
        # TODO: enable option for specific number of aps to qualify as threshold
        # todo: change find threshold to argument: condition=activation or block
        # todo: add error if condition is not activation or block
        # todo: use a single "target" threshold and move a certain percent or increment from that
        # todo: this can be hugely simplified
        # todo: two functions here, one checks block thresh and the other checks activation thresh

        find_block_thresh = condition == "block"
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
                n_aps = self.run_sim(
                    stimamp_top,
                    recording,
                    find_block_thresh=find_block_thresh,
                    t_init_ss=t_init_ss,
                    dt_init_ss=dt_init_ss,
                )

                if n_aps == 0:
                    if find_block_thresh == 0:
                        print(
                            'WARNING: Initial stimamp_top value does not elicit an AP - '
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
                n_aps = self.run_sim(
                    stimamp_bottom,
                    recording,
                    find_block_thresh=find_block_thresh,
                    t_init_ss=t_init_ss,
                    dt_init_ss=dt_init_ss,
                )

                if n_aps != 0:
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
            n_aps = self.run_sim(
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
                if n_aps < 1:
                    stimamp = stimamp_prev
                print(f'Done searching! stimamp: {stimamp:.6f} mA for extracellular\n')

                # Run one more time at threshold to save user-specified variables
                # TODO: Currently looks like it is recording every time
                n_aps = self.run_sim(
                    stimamp,
                    recording,
                    find_block_thresh=find_block_thresh,
                    t_init_ss=t_init_ss,
                    dt_init_ss=dt_init_ss,
                )
                # todo: remove saving class
                break
            elif n_aps >= 1:
                stimamp_top = stimamp
            elif n_aps < 1:
                stimamp_bottom = stimamp
        return stimamp, n_aps

    def run_sim(
        self,
        stimamp: float,
        recording: Recording,
        find_block_thresh: bool = False,
        t_init_ss: float = -200,
        dt_init_ss: float = 10,
    ):
        """Run a simulation for a single stimulation amplitude.

        :param stimamp: amplitude to be applied to extracellular stimulation
        :param recording: instance of Recording class
        :param find_block_thresh: true if BLOCK_THRESHOLD protocol, false otherwise
        :param t_init_ss: the time (<=0ms) for the system to reach steady state before starting the simulation [ms]
        :param dt_init_ss: the time step used to reach steady state [ms]
        :return: Fiber object
        """
        # TODO: make recording optional

        def balance():
            """Balance membrane currents for Tigerholm model."""
            # TODO: this should be a method of fiber? or decorate runsim with a function that balances if tigerholm
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

        # todo: reset recording at beginning
        # todo: remove repeated input variables such as init SS
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

        # init recording
        recording.reset()

        # If saving variables, record variables
        if recording.save_vm:
            recording.record_vm(self.fiber)
        if recording.save_gating:
            recording.record_gating(self.fiber)
        if recording.save_istim:
            recording.record_istim(self.istim)

        h.finitialize(self.fiber.v_init)  # Initialize the simulation
        if self.fiber.fiber_model == 'TIGERHOLM':  # Balance membrane currents if Tigerholm
            balance()

        self.initialize_extracellular()  # Set extracellular stimulation at each segment to zero
        steady_state(self.dt, t_init_ss, dt_init_ss)  # Allow system to reach steady-state before simulation
        h.celsius = self.fiber.temperature  # Set simulation temperature

        # Set up APcount
        recording.record_ap(self.fiber)

        # TODO: looks like this stops immediately upon reaching the end of the waveform?
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
        if recording.save_gating:
            recording.record_gating(self.fiber, fix_passive=True)

        # Check if APs occurred
        n_aps = recording.ap_checker(self.fiber, find_block_thresh)

        print(f'{int(n_aps)} AP(s) detected')

        return n_aps
