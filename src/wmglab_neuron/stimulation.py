"""Defines Stimulation class.

The copyrights of this software are owned by Duke University. Please
refer to the LICENSE and README.md files for licensing instructions. The
source code can be found on the following GitHub repository:
https://github.com/wmglab-duke/ascent
"""
from neuron import h

from src.wmglab_neuron import Fiber, FiberModel, Recording

h.load_file('stdrun.hoc')


class Stimulation:
    """Manage stimulation of NEURON simulations."""

    def __init__(
        self,
        fiber: Fiber,
        waveform: list[int],
        potentials: list[int],
        dt: float = 0.001,
        tstop: float = 50,
        t_init_ss: float = -200,
        dt_init_ss: float = 10,
    ):
        """Initialize Stimulation class.

        :param fiber: instance of Fiber class
        :param waveform: list of amplitudes at each time bounds_search_step of the simulation
        :param potentials: list of extracellular potentials to be applied along the fiber length
        :param dt: time bounds_search_step for simulation [seconds]
        :param tstop: time bounds_search_step for simulation [seconds]
        :param t_init_ss: the time (<=0ms) for the system to reach steady state before starting the simulation [ms]
        :param dt_init_ss: the time bounds_search_step used to reach steady state [ms]
        """
        # TODO: need to think about making this extensible so that users could add custom simulations
        self.fiber = fiber
        self.waveform = waveform
        self.dt = dt
        self.tstop = tstop
        self.istim = None
        self.t_init_ss = t_init_ss
        self.dt_init_ss = dt_init_ss
        assert len(potentials) == len(
            fiber.coordinates
        ), 'Number of fiber coordinates does not match number of potentials'
        self.potentials = potentials

        # todo: pass fiber as a run_sim argument so that a Stimulation class can operate on any fiber it is given.

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
        condition: str = "activation",  # TODO: change to enums
        bounds_search_mode: str = 'PERCENT_INCREMENT',  # TODO: change to enums
        bounds_search_step: float = 10,
        termination_mode: str = 'PERCENT_DIFFERENCE',  # TODO: change to enums
        termination_tolerance: float = 1,
        stimamp_top: float = -1,
        stimamp_bottom: float = -0.01,
        max_iterations=100,
    ):
        """Binary search to find threshold amplitudes.

        :param recording: instance of Recording class
        :param condition: condition to search for threshold (activation or block)
        :param bounds_search_mode: indicates how to change upper and lower bounds for the binary search
        :param bounds_search_step: the incremental increase/decrease of the upper/lower bound in the binary search
            if bounds_search_mode is 'PERCENT_INCREMENT' this is the percentage increase/decrease,
            if bounds_search_mode is 'ABSOLUTE_INCREMENT' this is the absolute increase/decrease
        :param termination_mode: indicates when upper and lower bounds converge on a solution of appropriate precision
        :param termination_tolerance: difference between upper and lower bounds that indicates convergence
            if absolute difference if termination_mode is 'ABSOLUTE_DIFFERENCE',
            or percentage difference if termination_mode is 'PERCENT_DIFFERENCE'
        :param stimamp_top: the upper-bound stimulation amplitude first tested in a binary search for thresholds
        :param stimamp_bottom: the lower-bound stimulation amplitude first tested in a binary search for thresholds
        :param max_iterations: the maximum number of iterations for finding search bounds
        :raises RuntimeError: If stimamp bottom is supra-threshold and stimamp top is sub-threshold
        :raises ValueError: If stimamp bottom and stimamp top have different signs
        :raises ValueError: If stimamp top does not exceed stimamp bottom
        :return: the threshold amplitude for the given condition, and the number of detected aps
        """
        # TODO: enable option for specific number of aps to qualify as threshold
        # todo: add error if condition is not activation or block
        if abs(stimamp_top) < abs(stimamp_bottom):
            raise ValueError("stimamp_top must be greater than stimamp_bottom in magnitude.")
        if stimamp_top * stimamp_bottom < 0:
            raise ValueError("stimamp_top and stimamp_bottom must have the same sign.")
        # Determine searching parameters for binary search bounds
        rel_increment = round(bounds_search_step / 100, 4)
        abs_increment = round(bounds_search_step, 4)

        # Determine searching parameters for termination of binary search
        abs_thresh_resoln = round(termination_tolerance, 4)
        rel_thresh_resoln = round(termination_tolerance / 100, 4)
        # todo: add comments to this function
        # first check stimamps
        supra_bot = self.run_sim(stimamp_bottom, check_threshold=condition)
        supra_top = self.run_sim(stimamp_top, check_threshold=condition)
        # Determine upper- and lower-bounds for simulation
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            if not supra_bot and supra_top:  # found search bounds
                break
            elif supra_bot and not supra_top:
                raise RuntimeError(
                    "stimamp_bottom was found to be supra-threshold, "
                    "and stimamp_top was found to be subthreshold, which should be impossible."
                )
            elif not supra_bot and not supra_top:
                # search upward with stimamp top
                stimamp_bottom = stimamp_top
                if bounds_search_mode == 'ABSOLUTE_INCREMENT':  # Todo: change to enum
                    stimamp_top = stimamp_top + abs_increment
                elif bounds_search_mode == 'PERCENT_INCREMENT':
                    stimamp_top = stimamp_top * (1 + rel_increment)
                supra_top = self.run_sim(stimamp_top, check_threshold=condition)
            elif supra_bot and supra_top:
                # search downward with stimamp bottom
                stimamp_top = stimamp_bottom
                if bounds_search_mode == 'ABSOLUTE_INCREMENT':
                    stimamp_bottom = stimamp_bottom - abs_increment
                elif bounds_search_mode == 'PERCENT_INCREMENT':
                    stimamp_bottom = stimamp_bottom * (1 - rel_increment)
                supra_bot = self.run_sim(stimamp_bottom, check_threshold=condition)
        else:  # todo: add print statements back in
            raise RuntimeError(f"Reached maximum number of iterations ({max_iterations}) without finding threshold.")
        # Enter binary search
        while True:
            stimamp_prev = stimamp_top

            stimamp = (stimamp_bottom + stimamp_top) / 2

            suprathreshold = self.run_sim(stimamp, check_threshold=condition)

            if termination_mode == 'PERCENT_DIFFERENCE':  # todo: make this a function
                thresh_resoln = abs(rel_thresh_resoln)
                tolerance = abs((stimamp_bottom - stimamp_top) / stimamp_top)
            elif termination_mode == 'ABSOLUTE_DIFFERENCE':  # todo: enum
                thresh_resoln = abs(abs_thresh_resoln)
                tolerance = abs(stimamp_bottom - stimamp_top)

            # Check to see if stimamp is at threshold
            if tolerance < thresh_resoln:
                if not suprathreshold:
                    stimamp = stimamp_prev
                # Run one more time at threshold to save user-specified variables
                n_aps = self.run_sim(stimamp, recording=recording)
                break
            elif suprathreshold:
                stimamp_top = stimamp
            elif not suprathreshold:
                stimamp_bottom = stimamp

        return stimamp, n_aps

    def run_sim(
        self, stimamp: float, recording: Recording = None, check_threshold: str = None, check_threshold_interval=10
    ):  # noqa: C901
        """Run a simulation for a single stimulation amplitude.

        :param stimamp: amplitude to be applied to extracellular stimulation
        :param recording: instance of Recording class
        :param check_threshold: condition to check for threshold (activation or block) or None
        :param check_threshold_interval: interval to check if threshold is reached and if so, exit ("activation" only)
        :raises ValueError: if waveform length is not equal to number of time steps
        :return: number of detected aps if check_threshold is None, else True if supra-threshold, else False
        """
        print('Running:', stimamp)

        def steady_state():
            """Allow system to reach steady-state by using a large dt before simulation."""
            h.t = self.t_init_ss  # Start before t=0
            h.dt = self.dt_init_ss  # Large dt
            while h.t <= -self.dt_init_ss:
                h.fadvance()
            h.dt = self.dt  # Set simulation time bounds_search_step to user-specified time bounds_search_step
            h.t = 0  # Probably redundant, reset simulation time to zero
            h.fcurrent()
            h.frecord_init()

        def initialize_recording(rec):
            # init rec
            if rec is None:
                rec = Recording(self.fiber)
            else:
                rec.reset()
            # If saving variables, record variables
            if rec.save_vm:
                rec.record_vm(self.fiber)
            if rec.save_gating:
                rec.record_gating(self.fiber)
            if rec.save_istim:
                assert self.istim is not None, 'There must be intracellular stimulation to record istim'
                rec.record_istim(self.istim)
            # Set up APcount
            rec.record_ap(self.fiber)
            return rec

        recording = initialize_recording(recording)

        # TODO: Add error if checkthresholdinterval and block
        h.finitialize(self.fiber.v_rest)  # Initialize the simulation
        if self.fiber.fiber_model == FiberModel.TIGERHOLM:  # Balance membrane currents if Tigerholm
            self.fiber.balance()

        self.initialize_extracellular()  # Set extracellular stimulation at each segment to zero
        steady_state()  # Allow system to reach steady-state before simulation
        h.celsius = self.fiber.temperature  # Set simulation temperature

        # Begin simulation
        n_tsteps = len(self.waveform)
        if n_tsteps != self.tstop / self.dt:
            raise ValueError(
                f"Waveform length ({n_tsteps}) not equal to the number of time steps (t_stop*dt={self.tstop/self.dt})."
            )
        for i in range(0, n_tsteps):
            amp = self.waveform[i]
            scaled_stim = [stimamp * amp * x for x in self.potentials]
            self.update_extracellular(scaled_stim)

            h.fadvance()
            if (
                check_threshold is not None
                and i % check_threshold_interval == 0
                and recording.threshold_checker(self.fiber)
            ):
                print(i * self.dt)
                break
            # todo: find time of threshold crossing and use 2x that for supra and subthreshold

        # Done with simulation

        # Insert vectors of 0's for gating parameters at passive end nodes
        if recording.save_gating:
            recording.record_gating(self.fiber, fix_passive=True)

        if not check_threshold:
            # print(f'{int(n_aps)} AP(s) detected')
            return recording.ap_checker(self.fiber)  # todo: not using istim delay or ap detect location
        elif check_threshold == "activation":
            return recording.threshold_checker(self.fiber)
        elif check_threshold == "block":
            return recording.threshold_checker(self.fiber, block=True)
