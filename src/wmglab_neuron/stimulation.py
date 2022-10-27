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
        dt: float = 0.001,
        tstop: float = 50,
        t_init_ss: float = -200,
        dt_init_ss: float = 10,
    ):
        """Initialize Stimulation class.

        :param fiber: instance of Fiber class
        :param waveform: list of amplitudes at each time step of the simulation
        :param dt: time step for simulation [seconds]
        :param tstop: time step for simulation [seconds]
        :param t_init_ss: the time (<=0ms) for the system to reach steady state before starting the simulation [ms]
        :param dt_init_ss: the time step used to reach steady state [ms]
        """
        # TODO: need to think about making this extensible so that users could add custom simulations
        self.fiber = fiber
        self.waveform = waveform
        self.dt = dt
        self.tstop = tstop
        self.istim = None
        self.t_init_ss = t_init_ss
        self.dt_init_ss = dt_init_ss
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

    def find_threshold(
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
        :raises RuntimeError: If stimamp bottom is supra-threshold and stimamp top is sub-threshold
        :return: the threshold amplitude for the given condition, and the number of detected aps
        """
        # TODO: only record data on threshold? or maybe once search bounds are set
        # TODO: enable option for specific number of aps to qualify as threshold
        # todo: change find threshold to argument: condition=activation or block
        # todo: add error if condition is not activation or block
        # todo: use a single "target" threshold and move a certain percent or increment from that
        # todo: this can be hugely simplified
        # todo: two functions here, one checks block thresh and the other checks activation thresh
        # todo change this to use target stimamp instead of bottom and top
        # todo: only record data on threshold
        # todo add requirement that the magnitude of stimamp top is greater than the magnitude of stimamp bottom

        # Determine searching parameters for binary search bounds
        rel_increment = round(step / 100, 4)
        abs_increment = round(step, 4)

        # Determine searching parameters for termination of binary search
        abs_thresh_resoln = round(termination_tolerance, 4)
        rel_thresh_resoln = round(termination_percent / 100, 4)
        # todo: add comments to this function
        # todo change condition to an enum
        # first check stimamps
        supra_bot = self.run_sim(stimamp_bottom, check_threshold=condition)
        supra_top = self.run_sim(stimamp_top, check_threshold=condition)
        # Determine upper- and lower-bounds for simulation
        iterations = 0
        while iterations < 100:  # TODO let user set max iterations
            iterations += 1
            if not supra_bot and supra_top:
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
            raise RuntimeError("Reached maximum number of iterations without finding threshold.")
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

    def run_sim(self, stimamp: float, recording: Recording = None, check_threshold: str = None):  # noqa: C901
        """Run a simulation for a single stimulation amplitude.

        :param stimamp: amplitude to be applied to extracellular stimulation
        :param recording: instance of Recording class
        :param check_threshold: condition to check for threshold (activation or block) or None
        :return: number of detected aps if check_threshold is None, else True if supra-threshold, else False
        """
        # TODO: make recording optional

        def steady_state():
            """Allow system to reach steady-state by using a large dt before simulation."""
            h.t = self.t_init_ss  # Start before t=0
            h.dt = self.dt_init_ss  # Large dt
            while h.t <= -self.dt_init_ss:
                h.fadvance()
            h.dt = self.dt  # Set simulation time step to user-specified time step
            h.t = 0  # Probably redundant, reset simulation time to zero
            h.fcurrent()
            h.frecord_init()

        def initialize_recording(recording):
            # init recording
            if recording is None:
                recording = Recording(self.fiber)
            else:
                recording.reset()
            # If saving variables, record variables
            if recording.save_vm:
                recording.record_vm(self.fiber)
            if recording.save_gating:
                recording.record_gating(self.fiber)
            if recording.save_istim:
                assert self.istim is not None, 'There must be intracellular stimulation to record istim'
                recording.record_istim(self.istim)
            # Set up APcount
            recording.record_ap(self.fiber)
            return recording

        h.finitialize(self.fiber.v_init)  # Initialize the simulation
        if self.fiber.fiber_model == FiberModel.TIGERHOLM:  # Balance membrane currents if Tigerholm
            self.fiber.balance()

        self.initialize_extracellular()  # Set extracellular stimulation at each segment to zero
        steady_state()  # Allow system to reach steady-state before simulation
        h.celsius = self.fiber.temperature  # Set simulation temperature

        recording = initialize_recording(recording)

        # TODO: looks like this stops immediately upon reaching the end of the waveform?
        # Begin simulation
        n_tsteps = len(self.waveform)
        for i in range(0, n_tsteps):
            if h.t > self.tstop:
                break
            amp = self.waveform[i]
            scaled_stim = [stimamp * amp * x for x in self.fiber.potentials]
            self.update_extracellular(scaled_stim)

            h.fadvance()
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
