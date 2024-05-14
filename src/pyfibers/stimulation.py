"""Defines ScaledStim class."""

from __future__ import annotations

import warnings
from enum import Enum, unique
from typing import Any, Callable

import numpy as np
from neuron import h
from numpy import bool_
from scipy.signal import argrelextrema

from pyfibers import Fiber, FiberModel

h.load_file('stdrun.hoc')


@unique
class ThresholdCondition(Enum):
    """ThresholdCondition."""

    ACTIVATION = 0
    BLOCK = 1


@unique
class BoundsSearchMode(Enum):
    """Bounds search modes."""

    PERCENT_INCREMENT = 0
    ABSOLUTE_INCREMENT = 1


@unique
class TerminationMode(Enum):
    """Termination modes."""

    PERCENT_DIFFERENCE = 0
    ABSOLUTE_DIFFERENCE = 1


@unique
class BisectionMean(Enum):
    """Termination modes."""

    GEOMETRIC = 0
    ARITHMETIC = 1


class ScaledStim:
    """Manage stimulation of NEURON simulations."""

    def __init__(
        self: ScaledStim,
        waveform: list[int],
        dt: float = 0.001,
        tstop: float = 50,
        t_init_ss: float = -200,
        dt_init_ss: float = 10,
        pad_waveform: bool = True,
        truncate_waveform: bool = True,
    ) -> None:
        """Initialize ScaledStim class.

        :param waveform: list of amplitudes at each time bounds_search_step of the simulation
        :param dt: time bounds_search_step for simulation [seconds]
        :param tstop: time bounds_search_step for simulation [seconds]
        :param t_init_ss: the time (<=0ms) for the system to reach steady state before starting the simulation [ms]
        :param dt_init_ss: the time bounds_search_step used to reach steady state [ms]
        :param pad_waveform: if true, extend waveform until it is of length tstop/dt
        :param truncate_waveform: if true, truncate waveform until it is of length tstop/dt
        :raises RuntimeError: if called before any sections/fibers are created
        """
        self.istim_params: dict[str, float] = {}
        self.waveform: np.typing.NDArray[np.float64] = np.array(waveform)
        self.dt = dt
        self.tstop = tstop
        self.istim: h.trainIClamp = None
        self.istim_record = None
        self.t_init_ss = t_init_ss
        self.dt_init_ss = dt_init_ss
        self.exit_t: float = None
        self.pad = pad_waveform
        self.truncate = truncate_waveform
        self.n_timesteps: int = None
        try:
            self.time = h.Vector().record(h._ref_t)
        except RuntimeError:
            raise RuntimeError(
                'Could not set up time recording vector. Maybe you haven\'t created any ' 'sections/fibers yet?'
            )
        self._prep_waveform()

    # MAGIC METHODS
    def __str__(self: ScaledStim) -> str:
        """Return string representation of ScaledStim class."""  # noqa: DAR201
        return (
            f'ScaledStim: {self.dt * self.tstop} ms (dt={self.dt} ms)'
            f' (t_init_ss={self.t_init_ss} ms, dt_init_ss={self.dt_init_ss} ms)'
            f' istim_params={self.istim_params}'
        )

    def __repr__(self: ScaledStim) -> str:
        """Return a string representation of the ScaledStim."""  # noqa: DAR201
        # TODO: make this more informative for developers
        return self.__str__()

    # PRIVATE METHODS
    def _add_istim(
        self: ScaledStim,
        fiber: Fiber,
        delay: float = 0,
        pw: float = 0,
        dur: float = 0,
        freq: float = 0,
        amp: float = 0,
        ind: int = None,
        loc: float = None,
    ) -> ScaledStim:
        """Create instance of trainIClamp for intracellular stimulation.

        :param fiber: instance of Fiber class to add intracellular stimulation to
        :param delay: the delay from the start of the simulation to the onset of the intracellular stimulation [ms]
        :param pw: the pulse duration of the intracellular stimulation [ms]
        :param dur: the duration from the start of the simulation to the end of the intracellular stimulation [ms]
        :param freq: the intracellular stimulation frequency [Hz]
        :param amp: the intracellular stimulation amplitude [nA]
        :param ind: the section index (unmyelinated) or node of Ranvier number (myelinated) receiving stimulation
        :param loc: node location along the fiber (using NEURON style indexing)
        :return: instance of ScaledStim class
        """
        assert (ind is None) != (loc is None), 'Must specify either ind or loc, but not both'
        ind = ind or fiber.loc_index(loc)
        if (ind == 0 or ind == len(fiber.sections) - 1) and fiber.passive_end_nodes is True:
            warnings.warn('You are trying to intracellularly stimulate a passive end node.', stacklevel=2)
        if ind is not None:
            intracellular_stim = h.trainIClamp(fiber[ind](0.5))
        intracellular_stim.delay = delay
        intracellular_stim.PW = pw
        intracellular_stim.train = dur
        intracellular_stim.freq = freq
        intracellular_stim.amp = amp
        self.istim = intracellular_stim
        self.istim_record = h.Vector().record(self.istim._ref_i)
        return self

    def _steady_state(self: ScaledStim) -> None:
        """Allow system to reach steady-state by using a large dt before simulation."""
        h.t = self.t_init_ss  # Start before t=0
        h.dt = self.dt_init_ss  # Large dt
        while h.t <= -self.dt_init_ss:
            h.fadvance()
        h.dt = self.dt  # Set simulation time bounds_search_step to user-specified time bounds_search_step
        h.t = 0  # Probably redundant, reset simulation time to zero
        h.fcurrent()
        h.frecord_init()

    @staticmethod
    def _initialize_extracellular(fiber: Fiber) -> None:
        """Set extracellular stimulation values to zero along entire fiber.

        :param fiber: instance of Fiber class to initialize extracellular stimulation for
        """
        for section in fiber.sections:
            section(0.5).e_extracellular = 0

    @staticmethod
    def _update_extracellular(fiber: Fiber, e_stims: list[float]) -> None:
        """Update the applied extracellular stimulation all along the fiber length.

        :param fiber: instance of Fiber class to apply extracellular stimulation to
        :param e_stims: list of extracellular stimulations to apply along fiber length
        """
        for x, section in enumerate(fiber.sections):
            section(0.5).e_extracellular = e_stims[x]

    # PUBLIC METHODS
    def set_intracellular_stim(
        self: ScaledStim,
        delay: float,
        pw: float,
        dur: float,
        freq: float,
        amp: float,
        ind: int = None,
        loc: float = None,
    ) -> None:
        """Set intracellular stimulation parameters.

        :param delay: the delay from the start of the simulation to the onset of the intracellular stimulation [ms]
        :param pw: the pulse duration of the intracellular stimulation [ms]
        :param dur: the duration from the start of the simulation to the end of the intracellular stimulation [ms]
        :param freq: the intracellular stimulation frequency [Hz]
        :param amp: the intracellular stimulation amplitude [nA]
        :param ind: the section index (unmyelinated) or node of Ranvier number (myelinated) receiving stimulation
        :param loc: the node along the fiber (unmyelinated) receiving stimulation
        """
        assert (ind is None) != (loc is None), 'Must specify either ind or loc, but not both'
        self.istim_params = {
            'delay': delay,
            'pw': pw,
            'dur': dur,
            'freq': freq,
            'amp': amp,
            'ind': ind,
            'loc': loc,
        }

    def _prep_potentials(self: ScaledStim, fiber: Fiber) -> None:
        """Prepare the fiber object's potentials for further processing.

        Converts potentials into a 2D numpy array if they are not already in that format.
        This function assumes that the input is either a single 1D numpy array or a list of 1D arrays.
        :param fiber: instance of Fiber class to prepare potentials for
        """
        assert fiber.potentials is not None, 'No fiber potentials found'

        # Check if potentials is a single 1D numpy array and wrap it in a list
        if isinstance(fiber.potentials, np.ndarray) and fiber.potentials.ndim == 1:
            fiber.potentials = [fiber.potentials]

        # Convert each potential in the list to a numpy array if not already
        processed_potentials = [np.array(potential) for potential in fiber.potentials]

        # asssert all potentials equal length of fiber coordinates
        assert all(
            len(potential) == len(fiber.coordinates) for potential in processed_potentials
        ), 'Length of fiber potentials does not match length of fiber coordinates'

        # Combine all processed potentials into a single 2D array
        fiber.potentials = np.vstack(processed_potentials)

        if np.all(fiber.potentials == 0):
            warnings.warn('All fiber potentials are zero.', RuntimeWarning, stacklevel=2)

    # BELONGS IN SCALEDSTIM SUBCLASS
    def _prep_waveform(self: ScaledStim) -> None:
        """Prepare waveform for simulation.

        Accepts waveform as either a list of 1D numpy arrays or a single 1D numpy array,
        processes each waveform independently, and returns a 2D numpy array.

        :raises AssertionError: if any processed waveform row length is not equal to the number of time steps
        """
        self.n_timesteps = int(self.tstop / self.dt)

        # Initialize list to store processed waveforms
        processed_waveforms = []

        # Check if waveform is a single 1D numpy array and wrap it in a list
        if isinstance(self.waveform, np.ndarray) and self.waveform.ndim == 1:
            self.waveform = [self.waveform]

        # Process each waveform
        for row in self.waveform:
            row = np.array(row)  # Ensure row is a numpy array
            if self.pad and (self.n_timesteps > len(row)):
                # Extend waveform row until it is of length tstop/dt
                print(f"Padding waveform {len(row) * self.dt} ms to {self.tstop} ms (with 0's)")
                if row[-1] != 0:
                    warnings.warn('Padding a waveform that does not end with 0.', stacklevel=2)
                row = np.hstack([row, [0] * (self.n_timesteps - len(row))])

            if self.truncate and (self.n_timesteps < len(row)):
                # Truncate waveform row until it is of length tstop/dt
                print(f"Truncating waveform {len(row) * self.dt} ms to {self.tstop} ms")
                if any(row[self.n_timesteps :]):
                    warnings.warn('Truncating waveform removed non-zero values.', stacklevel=2)
                row = row[: self.n_timesteps]

            # Check that waveform row length is equal to number of time steps
            assert (
                len(row) == self.n_timesteps
            ), f'Waveform row length does not match number of time steps {self.tstop / self.dt}'

            processed_waveforms.append(row)  # Append processed row to the list

        # Convert list of processed rows into a 2D numpy array
        self.waveform = np.vstack(processed_waveforms)

    def potentials_at_time(self: ScaledStim, i: int, fiber: Fiber) -> np.typing.NDArray[np.float64]:
        """Get potentials at a given time.

        :param i: index of time step
        :param fiber: fiber to get potentials for
        :return: list of potentials at time i
        """
        # go through each waveform and multiply value for this time point by the corresponding potential set
        potentials = np.zeros(fiber.potentials.shape[1])
        for potential_set, waveform in zip(fiber.potentials, self.waveform):
            potentials += waveform[i] * potential_set
        return potentials  # TODO test this

    def run_sim(
        self: ScaledStim,
        stimamp: float,
        fiber: Fiber,
        ap_detect_location: float = 0.9,
        exit_func: Callable = lambda x: False,
        exit_func_interval: int = 100,
        use_exit_t: bool = False,
    ) -> tuple[float, float]:
        """Run a simulation for a single stimulation amplitude.

        :param stimamp: amplitude to be applied to extracellular stimulation
        :param fiber: fiber to be stimulated
        :param ap_detect_location: location to detect action potentials (percent along fiber)
        :param exit_func: function to call to check if simulation should be exited
        :param exit_func_interval: interval to call exit_func
        :param use_exit_t: if True, use the time returned by exit_func as the simulation end time
        :return: number of detected aps and time of last detected ap
        """
        print('Running:', round(stimamp, 3))

        self.validate_scaling_inputs(fiber)

        self.pre_run_setup(fiber)

        # Run simulation
        for i in range(self.n_timesteps):
            timestep_potentials = stimamp * self.potentials_at_time(i, fiber)
            self._update_extracellular(fiber, timestep_potentials)

            h.fadvance()

            if i % exit_func_interval == 0 and exit_func(fiber):
                break
            if use_exit_t and self.exit_t and h.t >= self.exit_t:
                break

        # get precision from the number of decimal places in self.dt
        precision = len(str(self.dt).split('.')[1])

        return self.ap_checker(fiber, ap_detect_location=ap_detect_location, precision=precision)

    def validate_scaling_inputs(self: ScaledStim, fiber: Fiber) -> None:
        """Validate scaling inputs before running simulation.

        :param fiber: instance of Fiber class to validate scaling inputs for
        """
        self._prep_waveform()
        self._prep_potentials(fiber)
        assert len(fiber.potentials) == len(
            self.waveform
        ), 'Number of fiber potentials sets does not match number of waveforms'

    def pre_run_setup(self: ScaledStim, fiber: Fiber) -> None:
        """Set up simulation before running.

        :param fiber: instance of Fiber class to set up simulation for
        """
        h.celsius = fiber.temperature  # Set simulation temperature
        h.finitialize(fiber.v_rest)  # Initialize the simulation
        fiber.apcounts()  # record action potentials
        if fiber.fiber_model == FiberModel.TIGERHOLM:  # Balance membrane currents if Tigerholm
            fiber.balance()
        self._initialize_extracellular(fiber)  # Set extracellular stimulation at each segment to zero
        self._steady_state()  # Allow system to reach steady-state before simulation
        if self.istim_params:  # add istim train if specified
            self._add_istim(fiber, **self.istim_params)  # type: ignore

    # MAYBE DOESNT BELONG IN CLASS AT ALL (THRESHOLD RELATED)

    @staticmethod
    def ap_checker(
        fiber: Fiber,
        ap_detect_location: float = 0.9,
        precision: int = 3,
    ) -> tuple[int, float]:
        """Check to see if an action potential occurred at the end of a run.

        # remove this function and check in the respective functions

        :param fiber: instance of fiber class
        :param ap_detect_location: is the location (decimal % of fiber length) where APs are detected for threshold
        :param precision: number of decimal places to round to
        :return: number of action potentials that occurred
        """
        # Determine user-specified location along axon to check for action potential
        detect_apc = fiber.apc[fiber.loc_index(ap_detect_location)]
        detect_time = None if detect_apc.n == 0 else round(detect_apc.time, precision)
        if detect_time:
            assert detect_time > 0, 'Action potentials at t<=0 should be impossible'
        if np.any([apc.n > 0 for apc in fiber.apc]) and not bool(detect_apc.n):
            warnings.warn(
                "APs detected at locations other than the set detection location. "
                "This could mean your stimamp is high enough for virtual anode block.",
                stacklevel=2,
            )
        return detect_apc.n, detect_time

    @staticmethod
    def threshold_checker(
        fiber: Fiber,
        block: bool = False,
        ap_detect_location: float = 0.9,
        istim_delay: float = 0,
    ) -> bool:
        """Check if stimulation was above or below threshold.

        :param fiber: instance of fiber class
        :param block: true if BLOCK_THRESHOLD protocol, false otherwise
        :param ap_detect_location: is the location (decimal % of fiber length) where APs are detected for threshold
        :param istim_delay: the delay from the simulation start to the onset of the intracellular stimulation [ms]
        :return: True if stim was supra-threshold, False if sub-threshold
        """
        # Determine user-specified location along axon to check for action potential
        detect_apc = fiber.apc[fiber.loc_index(ap_detect_location)]
        if block:
            return detect_apc.time <= istim_delay  # check for block
        return bool(detect_apc.n)  # otherwise check for activation

    def find_threshold(  # noqa: C901 #TODO clean up and reduce complexity
        self: ScaledStim,
        fiber: Fiber,
        condition: ThresholdCondition = ThresholdCondition.ACTIVATION,
        bounds_search_mode: BoundsSearchMode = BoundsSearchMode.PERCENT_INCREMENT,
        bounds_search_step: float = 10,
        termination_mode: TerminationMode = TerminationMode.PERCENT_DIFFERENCE,
        termination_tolerance: float = 1,
        stimamp_top: float = -1,
        stimamp_bottom: float = -0.01,
        max_iterations: int = 100,
        exit_t_shift: float = 5,
        bisection_mean: BisectionMean = BisectionMean.ARITHMETIC,
        fail_on_end_excitation: bool = True,
        **kwargs,
    ) -> tuple[float | Any, tuple[float, float]]:
        """Bisection search to find threshold amplitudes. #TODO clean up this docstring.

        :param fiber: instance of Fiber class to apply stimulation to
        :param condition: condition to search for threshold (activation or block)
        :param bounds_search_mode: indicates how to change upper and lower bounds during initial search
        :param bounds_search_step: the incremental increase/decrease of the upper/lower bound in the initial search
            if bounds_search_mode is 'PERCENT_INCREMENT' this is the percentage increase/decrease,
            if bounds_search_mode is 'ABSOLUTE_INCREMENT' this is the absolute increase/decrease
        :param termination_mode: indicates when upper and lower bounds converge on a solution of appropriate precision
        :param termination_tolerance: difference between upper and lower bounds that indicates convergence
            if absolute difference if termination_mode is 'ABSOLUTE_DIFFERENCE',
            or percentage difference if termination_mode is 'PERCENT_DIFFERENCE'
        :param stimamp_top: the upper-bound stimulation amplitude first tested to establish search bounds
        :param stimamp_bottom: the lower-bound stimulation amplitude first tested to establish search bounds
        :param max_iterations: the maximum number of iterations for finding search bounds
        :param exit_t_shift: shift (in ms) for detected action potential time to exit subthreshold stimulation
        :param bisection_mean: the type of mean to use for bisection search (arithmetic or geometric)
        :param fail_on_end_excitation: if True, raise error if end excitation is detected
        :param kwargs: additional keyword arguments to pass to the run_sim method
        :raises RuntimeError: If stimamp bottom is supra-threshold and stimamp top is sub-threshold
        :raises ValueError: If stimamp bottom and stimamp top have different signs
        :raises ValueError: If stimamp top does not exceed stimamp bottom
        :return: the threshold amplitude for the given condition, and the number of detected aps
        """
        if abs(stimamp_top) < abs(stimamp_bottom):
            raise ValueError("stimamp_top must be greater than stimamp_bottom in magnitude.")
        if stimamp_top * stimamp_bottom < 0:
            raise ValueError("stimamp_top and stimamp_bottom must have the same sign.")
        if self.istim is not None and condition == ThresholdCondition.ACTIVATION:
            warnings.warn(
                "Intracellular stimulation is enabled for this ScaledStim instance, are you sure you want "
                "to search for activation threshold?",
                stacklevel=2,
            )
        if self.istim is None and condition == ThresholdCondition.BLOCK:
            warnings.warn(
                "Intracellular stimulation is not enabled for this ScaledStim instance, are you sure you want "
                "to search for block threshold?",
                stacklevel=2,
            )
        assert exit_t_shift is None or exit_t_shift > 0, 'exit_t_shift must be nonzero and positive'
        self.exit_t = float('Inf')

        # first check stimamps
        supra_top, t = self.threshsim(stimamp_top, fiber, check_threshold=condition, **kwargs)
        supra_bot, _ = self.threshsim(stimamp_bottom, fiber, check_threshold=condition, **kwargs)

        # Determine upper- and lower-bounds for simulation (prior to bisection search)
        iterations = 0
        while iterations < max_iterations:
            print(f'Search bounds: top={round(stimamp_top,3)}, bottom={round(stimamp_bottom,3)}')
            iterations += 1
            if supra_top and exit_t_shift:
                self.exit_t = t + exit_t_shift
                print(
                    f'Found AP at {t} ms, all future runs of this threshold search will exit at {self.exit_t} ms. '
                    'Can be changed by setting exit_t_shift.'
                )
            if not supra_bot and supra_top:  # noqa: R508
                break  # found search bounds
            elif supra_bot and not supra_top:
                raise RuntimeError(
                    "stimamp_bottom was found to be supra-threshold, "
                    "and stimamp_top was found to be subthreshold, which should be impossible."
                )
            elif not supra_bot and not supra_top:
                # search upward with stimamp top
                stimamp_bottom = stimamp_top
                if bounds_search_mode == BoundsSearchMode.ABSOLUTE_INCREMENT:
                    stimamp_top = stimamp_top + bounds_search_step
                elif bounds_search_mode == BoundsSearchMode.PERCENT_INCREMENT:
                    stimamp_top = stimamp_top * (1 + bounds_search_step / 100)
                supra_top, t = self.threshsim(stimamp_top, fiber, check_threshold=condition, **kwargs)
            elif supra_bot and supra_top:
                # search downward with stimamp bottom
                stimamp_top = stimamp_bottom
                if bounds_search_mode == BoundsSearchMode.ABSOLUTE_INCREMENT:
                    stimamp_bottom = stimamp_bottom - bounds_search_step
                elif bounds_search_mode == BoundsSearchMode.PERCENT_INCREMENT:
                    stimamp_bottom = stimamp_bottom * (1 - bounds_search_step / 100)
                supra_bot, _ = self.threshsim(stimamp_bottom, fiber, check_threshold=condition, **kwargs)
        else:
            raise RuntimeError(f"Reached maximum number of iterations ({max_iterations}) without finding threshold.")

        # Now that bounds are set, enter bisection search
        print('Beginning bisection search')
        while True:
            print(f'Search bounds: top={round(stimamp_top,3)}, bottom={round(stimamp_bottom,3)}')
            stimamp_prev = stimamp_top

            # calculate new stimamp
            if bisection_mean == BisectionMean.ARITHMETIC:
                stimamp = (stimamp_bottom + stimamp_top) / 2
            elif bisection_mean == BisectionMean.GEOMETRIC:
                stimamp = np.sign(stimamp_top) * (stimamp_bottom * stimamp_top) ** (1 / 2)

            suprathreshold, _ = self.threshsim(stimamp, fiber, check_threshold=condition, **kwargs)

            if termination_mode == TerminationMode.PERCENT_DIFFERENCE:
                thresh_resoln = abs(termination_tolerance / 100)
                tolerance = abs((stimamp_bottom - stimamp_top) / stimamp_top)
            elif termination_mode == TerminationMode.ABSOLUTE_DIFFERENCE:
                thresh_resoln = abs(termination_tolerance)
                tolerance = abs(stimamp_bottom - stimamp_top)

            # Check to see if stimamp is at threshold
            if tolerance < thresh_resoln:
                if not suprathreshold:
                    stimamp = stimamp_prev
                print('Threshold found at stimamp =', round(stimamp, 3))
                print('Validating threshold...')
                # Run one more time at threshold to save run variables, get n_aps, and confirm above threshold
                if 'istim_delay' in kwargs:
                    kwargs.pop('istim_delay')  # remove istim_delay if present
                n_aps, aptime = self.run_sim(stimamp, fiber, **kwargs)
                assert self.threshold_checker(
                    fiber, **kwargs
                ), 'Threshold stimulation did not generate an action potential'
                self.end_excitation_checker(fiber, fail_on_end_excitation=fail_on_end_excitation)
                break
            elif suprathreshold:
                stimamp_top = stimamp
            elif not suprathreshold:
                stimamp_bottom = stimamp

        return stimamp, (n_aps, aptime)

    def supra_exit(self: ScaledStim, fiber: Fiber) -> bool:
        """Exit simulation if threshold is reached, activation searches only.

        :param fiber: Fiber object to check for threshold
        :return: True if threshold is reached, False otherwise
        """
        return self.threshold_checker(fiber)

    @staticmethod
    def end_excitation_checker(
        fiber: Fiber, multi_site_check: bool = True, fail_on_end_excitation: bool = True
    ) -> bool_:
        """Check for end excitation.

        :param fiber: Fiber object to check for end excitation
        :param multi_site_check: If True, warn if multiple activation sites are detected
        :param fail_on_end_excitation: If True, raise error if end excitation is detected
        :raises RuntimeError: If end excitation is detected and fail_on_end_excitation is True
        :return: True if end excitation occurs, error otherwise
        """
        # get times of apc
        times = np.array([apc.time for apc in fiber.apc])
        times[np.where(times == 0)] = float('Inf')

        # find number of local minima in the aploc_data
        init_nodes = argrelextrema(times, np.less)[0]

        # check if aps initiated in multiple places
        if len(init_nodes) > 1 and multi_site_check:
            warnings.warn('Found multiple activation sites.', RuntimeWarning, stacklevel=2)

        end_excitation = np.any(init_nodes <= 1) or np.any(init_nodes >= len(times - 2))

        if end_excitation and fail_on_end_excitation:
            raise RuntimeError('End excitation occurred.')
        return end_excitation

    def threshsim(
        self: ScaledStim,
        stimamp: float,
        fiber: Fiber,
        check_threshold: ThresholdCondition = ThresholdCondition.ACTIVATION,
        ap_detect_location: float = 0.9,
        istim_delay: int = 0,
        *args,
        **kwargs,
    ) -> tuple[int, float]:
        """Run a simulation with a given stimulus amplitude and check for threshold.

        :param stimamp: the stimulus amplitude
        :param fiber: the fiber to stimulate
        :param check_threshold: the condition to check for threshold
        :param ap_detect_location: the location to detect action potentials
        :param istim_delay: the delay of the stimulus
        :param args: additional arguments to pass to the run_sim method
        :param kwargs: additional keyword arguments to pass to the run_sim method
        :return: Number of detected aps and time of last detected ap
        :raises NotImplementedError: if threshold condition is not implemented
        """
        if check_threshold == ThresholdCondition.ACTIVATION:  # noqa: R505
            n_aps, aptime = self.run_sim(
                stimamp, fiber, *args, exit_func=self.supra_exit, use_exit_t=True, **kwargs
            )  # type: ignore
            return self.threshold_checker(fiber, ap_detect_location=ap_detect_location), aptime
        elif check_threshold == ThresholdCondition.BLOCK:
            n_aps, aptime = self.run_sim(stimamp, fiber, *args, **kwargs)
            return (
                self.threshold_checker(
                    fiber, ap_detect_location=ap_detect_location, block=True, istim_delay=istim_delay
                ),
                aptime,
            )
        else:
            raise NotImplementedError(f'{check_threshold} is not implemented')
