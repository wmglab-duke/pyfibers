"""Defines classes for running simulations using model fibers.

This module provides classes and functionalities to manage stimulation
of model fibers. It includes enumerations for different threshold,
termination, and bounds search modes, as well as a base ``Stimulation``
class and a more specialized ``ScaledStim`` class for
extracellular stimulation.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from enum import Enum, unique
from typing import TYPE_CHECKING

import numpy as np
from neuron import h
from scipy.signal import find_peaks

if TYPE_CHECKING:
    from pyfibers import Fiber

from pyfibers import FiberModel

h.load_file('stdrun.hoc')

### Enumerators to define configuration options ### noqa: E266
# In Python 3.11+, can instead directly use StrEnum instead of inheriting str
# For compatibility with 3.10, using Enum and inheriting str


@unique
class ThresholdCondition(str, Enum):
    """Different threshold search conditions.

    - ACTIVATION: Search for the minimal stimulus amplitude required to generate an action potential.
    - BLOCK: Search for the stimulus amplitude required to block propagation (i.e., prevent action potentials).
    """

    ACTIVATION = "activation"
    BLOCK = "block"


@unique
class BoundsSearchMode(str, Enum):
    """Modes for adjusting bounds in the bounds search phase of finding threshold.

    - PERCENT_INCREMENT: Adjust bounds by multiplying/dividing by a percentage factor.
    - ABSOLUTE_INCREMENT: Adjust bounds by adding/subtracting a fixed increment.
    """

    PERCENT_INCREMENT = "percent"
    ABSOLUTE_INCREMENT = "absolute"


@unique
class TerminationMode(str, Enum):
    """Modes for determining when to terminate bisection search phase of finding threshold.

    - PERCENT_DIFFERENCE: Convergence is based on percentage difference between bounds.
    - ABSOLUTE_DIFFERENCE: Convergence is based on the absolute difference between bounds.
    """

    PERCENT_DIFFERENCE = "percent"
    ABSOLUTE_DIFFERENCE = "absolute"


@unique
class BisectionMean(str, Enum):
    """Mean type used during bisection search phase of finding threshold.

    - GEOMETRIC: Use geometric mean (sqrt(bottom * top)).
    - ARITHMETIC: Use arithmetic mean ((bottom + top) / 2).
    """

    GEOMETRIC = "geometric"
    ARITHMETIC = "arithmetic"


@unique
class StimAmpTarget(str, Enum):
    """Target for applying the stimulus amplitude.

    - INTRACELLULAR: Scale the intracellular stim current.
    - EXTRACELLULAR: Scale the extracellular potentials.
    """

    INTRACELLULAR = "intracellular"
    EXTRACELLULAR = "extracellular"


class Stimulation:
    """Manage stimulation of NEURON simulations.

    Provides methods to configure time stepping, run simulations, and perform threshold search.
    This class is not meant to be used as is; subclasses should override the ``run_sim`` method.
    For example, the :class:`ScaledStim` subclass provides extracellular stimulation capabilities.

    For more details on using this class to write custom stimulation routines,
    see :doc:`/custom`.
    """

    def __init__(
        self: Stimulation,
        dt: float = 0.001,
        tstop: float = 50,
        t_init_ss: float = -200,
        dt_init_ss: float = 10,
        custom_run_sim: Callable[..., tuple[int, float | None]] | None = None,
    ) -> None:
        """Initialize the Stimulation class.

        :param dt: Time step for the simulation (ms).
        :param tstop: Total duration of the simulation (ms).
        :param t_init_ss: Start time (<=0) used to let the system reach steady-state before t=0.
        :param dt_init_ss: Larger time step used during the steady-state period (ms).
        :param custom_run_sim: Custom simulation function provided by the user; otherwise,
            the subclass must override ``run_sim()``.
        :raises RuntimeError: If NEURON sections/fibers have not been created yet.
        """
        self.dt = dt
        self.tstop = tstop
        self.t_init_ss = t_init_ss
        self.dt_init_ss = dt_init_ss
        self.istim: h.trainIClamp = None
        self.istim_record = None
        self.istim_params: dict[str, float | int | None] = {}
        self._exit_t: float = None
        self.custom_run_sim = custom_run_sim  # Store the custom run_sim function

        # Attempt to record the global simulation time
        try:
            self.time = h.Vector().record(h._ref_t)
        except RuntimeError:
            raise RuntimeError(
                "Could not set up time recording vector. Make sure you have created "
                "a fiber before initializing stimulation."
            )

    def __str__(self: Stimulation) -> str:
        """Return a brief string representation of the Stimulation instance."""  # noqa: DAR201
        return (
            f"{type(self).__name__}: {self.dt * self.tstop} ms (dt={self.dt} ms)"
            f" (t_init_ss={self.t_init_ss} ms, dt_init_ss={self.dt_init_ss} ms)"
            f" istim_params={self.istim_params}"
        )

    def __repr__(self: Stimulation) -> str:
        """Return a string representation of the Stimulation instance."""  # noqa: DAR201
        return self.__str__()

    def _add_istim(
        self: Stimulation,
        fiber: Fiber,
        delay: float = 0,
        pw: float = 0,
        dur: float = 0,
        freq: float = 0,
        amp: float = 0,
        ind: int | None = None,
        loc: float | None = None,
    ) -> Stimulation:
        """Create an instance of trainIClamp for intracellular stimulation.

        This method is not called by the user, instead use :meth:`set_intracellular_stim`.
        The parameters from this method inform the trainIClamp object created in the pre_run_setup phase.

        :param fiber: The Fiber object to attach intracellular stimulation to.
        :param delay: Delay before the stimulation starts (ms).
        :param pw: Duration of each pulse (ms).
        :param dur: Total duration over which pulses occur (ms).
        :param freq: Frequency of pulse trains (Hz).
        :param amp: Stimulation amplitude (nA).
        :param ind: Index of the fiber section/node to stimulate (mutually exclusive with loc).
        :param loc: Normalized location along the fiber in [0,1] (mutually exclusive with ind).
        :return: The Stimulation instance (self).
        :raises AssertionError: If both ind and loc are specified or both are None.
        """
        assert (ind is None) != (loc is None), "Must specify either ind or loc, but not both"
        # If loc was provided, convert it to an index
        ind = ind or fiber.loc_index(loc)

        # Warn if we're about to stimulate a passive end node
        if (ind == 0 or ind == len(fiber.sections) - 1) and fiber.passive_end_nodes is True:
            warnings.warn("You are trying to intracellularly stimulate a passive end node.", stacklevel=2)

        # Create the trainIClamp object in the specified section
        intracellular_stim = h.trainIClamp(fiber[ind](0.5))
        intracellular_stim.delay = delay
        intracellular_stim.PW = pw
        intracellular_stim.train = dur
        intracellular_stim.freq = freq
        intracellular_stim.amp = amp
        self.istim = intracellular_stim

        # Record the stimulus current over time
        self.istim_record = h.Vector().record(self.istim._ref_i)
        return self

    def _steady_state(self: Stimulation) -> None:
        """Advance the simulation from t_init_ss to 0 ms using a large dt to reach steady-state."""
        # Begin simulation at t=t_init_ss
        h.t = self.t_init_ss
        # Use large dt during steady-state period
        h.dt = self.dt_init_ss
        while h.t <= -self.dt_init_ss:
            h.fadvance()
        # Restore the smaller dt
        h.dt = self.dt
        # Reset simulation time to 0
        h.t = 0
        h.fcurrent()
        h.frecord_init()

    @staticmethod
    def _initialize_extracellular(fiber: Fiber) -> None:
        """Set extracellular potentials to zero along the fiber.

        :param fiber: The Fiber object to initialize.
        """
        for section in fiber.sections:
            section(0.5).e_extracellular = 0

    @staticmethod
    def _update_extracellular(fiber: Fiber, e_stims: np.ndarray) -> None:
        """Apply a set of extracellular potential values along the fiber.

        :param fiber: The Fiber object to stimulate.
        :param e_stims: List or array of potential values, one per fiber section.
        """
        for x, section in enumerate(fiber.sections):
            section(0.5).e_extracellular = e_stims[x]

    def run_sim(self: Stimulation, *args, **kwargs) -> tuple[int, float | None]:
        """Run a simulation using either a custom_run_sim method or a subclass override."""  # noqa DAR401, DAR201, DAR101
        if self.custom_run_sim:
            return self.custom_run_sim(self, *args, **kwargs)
        raise NotImplementedError(
            "The run_sim method must be overridden by the subclass or provided as a custom function."
        )

    def set_intracellular_stim(
        self: Stimulation,
        delay: float,
        pw: float,
        dur: float,
        freq: float,
        amp: float,
        ind: int | None = None,
        loc: float | None = None,
    ) -> None:
        """Update intracellular stimulation parameters (trainIClamp) without immediately creating it.

        The trainIClamp object is created during the pre_run_setup phase, not when this method is called.

        Note that trainIClamp is a mod file included in this package. It is an
        extension of NEURON's built-in IClamp class that allows repeated pulses.

        :param delay: Delay before the stimulation starts (ms).
        :param pw: Duration of each pulse (ms).
        :param dur: Total duration over which pulses occur (ms).
        :param freq: Frequency of pulses (Hz).
        :param amp: Stimulation amplitude (nA).
        :param ind: Index of the fiber section/node to stimulate (mutually exclusive with loc).
        :param loc: Normalized location along the fiber in [0,1] (mutually exclusive with ind).
        :raises AssertionError: If both ind and loc are specified or both are None.
        """
        assert (ind is None) != (loc is None), "Must specify either ind or loc, but not both"
        self.istim_params = {
            "delay": delay,
            "pw": pw,
            "dur": dur,
            "freq": freq,
            "amp": amp,
            "ind": ind,
            "loc": loc,
        }

    def pre_run_setup(self: Stimulation, fiber: Fiber, ap_detect_threshold: float = -30) -> None:
        """Prepare the simulation environment before running.

        This method sets the temperature, initializes the membrane potential,
        configures AP detection, and optionally balances the membrane currents
        for certain fiber models (e.g., Tigerholm). It also applies any
        intracellular stimulation parameters if provided.

        :param fiber: The Fiber object for which the simulation will be configured.
        :param ap_detect_threshold: The voltage threshold for detecting action potentials (mV).
        """
        # Set simulation temperature based on the fiber's temperature
        h.celsius = fiber.temperature
        # Initialize the simulation to the fiber's rest potential
        h.finitialize(fiber.v_rest)
        # Set up AP detectors at each node
        fiber.apcounts(thresh=ap_detect_threshold)

        # If the fiber uses the Tigerholm model, balance membrane currents first
        if fiber.fiber_model == FiberModel.TIGERHOLM:
            fiber.balance()

        # Initialize extracellular potentials at zero
        self._initialize_extracellular(fiber)
        # Run the steady-state phase
        self._steady_state()

        # If intracellular stimulus parameters were provided, apply them
        if self.istim_params:
            self._add_istim(fiber, **self.istim_params)  # type: ignore
        else:
            # Otherwise, ensure no existing istim object is carried over
            self.istim = None
            self.istim_record = None

    @staticmethod
    def ap_checker(
        fiber: Fiber,
        ap_detect_location: float = 0.9,
        precision: int = 3,
        check_all_apc: bool = True,
    ) -> tuple[int, float | None]:
        """Check how many action potentials occurred at a specified node location.

        :param fiber: The Fiber object to evaluate for APs.
        :param ap_detect_location: Normalized location along the fiber in [0,1] to check for APs.
        :param precision: Decimal places to round the detected AP time.
        :param check_all_apc: If True, raise a warning if APs occur elsewhere but not at the detect location.
        :return: A tuple (num_aps, last_ap_time). If no APs are detected, last_ap_time is None.
        :raises AssertionError: If the detected AP time is non-positive.
        """
        # Convert user-specified location to an integer node index
        ind = fiber.loc_index(ap_detect_location)

        # Warn if the chosen location is a passive node
        if "passive" in fiber[ind].name():
            warnings.warn(
                "Checking for action potentials on a passive node. This may yield erroneous results.",
                stacklevel=2,
            )

        # Access the APCount object for the chosen node
        detect_apc = fiber.apc[ind]
        detect_time = None if detect_apc.n == 0 else round(detect_apc.time, precision)

        # If an AP was detected, ensure its time is positive
        if detect_time:
            assert detect_time > 0, "Action potentials at t<=0 are unexpected."

        # If requested, check for APs at other nodes
        if check_all_apc and not bool(detect_apc.n) and np.any([apc.n > 0 for apc in fiber.apc]):
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
        block_delay: float = 0,
        thresh_num_aps: int = 1,
        check_all_apc: bool = True,
    ) -> bool:
        """Determine whether a stimulation was above or below threshold, for activation or block.

        :param fiber: The Fiber object to evaluate.
        :param block: If True, check for block threshold; otherwise, check for activation threshold.
        :param ap_detect_location: Normalized location in [0,1] where APs are detected.
        :param block_delay: Time after simulation start to check for block (ms).
        :param thresh_num_aps: Number of APs that constitutes a suprathreshold response.
        :param check_all_apc: Passed to ap_checker for additional warnings.
        :return: True if stimulation is suprathreshold; False if subthreshold.
        :raises NotImplementedError: If block is True and thresh_num_aps != 1.
        :raises RuntimeError: If no APs are detected at all in a block threshold search.
        """
        assert thresh_num_aps > 0, "thresh_num_aps must be positive."
        detect_n, detect_time = Stimulation.ap_checker(
            fiber, ap_detect_location=ap_detect_location, check_all_apc=check_all_apc
        )

        if block:
            if thresh_num_aps != 1:
                raise NotImplementedError(
                    "Block threshold only supports thresh_num_aps=1. NEURON APCount only records time of the last AP."
                )
            if detect_time is None:
                raise RuntimeError(
                    "No APs detected for block threshold. Possibly the intrinsic activity weight is too low, "
                    "or no excitation is triggered at all. Check block_delay and/or start time of intrinsic activity."
                )
            # If no APs occur after block_delay, we interpret that as a successful block (supra-threshold for block).
            return detect_time <= block_delay

        # If not a block search, check for activation (detect_n >= thresh_num_aps).
        return detect_n >= thresh_num_aps

    def find_threshold(  # noqa: C901 #TODO clean up and reduce complexity
        self: Stimulation,
        fiber: Fiber,
        condition: ThresholdCondition = ThresholdCondition.ACTIVATION,
        bounds_search_mode: BoundsSearchMode = BoundsSearchMode.PERCENT_INCREMENT,
        bounds_search_step: float = 10,
        termination_mode: TerminationMode = TerminationMode.PERCENT_DIFFERENCE,
        termination_tolerance: float = 1,
        stimamp_top: float = -1,
        stimamp_bottom: float = -0.01,
        max_iterations: int = 50,
        exit_t_shift: float = 5,
        bisection_mean: BisectionMean = BisectionMean.ARITHMETIC,
        block_delay: float = 0,
        thresh_num_aps: int = 1,
        silent: bool = False,
        **kwargs,
    ) -> tuple[float, tuple[int, float | None]]:
        """Perform a bisection search to find the threshold stimulus amplitude.

        This method first locates bounds where one amplitude is subthreshold and
            another is suprathreshold (bounds search phase). Then, it repeatedly narrows the bounds until they
            converge based on the specified termination mode and tolerance (bisection search phase).
            Ideally, the initial bounds should be chosen such that
            stimamp_top is supra-threshold and stimamp_bottom is sub-threshold.

        Note that enums (ThresholdCondition, BoundsSearchMode, TerminationMode, BisectionMean) can be
            provided as an enum member (e.g. ThresholdCondition.ACTIVATION) or as
            the member's string value (e.g. "activation").

        :param fiber: instance of Fiber class to apply stimulation to
        :param condition: condition to search for threshold ("activation" or "block")
        :param bounds_search_mode: indicates how to change upper and lower bounds during bounds_search
        :param bounds_search_step: the iterative increase/decrease of the upper/lower bound during bounds search
            - if bounds_search_mode is "percent" this is the percentage increase/decrease
            - if bounds_search_mode is "absolute" this is the absolute increase/decrease
        :param termination_mode: indicates when upper and lower bounds converge on a solution of appropriate precision
        :param termination_tolerance: difference between upper and lower bounds that indicates convergence
            - absolute difference if termination_mode is "absolute"
            - percentage difference if termination_mode is "percent"
        :param stimamp_top: Initial upper-bound stimulus amplitude to test.
        :param stimamp_bottom: Initial lower-bound stimulus amplitude to test.
        :param max_iterations: Maximum attempts to find bounding amplitudes before bisection.
        :param exit_t_shift: Extra time (ms) after an AP is detected, beyond which the simulation can be cut short.
        :param bisection_mean: Mean type for the bisection step ("arithmetic" or "geometric").
        :param block_delay: Time (ms) after start to check for a blocked AP, used in block searches.
        :param thresh_num_aps: number of action potentials for threshold search
            - if activation, suprathreshold requires detected aps >= thresh_num_aps
            - if block, suprathreshold requires detected aps < thresh_num_aps
        :param silent: If True, suppress print statements for the search process.
        :param kwargs: Additional arguments passed to the run_sim method.
        :return: A tuple (threshold_amplitude, (num_detected_aps, last_detected_ap_time)).
        :raises RuntimeError: If contradictory bounding conditions occur or if the search fails to converge.
        """
        self._validate_threshold_args(condition, stimamp_top, stimamp_bottom, exit_t_shift, fiber)
        # Validate enums. Using "in" directly on enum requires Python 3.12+, so using list comp instead
        assert condition in [mem.value for mem in ThresholdCondition], "Invalid threshold condition."
        assert bounds_search_mode in [mem.value for mem in BoundsSearchMode], "Invalid bounds search mode."
        assert termination_mode in [mem.value for mem in TerminationMode], "Invalid termination mode."
        assert bisection_mean in [mem.value for mem in BisectionMean], "Invalid bisection mean."

        # First test the initial top and bottom amplitudes
        supra_top, (_, t) = self.threshsim(
            stimamp_top, fiber, condition=condition, block_delay=block_delay, thresh_num_aps=thresh_num_aps, **kwargs
        )
        supra_bot, _ = self.threshsim(
            stimamp_bottom, fiber, condition=condition, block_delay=block_delay, thresh_num_aps=thresh_num_aps, **kwargs
        )

        # Begin the bounds search phase
        iterations = 0
        while iterations < max_iterations:
            if not silent:
                print(f"Search bounds: top={round(stimamp_top, 6)}, bottom={round(stimamp_bottom, 6)}")
            iterations += 1

            # If top is supra-threshold, set an early exit time for activation searches
            if supra_top and exit_t_shift and condition == ThresholdCondition.ACTIVATION:
                self._exit_t = t + exit_t_shift
                if not silent:
                    print(
                        f"Found AP at {t} ms, subsequent runs will exit at {self._exit_t} ms. "
                        "Change 'exit_t_shift' to modify this."
                    )

            if not supra_bot and supra_top:  # noqa: R508
                break  # Bounds are found
            elif supra_bot and not supra_top:
                # Contradictory bounds
                raise RuntimeError(
                    "stimamp_bottom is supra-threshold while stimamp_top is subthreshold, which is unexpected."
                )
            elif not supra_bot and not supra_top:
                # Increase top bound
                stimamp_bottom = stimamp_top
                if bounds_search_mode == BoundsSearchMode.ABSOLUTE_INCREMENT:
                    stimamp_top = stimamp_top + bounds_search_step
                else:
                    stimamp_top = stimamp_top * (1 + bounds_search_step / 100)
                supra_top, (_, t) = self.threshsim(
                    stimamp_top,
                    fiber,
                    condition=condition,
                    block_delay=block_delay,
                    thresh_num_aps=thresh_num_aps,
                    **kwargs,
                )
            elif supra_bot and supra_top:
                # Decrease bottom bound
                stimamp_top = stimamp_bottom
                if bounds_search_mode == BoundsSearchMode.ABSOLUTE_INCREMENT:
                    stimamp_bottom = stimamp_bottom - bounds_search_step
                else:
                    stimamp_bottom = stimamp_bottom * (1 - bounds_search_step / 100)
                supra_bot, _ = self.threshsim(
                    stimamp_bottom,
                    fiber,
                    condition=condition,
                    block_delay=block_delay,
                    thresh_num_aps=thresh_num_aps,
                    **kwargs,
                )
        else:
            # If we exit the while loop normally, we've hit max_iterations
            raise RuntimeError(
                f"Reached max_iterations={max_iterations} without valid bounds. "
                f"stimamp_top={stimamp_top} is {'supra' if supra_top else 'sub'}-threshold, "
                f"stimamp_bottom={stimamp_bottom} is {'supra' if supra_bot else 'sub'}-threshold. "
                "Typically this indicates a problem with the initial bounds or the stimulation setup. "
                "Check your starting bounds, this can commonly occur in activation threshold searches"
                "if the initial top bound is high enough for virtual anode block. "
                "For block threshold searches, this can occur if the top bound is high enough for re-excitation."
            )

        # Begin the bisection search phase
        if not silent:
            print("Beginning bisection search")

        while True:
            if not silent:
                print(f"Search bounds: top={round(stimamp_top, 6)}, bottom={round(stimamp_bottom, 6)}")
            stimamp_prev = stimamp_top

            # Compute the midpoint based on the chosen mean
            if bisection_mean == BisectionMean.ARITHMETIC:
                stimamp = (stimamp_bottom + stimamp_top) / 2
            else:  # GEOMETRIC
                stimamp = np.sign(stimamp_top) * (stimamp_bottom * stimamp_top) ** 0.5

            suprathreshold, _ = self.threshsim(
                stimamp, fiber, condition=condition, block_delay=block_delay, thresh_num_aps=thresh_num_aps, **kwargs
            )

            if termination_mode == TerminationMode.PERCENT_DIFFERENCE:
                thresh_resoln = abs(termination_tolerance / 100)
                tolerance = abs((stimamp_bottom - stimamp_top) / stimamp_top)
            else:  # ABSOLUTE_DIFFERENCE
                thresh_resoln = abs(termination_tolerance)
                tolerance = abs(stimamp_bottom - stimamp_top)

            # Convergence check
            if tolerance < thresh_resoln:
                if not suprathreshold:
                    stimamp = stimamp_prev
                if not silent:
                    print(f"Threshold found at stimamp = {round(stimamp, 6)}")
                    print("Validating threshold...")

                # Confirm the final run at the chosen amplitude
                n_aps, aptime = self.run_sim(stimamp, fiber, **kwargs)  # type: ignore
                # Check that it indeed triggers or blocks (i.e., is suprathreshold)
                assert self.threshold_checker(
                    fiber,
                    condition == ThresholdCondition.BLOCK,
                    kwargs.get("ap_detect_location", 0.9),
                    block_delay=block_delay,
                    thresh_num_aps=thresh_num_aps,
                ), "Threshold stimulation did not generate the expected action potential condition."
                break
            elif suprathreshold:
                stimamp_top = stimamp
            else:
                stimamp_bottom = stimamp

        return stimamp, (n_aps, aptime)

    def _validate_threshold_args(
        self: Stimulation,
        condition: ThresholdCondition,
        stimamp_top: float,
        stimamp_bottom: float,
        exit_t_shift: float | None,
        fiber: Fiber,
    ) -> None:
        """Check that threshold arguments are logically consistent.

        :param condition: Whether searching for activation or block threshold.
        :param stimamp_top: The initial upper-bound stimulus amplitude.
        :param stimamp_bottom: The initial lower-bound stimulus amplitude.
        :param exit_t_shift: Extra time added after detecting an AP in an activation threshold search.
        :param fiber: The Fiber object being stimulated.
        :raises ValueError: If stimamp_top and stimamp_bottom have different signs or invalid magnitudes.
        :raises AssertionError: If exit_t_shift is not positive.
        """
        if abs(stimamp_top) < abs(stimamp_bottom):
            raise ValueError("stimamp_top must be greater in magnitude than stimamp_bottom.")
        if stimamp_top * stimamp_bottom < 0:
            raise ValueError("stimamp_top and stimamp_bottom must have the same sign.")
        if fiber.stim is not None and condition == ThresholdCondition.ACTIVATION:
            warnings.warn(
                "This fiber has intrinsic activity; check if you truly want an activation threshold search.",
                stacklevel=2,
            )
        if fiber.stim is None and condition == ThresholdCondition.BLOCK:
            warnings.warn(
                "This fiber lacks intrinsic activity; a block threshold search may be meaningless.",
                stacklevel=2,
            )
        assert exit_t_shift is None or exit_t_shift > 0, "exit_t_shift must be nonzero and positive."
        self._exit_t = float("Inf")

    def supra_exit(self: Stimulation, fiber: Fiber, ap_detect_location: float, thresh_num_aps: int = 1) -> bool:
        """Determine if simulation can be exited early, for activation threshold searches only.

        :param fiber: The Fiber object to check for an action potential.
        :param ap_detect_location: Normalized location to check in [0,1].
        :param thresh_num_aps: Number of APs required to consider it suprathreshold.
        :return: True if the specified number of APs has occurred at or before this time.
        """
        return self.threshold_checker(
            fiber, ap_detect_location=ap_detect_location, thresh_num_aps=thresh_num_aps, check_all_apc=False
        )

    @staticmethod
    def end_excitation_checker(
        fiber: Fiber,
        multi_site_check: bool = True,
        fail_on_end_excitation: bool | None = True,
    ) -> bool:
        """Check for end-excitation.

        Determines activation sites by finding local minima in each node's AP time.
        If an AP is detected near in or adjacent to the passive end nodes, raise an error
        or issue a warning based on fail_on_end_excitation.

        :param fiber: The Fiber object to check.
        :param multi_site_check: If True, warn if multiple activation sites are detected.
        :param fail_on_end_excitation: Controls handling of end-excitation:
            - True: Raise RuntimeError if end-excitation is detected.
            - False: Only warn if end-excitation is detected.
            - None: Skip the check entirely.
        :return: True if end excitation is detected, False otherwise.
        :raises RuntimeError: If end excitation is detected and fail_on_end_excitation is True.
        """
        times = np.array([0] + [apc.time for apc in fiber.apc] + [0])
        times[np.where(times == 0)] = float("Inf")

        # Find troughs (local minima) in the negative times array
        troughs, edges = find_peaks(-times, plateau_size=(0, float("inf")))

        # Identify the node indices for each trough or plateau
        init_nodes = []
        for left_edge, right_edge in zip(edges["left_edges"], edges["right_edges"]):
            # Correct for the padding we added
            init_nodes += list(range(left_edge - 1, right_edge))
        init_nodes = np.array(init_nodes)

        # If more than one trough is found, we might have multiple activation sites
        if len(troughs) > 1 and multi_site_check:  # TODO, separate out from this function
            warnings.warn("Multiple activation sites detected.", RuntimeWarning, stacklevel=2)

        # Identify indices near the start or end of the fiber
        end_excited_nodes = init_nodes[
            (init_nodes <= int(fiber.passive_end_nodes)) | (init_nodes >= len(times) - int(fiber.passive_end_nodes) - 3)
        ]

        if len(end_excited_nodes) and fail_on_end_excitation is not None:
            if fail_on_end_excitation:
                raise RuntimeError(f"End excitation detected on fiber.nodes{end_excited_nodes}.")
            warnings.warn(f"End excitation detected on fiber.nodes{end_excited_nodes}.", stacklevel=2)

        return bool(len(end_excited_nodes))

    def threshsim(
        self: Stimulation,
        stimamp: float,
        fiber: Fiber,
        condition: ThresholdCondition = ThresholdCondition.ACTIVATION,
        block_delay: float = 0,
        thresh_num_aps: int = 1,
        **kwargs,
    ) -> tuple[bool, tuple[int, float | None]]:
        """Run a single stimulation trial at a given amplitude and check for threshold.

        :param stimamp: Stimulus amplitude to apply.
        :param fiber: The Fiber object to stimulate.
        :param condition: Threshold condition (ACTIVATION or BLOCK).
        :param block_delay: If condition=BLOCK, time after which AP detection is considered blocked (ms).
        :param thresh_num_aps: Number of APs required to be considered suprathreshold.
        :param kwargs: Additional arguments for the run_sim method.
        :return: A tuple (is_suprathreshold, (num_aps, last_ap_time)).
        """
        # Deactivate end-excitation check for intermediate threshold sims
        kwargs["fail_on_end_excitation"] = None

        if condition == ThresholdCondition.ACTIVATION:
            # Use supra_exit only for single-AP detection
            exit_func = self.supra_exit if thresh_num_aps == 1 else lambda *x, **y: False
            exit_func_kws = {"thresh_num_aps": thresh_num_aps}

            n_aps, aptime = self.run_sim(
                stimamp, fiber, exit_func=exit_func, use_exit_t=True, **kwargs, exit_func_kws=exit_func_kws
            )
            # Determine whether it is above threshold
            is_supra = self.threshold_checker(
                fiber, ap_detect_location=kwargs.get("ap_detect_location", 0.9), thresh_num_aps=thresh_num_aps
            )
            return is_supra, (n_aps, aptime)
        if condition == ThresholdCondition.BLOCK:  # noqa: R503
            # BLOCK condition
            n_aps, aptime = self.run_sim(stimamp, fiber, **kwargs)
            is_block = self.threshold_checker(
                fiber,
                ap_detect_location=kwargs.get("ap_detect_location", 0.9),
                block=True,
                block_delay=block_delay,
                thresh_num_aps=thresh_num_aps,
            )
            return is_block, (n_aps, aptime)


class ScaledStim(Stimulation):
    """Manage extracellular stimulation in NEURON simulations using custom waveform(s).

    # TODO add example usage

    A specialized class that applies user-provided waveform(s)
    (scaled by a specified stimulus amplitude(s)) as an extracellular stimulus.
    Waveforms can be padded or truncated to match simulation time,
    and the fiber's extracellular potentials can be scaled accordingly.
    """

    def __init__(
        self: ScaledStim,
        waveform: list[list[float]] | np.ndarray,
        dt: float = 0.001,
        tstop: float = 50,
        t_init_ss: float = -200,
        dt_init_ss: float = 10,
        pad_waveform: bool = True,
        truncate_waveform: bool = True,
    ) -> None:
        """Initialize the ScaledStim class.

        This class takes one or more waveforms (rows in the waveform array), each of which
        is expected to match one corresponding set of fiber potentials (i.e., one source)
        in the fiber being stimulated.
        Therefore, if you have N potential sets on the fiber, you must provide N rows in
        the waveform array. Each row describes the timecourse of stimulation for that source.

        The waveform can be either:
          - A 1D array of shape (time_steps,) if there is only one source.
          - A 2D array of shape (num_sources, time_steps) if there are multiple sources.

        For more information, see :doc:`/extracellular_potentials`.

        :param waveform: A list or array of amplitude values over time for the simulation.
        :param dt: Main simulation time step (ms).
        :param tstop: Total simulation duration (ms).
        :param t_init_ss: Time (<=0) to reach steady-state prior to the main simulation.
        :param dt_init_ss: Large time step used during the steady-state period (ms).
        :param pad_waveform: If True, extend the waveform with zeros to match simulation time.
        :param truncate_waveform: If True, truncate the waveform if it exceeds the simulation time.
        """
        super().__init__(dt, tstop, t_init_ss, dt_init_ss)
        self.waveform = np.array(waveform)
        self.pad = pad_waveform
        self.truncate = truncate_waveform
        self.n_timesteps: int = None
        self._prep_waveform()

    def _prep_potentials(self: ScaledStim, fiber: Fiber) -> None:
        """Prepare the fiber's potentials for scaled stimulation.

        Ensures they are in a suitable 2D numpy array form, one row per
        potential set. Each row must match the length of the fiber coordinates.

        :param fiber: The Fiber object containing the potentials to be prepared.
        :raises AssertionError: If no potentials are found or mismatch in lengths of fiber coordinates.
        """
        assert fiber.potentials is not None, "No fiber potentials found."
        fiber.potentials = np.array(fiber.potentials)

        # If it's just one 1D array, wrap it in a list for stacking
        if isinstance(fiber.potentials, np.ndarray) and fiber.potentials.ndim == 1:
            fiber.potentials = [fiber.potentials]

        # Convert each potential to a np.array and check lengths
        processed_potentials = [np.array(potential) for potential in fiber.potentials]
        assert all(
            len(potential) == len(fiber.coordinates) for potential in processed_potentials
        ), "Potential arrays must match the length of fiber.coordinates."

        # Stack them into a 2D array
        fiber.potentials = np.vstack(processed_potentials)

    def _prep_waveform(self: ScaledStim) -> None:
        """Process user-provided waveform(s) to match the simulation length.

        Also checks if the waveform has a max absolute value of 1 (recommended).
        """
        self.waveform = np.array(self.waveform)
        self.n_timesteps = int(self.tstop / self.dt)

        processed_waveforms = []

        # If it's a single 1D array, wrap it in a list
        if self.waveform.ndim == 1:
            self.waveform = [self.waveform]

        # Pad or truncate each waveform
        for row in self.waveform:
            row = np.array(row)
            if self.pad and (self.n_timesteps > len(row)):
                if row[-1] != 0:
                    warnings.warn("Padding a waveform that does not end with 0.", stacklevel=2)
                row = np.hstack([row, [0] * (self.n_timesteps - len(row))])

            if self.truncate and (self.n_timesteps < len(row)):
                if any(row[self.n_timesteps :]):
                    warnings.warn("Truncating waveform removed non-zero values.", stacklevel=2)
                row = row[: self.n_timesteps]

            assert (
                len(row) == self.n_timesteps
            ), "Processed waveform length must match the number of time steps (tstop / dt)."

            if np.max(np.abs(row)) != 1:
                warnings.warn(
                    "Waveform does not have a maximum absolute value of 1. " "This is recommended to simplify scaling.",
                    stacklevel=2,
                )
            processed_waveforms.append(row)

        # Combine into a 2D array
        self.waveform = np.vstack(processed_waveforms)

    def _potentials_at_time(self: ScaledStim, i: int, fiber: Fiber, stimamps: np.ndarray) -> np.ndarray:
        """Compute the total extracellular potential at time index i.

        Each row of the fiber's potentials is multiplied by the corresponding
        waveform value at time i and the corresponding stimamp, then summed.

        :param i: Current time index in the simulation.
        :param fiber: The Fiber object whose potentials are being scaled.
        :param stimamps: Array of amplitude scaling factors (one per waveform row).
        :return: 1D array of summed potentials along the fiber sections.
        """
        potentials = np.zeros(fiber.potentials.shape[1])
        # Multiply each potential row by the waveform and amplitude at this time
        for potential_set, waveform_row, amp in zip(fiber.potentials, self.waveform, stimamps, strict=True):
            potentials += amp * waveform_row[i] * potential_set
        return potentials

    def _validate_scaling_inputs(
        self: ScaledStim,
        fiber: Fiber,
        stimamp_target: StimAmpTarget,
        stimamps: np.ndarray,
    ) -> np.ndarray:
        """Validate inputs before applying scaled stimulation.

        Ensures that the waveforms and fiber potentials are prepared and that
        intracellular vs. extracellular usage is consistent with the data.

        :param fiber: The Fiber object to be stimulated.
        :param stimamp_target: Whether to apply stimamps to intracellular or extracellular stimulation.
        :param stimamps: Single amplitude or array of amplitudes for each waveform row.
        :return: Numpy array of stimulation amplitudes broadcasted to match the number of waveforms.
        :raises AssertionError: If waveform/fiber potentials are empty or if inconsistencies exist.
        """
        self._prep_waveform()
        self._prep_potentials(fiber)

        assert len(fiber.potentials) == len(
            self.waveform
        ), "Number of fiber potential rows must match number of waveform rows."

        # Intracellular stimulation requires all-zero potentials/waveforms
        if stimamp_target == StimAmpTarget.INTRACELLULAR:
            assert np.all(fiber.potentials == 0), "For intracellular stimulation, the fiber's potentials must be zero."
            assert np.all(self.waveform == 0), "For intracellular stimulation, the waveform must be zero."
            assert (
                len(stimamps.shape) == 0
            ), "Intracellular stimulation requires a single float for stimamp, not an array."

        # Extracellular stimulation requires nonzero potentials/waveforms
        elif stimamp_target == StimAmpTarget.EXTRACELLULAR:
            assert (
                not self.istim_params
            ), "Extracellular stimulation does not support simultaneously configured intracellular parameters."
            assert not np.all(
                fiber.potentials == 0
            ), "For extracellular stimulation, the fiber.potentials must be nonzero."
            assert not np.all(self.waveform == 0), "For extracellular stimulation, the waveform must be nonzero."

        # If single float is provided, broadcast to match the number of waveforms
        if len(stimamps.shape) == 0:
            return np.array([stimamps] * len(self.waveform))

        assert len(stimamps) == len(self.waveform), "Number of stimamps must match the number of waveform rows."
        return np.array(stimamps)

    def run_sim(
        self: ScaledStim,
        stimamp: float | list[float],
        fiber: Fiber,
        ap_detect_location: float = 0.9,
        exit_func: Callable[..., bool] = lambda *x, **y: False,
        exit_func_interval: int = 100,
        exit_func_kws: dict | None = None,
        use_exit_t: bool = False,
        stimamp_target: StimAmpTarget = StimAmpTarget.EXTRACELLULAR,
        fail_on_end_excitation: bool | None = True,
        ap_detect_threshold: float = -30,
    ) -> tuple[int, float]:
        """Run a simulation with a given stimulus amplitude and waveform scaling.

        :param stimamp: Amplitude to scale the waveform.
            Can be a single float or an array matching the number of waveforms.
        :param fiber: The Fiber object to stimulate.
        :param ap_detect_location: Normalized location in [0,1] to check for APs.
        :param exit_func: Callback to check if the simulation can be ended early (e.g., upon detection of an AP).
        :param exit_func_interval: How often (in time steps) to call exit_func.
        :param exit_func_kws: Additional arguments for exit_func.
        :param use_exit_t: If True, simulation will stop after self._exit_t (if set).
        :param stimamp_target: Whether to apply stimamp to intracellular or extracellular stimulation.
        :param fail_on_end_excitation: If True, raise error on end-excitation; False warns; None disables check.
        :param ap_detect_threshold: Threshold voltage for detecting action potentials (mV).
        :return: Tuple (number_of_aps, time_of_last_ap).
        :raises RuntimeError: If NaNs are detected in membrane potentials or if required setup (e.g., istim) is missing.
        """
        stimamps = np.array(stimamp)
        print("Running:", stimamps.round(6), end="")

        # Validate waveform and potential data, plus user config
        stimamps = self._validate_scaling_inputs(fiber, stimamp_target, stimamps)

        # Configure the simulation environment
        self.pre_run_setup(fiber, ap_detect_threshold=ap_detect_threshold)

        exit_func_kws = exit_func_kws or {}

        # If target is intracellular, scale the amplitude of istim directly
        if stimamp_target == StimAmpTarget.INTRACELLULAR:
            assert self.istim is not None, "Intracellular stimulation is requested, but no istim object is configured."
            if stimamps < 0:
                warnings.warn("Negative intracellular stimulation amplitude.", stacklevel=2)
            # Only one amplitude is expected for intracellular mode
            assert len(stimamps) == 1, "Intracellular stimulation expects a single amplitude."
            self.istim.amp *= stimamps
            # Set the main stimamp to zero so we don't apply it to extracellular potentials
            stimamp *= 0

        # Advance the simulation in small steps, updating extracellular potentials each time
        for i in range(self.n_timesteps):
            timestep_potentials = self._potentials_at_time(i, fiber, stimamps)
            self._update_extracellular(fiber, timestep_potentials)

            h.fadvance()

            # Check for invalid (NaN) voltages
            if np.any(np.isnan([s.v for s in fiber.sections])):
                raise RuntimeError("NaN detected in fiber potentials.")

            # Periodically call exit_func to see if we should stop early
            if i % exit_func_interval == 0 and exit_func(fiber, ap_detect_location, **exit_func_kws):
                break
            # If _exit_t was set (e.g., after first AP detection) and time has passed it, stop simulation
            if use_exit_t and self._exit_t and h.t >= self._exit_t:
                break

        # Optionally check for end excitation
        if fail_on_end_excitation is not None:
            self.end_excitation_checker(fiber, fail_on_end_excitation=fail_on_end_excitation)

        # Finally, count action potentials at the chosen location
        n_ap, time = self.ap_checker(fiber, ap_detect_location=ap_detect_location)
        print(f"\tN aps: {int(n_ap)}, time {time}")
        return n_ap, time
