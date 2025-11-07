"""Defines classes for running simulations using model fibers.

This module provides classes and functionalities to manage stimulation
of model fibers. It includes enumerations for different threshold,
termination, and bounds search modes, as well as a base :class:`Stimulation`
class and the more specialized :class:`ScaledStim` class for
extracellular stimulation and :class:`IntraStim` class for intracellular stimulation.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from enum import Enum, unique
from typing import TYPE_CHECKING

import numpy as np
from neuron import h
from scipy.signal import find_peaks

if TYPE_CHECKING:
    from pyfibers import Fiber

h.load_file('stdrun.hoc')

# Set up module-level logger
logger = logging.getLogger(__name__)

### Enumerators to define configuration options ### noqa: E266
# In Python 3.11+, can instead directly use StrEnum instead of inheriting str
# For compatibility with 3.10, using Enum and inheriting str


@unique
class ThresholdCondition(str, Enum):
    """Different threshold search conditions.

    - :attr:`ThresholdCondition.ACTIVATION`:
        Search for the minimal stimulus amplitude required to generate an action potential.
    - :attr:`ThresholdCondition.BLOCK`:
        Search for the stimulus amplitude required to block propagation (i.e., prevent action potentials).
    """

    ACTIVATION = "activation"
    BLOCK = "block"


@unique
class BoundsSearchMode(str, Enum):
    """Modes for adjusting bounds in the bounds search phase of finding threshold.

    - :attr:`BoundsSearchMode.PERCENT_INCREMENT`: Adjust bounds by multiplying/dividing by a percentage factor.
    - :attr:`BoundsSearchMode.ABSOLUTE_INCREMENT`: Adjust bounds by adding/subtracting a fixed increment.
    """

    PERCENT_INCREMENT = "percent"
    ABSOLUTE_INCREMENT = "absolute"


@unique
class TerminationMode(str, Enum):
    """Modes for determining when to terminate bisection search phase of finding threshold.

    - :attr:`TerminationMode.PERCENT_DIFFERENCE`: Convergence is based on percentage difference between bounds.
    - :attr:`TerminationMode.ABSOLUTE_DIFFERENCE`: Convergence is based on the absolute difference between bounds.
    """

    PERCENT_DIFFERENCE = "percent"
    ABSOLUTE_DIFFERENCE = "absolute"


@unique
class BisectionMean(str, Enum):
    """Mean type used during bisection search phase of finding threshold.

    - :attr:`BisectionMean.GEOMETRIC`: Use geometric mean (sqrt(bottom * top)).
    - :attr:`BisectionMean.ARITHMETIC`: Use arithmetic mean ((bottom + top) / 2).
    """

    GEOMETRIC = "geometric"
    ARITHMETIC = "arithmetic"


class Stimulation:
    """Manage stimulation of NEURON simulations.

    Provides methods to configure time stepping, run simulations, and perform threshold search.
    This class is not meant to be used as is; subclasses should override the :meth:`Stimulation.run_sim` method.
    For example, the :class:`ScaledStim` subclass provides extracellular stimulation capabilities.

    For more details on using this class to write custom stimulation routines,
    see :doc:`/custom_stim`.
    """

    def __init__(
        self: Stimulation,
        dt: float = 0.001,
        tstop: float = 50,
        t_init_ss: float = -200,
        dt_init_ss: float = 5,
        custom_run_sim: Callable[..., tuple[int, float | None]] | None = None,
    ) -> None:
        """Initialize the Stimulation class.

        :param dt: Time step for the simulation (ms).
        :param tstop: Total duration of the simulation (ms).
        :param t_init_ss: Start time (<=0) used to let the system reach steady-state before t=0.
        :param dt_init_ss: Larger time step used during the steady-state period (ms).
        :param custom_run_sim: Custom simulation function provided by the user; otherwise,
            the subclass must override :meth:`Stimulation.run_sim`.
        :raises RuntimeError: If NEURON sections/fibers have not been created yet.
        :ivar dt: Time step for the simulation (ms).
        :ivar tstop: Total duration of the simulation (ms).
        :ivar t_init_ss: Start time (<=0) used to let the system reach steady-state before t=0 (ms).
        :ivar dt_init_ss: Larger time step used during the steady-state period (ms).
        :ivar custom_run_sim: Custom simulation function provided by the user.
        :ivar time: NEURON :class:`Vector <neuron:Vector>` recording the global simulation time.
        """
        self.dt = dt
        self.tstop = tstop
        self.t_init_ss = t_init_ss
        self.dt_init_ss = dt_init_ss
        self._exit_t: float = None
        self.custom_run_sim = custom_run_sim  # Store the custom run_sim function
        self._n_timesteps = int(self.tstop / self.dt)
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
            f'{type(self).__name__}: {self.tstop} ms (dt={self.dt} ms)'
            f' (t_init_ss={self.t_init_ss} ms, dt_init_ss={self.dt_init_ss} ms)'
        )

    def __repr__(self: Stimulation) -> str:
        """Return a string representation of the Stimulation instance."""  # noqa: DAR201
        return self.__str__()

    def _steady_state(self: Stimulation, fiber: Fiber) -> None:
        """Run from t_init_ss to 0 ms using a large dt to reach steady-state."""  # noqa: DAR101, DAR401
        # Begin simulation at t=t_init_ss
        h.t = -np.abs(self.t_init_ss)
        # Use large dt during steady-state period
        h.dt = self.dt_init_ss
        vs = [fiber(0.5).v]  # Record the membrane potential at the center of the fiber
        while h.t < 0:
            h.fadvance()
            vs.append(fiber(0.5).v)
        if np.abs(np.diff(vs))[-1] > 1:  # if vm was still changing at the end of steady state
            diff = vs[-1] - vs[-2]
            raise RuntimeError(
                "The steady-state period did not reach a stable Vm."
                f"The difference between the last and second-to-last Vm was {diff} mV. "
                "Check for oscillations in your Vm."
            )
        if vs[-1] - vs[0] > 1:  # starts out at v_rest, fiber might not be properly specified
            diff = vs[-1] - vs[0]
            if not np.isclose(fiber.v_rest, vs[-1], atol=0.01):
                raise RuntimeError(
                    f"Fiber model rest potential ({fiber.v_rest} mV) and "
                    f"steady-state potential ({vs[-1]} mV) are different."
                )
            raise RuntimeError(
                f"Fiber model rest potential is specified as {fiber.v_rest} mV,"
                f"but the transmembrane potential at the end of the steady-state period was {vs[-1]} mV."
            )
        # Restore the smaller dt
        h.dt = self.dt
        # Reset simulation time to 0
        h.t = 0
        h.fcurrent()
        h.frecord_init()

    @staticmethod
    def _initialize_extracellular(fiber: Fiber) -> None:
        """Set extracellular potentials to zero along the fiber.

        :param fiber: The :class:`~pyfibers.fiber.Fiber` object to initialize.
        """
        for section in fiber.sections:
            section(0.5).e_extracellular = 0

    @staticmethod
    def _update_extracellular(fiber: Fiber, e_stims: np.ndarray) -> None:
        """Apply a set of extracellular potential values along the fiber.

        :param fiber: The :class:`~pyfibers.fiber.Fiber` object to stimulate.
        :param e_stims: List or array of potential values, one per fiber section.
        """
        for x, section in enumerate(fiber.sections):
            section(0.5).e_extracellular = e_stims[x]

    def run_sim(self: Stimulation, *args, **kwargs) -> tuple[int, float | None]:
        """Run a simulation using either a custom_run_sim method or a subclass override."""  # noqa: DAR
        if self.custom_run_sim:
            return self.custom_run_sim(self, *args, **kwargs)
        raise NotImplementedError(
            "The run_sim method must be overridden by the subclass or provided as a custom function."
        )

    def pre_run_setup(self: Stimulation, fiber: Fiber, ap_detect_threshold: float = -30) -> None:
        """Prepare the simulation environment before running.

        This method sets the temperature, initializes the membrane potential,
        configures AP detection, and optionally balances the membrane currents
        for certain fiber models (e.g., Tigerholm). It also applies any
        intracellular stimulation parameters if provided.

        :param fiber: The :class:`~pyfibers.fiber.Fiber` object for which the simulation will be configured.
        :param ap_detect_threshold: The voltage threshold for detecting action potentials (mV).
        """
        # reassign time recorder
        # without this, time recording can get messed up for unclear reasons
        fiber.time = self.time = h.Vector().record(h._ref_t)
        # recompute timesteps
        self._n_timesteps = int(self.tstop / self.dt)
        # Set simulation temperature based on the fiber's temperature
        h.celsius = fiber.temperature
        # Initialize the simulation to the fiber's rest potential
        h.finitialize(fiber.v_rest)
        # Set up AP detectors at each node
        fiber.apcounts(thresh=ap_detect_threshold)
        self._initialize_extracellular(fiber)  # Set extracellular stimulation at each segment to zero
        self._steady_state(fiber)  # Allow system to reach steady-state before simulation

    @staticmethod
    def ap_checker(
        fiber: Fiber,
        ap_detect_location: float = 0.9,
        precision: int = 3,
        check_all_apc: bool = True,
    ) -> tuple[int, float | None]:
        """Check how many action potentials occurred at a specified node location.

        :param fiber: The :class:`~pyfibers.fiber.Fiber` object to evaluate for APs.
        :param ap_detect_location: Normalized location along the fiber in [0,1] to check for APs.
        :param precision: Decimal places to round the detected AP time.
        :param check_all_apc: If True, raise a warning if APs occur elsewhere but not at the detect location.
        :return: A tuple (num_aps, last_ap_time). If no APs are detected, last_ap_time is None.
        :raises RuntimeError: If the detected AP time is non-positive.
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
        if detect_time and detect_time <= 0:
            raise RuntimeError("Action potentials at t<=0 are unexpected.")

        # If requested, check for APs at other nodes
        if check_all_apc and not bool(detect_apc.n) and np.any([apc.n > 0 for apc in fiber.apc]):
            warnings.filterwarnings(
                "always",
                message="APs detected at locations other than the set detection location. "
                "This could mean your stimamp is high enough for virtual anode block.",
            )
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

        :param fiber: The :class:`~pyfibers.fiber.Fiber` object to evaluate.
        :param ap_detect_location: Normalized location in [0,1] where APs are detected.
        :param block: If True, check for block threshold; otherwise, check for activation threshold.
        :param block_delay: Time after simulation start to check for block (ms).
        :param thresh_num_aps: Number of APs that constitutes a suprathreshold response.
        :param check_all_apc: Passed to :meth:`Stimulation.ap_checker` for additional warnings.
        :return: True if stimulation is suprathreshold; False if subthreshold.
        :raises ValueError: If thresh_num_aps is not positive.
        :raises NotImplementedError: If block is True and thresh_num_aps != 1.
        :raises RuntimeError: If no APs are detected at all in a block threshold search.
        """
        if thresh_num_aps <= 0:
            raise ValueError("thresh_num_aps must be positive.")
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

    def find_threshold(  # noqa: C901
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
        **kwargs,
    ) -> tuple[float, tuple[int, float | None]]:
        """Perform a bisection search to find the threshold stimulus amplitude.

        This method first locates bounds where one amplitude is subthreshold and
            another is suprathreshold (bounds search phase). Then, it repeatedly narrows the bounds until they
            converge based on the specified termination mode and tolerance (bisection search phase).
            Ideally, the initial bounds should be chosen such that
            stimamp_top is supra-threshold and stimamp_bottom is sub-threshold.

        Note that enums (:class:`ThresholdCondition`, :class:`BoundsSearchMode`,
            :class:`TerminationMode`, :class:`BisectionMean`) can be
            provided as an enum member (e.g. :attr:`ThresholdCondition.ACTIVATION`) or as
            the member's string value (e.g. "activation").

        .. seealso::

            For more details on the threshold search process, see :doc:`/algorithms`.

        :param fiber: Instance of :class:`pyfibers.fiber.Fiber` to apply stimulation to.
        :param condition: The threshold condition
            (:attr:`ThresholdCondition.ACTIVATION` or :attr:`ThresholdCondition.BLOCK`).
        :param bounds_search_mode: The bounds search mode
            (:attr:`BoundsSearchMode.PERCENT_INCREMENT` or :attr:`BoundsSearchMode.ABSOLUTE_INCREMENT`).
        :param bounds_search_step: The iterative increase/decrease of the upper/lower bound during bounds search
            - if bounds_search_mode is "percent" this is the percentage increase/decrease
            - if bounds_search_mode is "absolute" this is the absolute increase/decrease
        :param termination_mode: The termination mode
            (:attr:`TerminationMode.PERCENT_DIFFERENCE` or :attr:`TerminationMode.ABSOLUTE_DIFFERENCE`).
        :param termination_tolerance: Difference between upper and lower bounds that indicates convergence
            - absolute difference if termination_mode is "absolute"
            - percentage difference if termination_mode is "percent"
        :param stimamp_top: Initial upper-bound stimulus amplitude to test.
        :param stimamp_bottom: Initial lower-bound stimulus amplitude to test.
        :param max_iterations: Maximum attempts to find bounding amplitudes before bisection.
        :param exit_t_shift: Extra time (ms) after an AP is detected, beyond which the simulation can be cut short.
        :param bisection_mean: The bisection mean type
            (:attr:`BisectionMean.ARITHMETIC` or :attr:`BisectionMean.GEOMETRIC`).
        :param block_delay: Time (ms) after start to check for a blocked AP, used in block searches.
        :param thresh_num_aps: Number of action potentials for threshold search
            - if activation, suprathreshold requires detected aps >= thresh_num_aps
            - if block, suprathreshold requires detected aps < thresh_num_aps
        :param kwargs: Additional arguments passed to the run_sim method.
        :return: A tuple (threshold_amplitude, (num_detected_aps, last_detected_ap_time)).
        :raises ValueError: If invalid enum values are provided for
            condition, bounds_search_mode, termination_mode, or bisection_mean.
        :raises RuntimeError: If contradictory bounding conditions occur or if the search fails to converge.
        """
        # Handle deprecated silent parameter
        if 'silent' in kwargs:
            warnings.warn(
                "The 'silent' parameter is deprecated and will be removed in a future version. "
                "Use pyfibers.enable_logging() to control logging output instead.",
                FutureWarning,
                stacklevel=2,
            )
            # Remove it from kwargs to avoid passing it to run_sim
            kwargs.pop('silent')

        self._validate_threshold_args(condition, stimamp_top, stimamp_bottom, exit_t_shift, fiber)
        # Validate enums. Using "in" directly on enum requires Python 3.12+, so using list comp instead
        if condition not in [mem.value for mem in ThresholdCondition]:
            raise ValueError("Invalid threshold condition.")
        if bounds_search_mode not in [mem.value for mem in BoundsSearchMode]:
            raise ValueError("Invalid bounds search mode.")
        if termination_mode not in [mem.value for mem in TerminationMode]:
            raise ValueError("Invalid termination mode.")
        if bisection_mean not in [mem.value for mem in BisectionMean]:
            raise ValueError("Invalid bisection mean.")

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
            logger.info("Search bounds: top=%s, bottom=%s", round(stimamp_top, 6), round(stimamp_bottom, 6))
            iterations += 1

            # If top is supra-threshold, set an early exit time for activation searches
            if supra_top and exit_t_shift and condition == ThresholdCondition.ACTIVATION:
                self._exit_t = t + exit_t_shift
                logger.info(
                    "Found AP at %s ms, subsequent runs will exit at %s ms. " "Change 'exit_t_shift' to modify this.",
                    t,
                    self._exit_t,
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
        logger.info("Beginning bisection search")

        while True:
            logger.info("Search bounds: top=%s, bottom=%s", round(stimamp_top, 6), round(stimamp_bottom, 6))
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
                logger.info("Threshold found at stimamp = %s", round(stimamp, 6))
                logger.info("Validating threshold...")

                # Confirm the final run at the chosen amplitude
                n_aps, aptime = self.run_sim(stimamp, fiber, **kwargs)  # type: ignore
                # Check that it indeed triggers or blocks (i.e., is suprathreshold)
                if not self.threshold_checker(
                    fiber,
                    condition == ThresholdCondition.BLOCK,
                    kwargs.get("ap_detect_location", 0.9),
                    block_delay=block_delay,
                    thresh_num_aps=thresh_num_aps,
                ):
                    raise RuntimeError(
                        "Threshold stimulation did not generate the expected action potential condition."
                    )
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

        :param condition: Whether searching for activation or block threshold
             (:attr:`ThresholdCondition.ACTIVATION` or :attr:`ThresholdCondition.BLOCK`).
        :param stimamp_top: The initial upper-bound stimulus amplitude.
        :param stimamp_bottom: The initial lower-bound stimulus amplitude.
        :param exit_t_shift: Extra time added after detecting an AP in an activation threshold search.
        :param fiber: The :class:`~pyfibers.fiber.Fiber` object being stimulated.
        :raises ValueError: If stimamp_top and stimamp_bottom have different signs or invalid magnitudes.
        :raises ValueError: If exit_t_shift is not positive.
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
        if exit_t_shift is not None and exit_t_shift <= 0:
            raise ValueError("exit_t_shift must be nonzero and positive.")
        self._exit_t = float("Inf")

    def supra_exit(self: Stimulation, fiber: Fiber, ap_detect_location: float, thresh_num_aps: int = 1) -> bool:
        """Determine if simulation can be exited early, for activation threshold searches only.

        :param fiber: The :class:`~pyfibers.fiber.Fiber` object to check for an action potential.
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

        :param fiber: The :class:`~pyfibers.fiber.Fiber` object to check.
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
        if len(troughs) > 1 and multi_site_check:
            warnings.warn(
                "Multiple activation sites detected. "
                "(Can sometimes mean threshold is incorrect due to virtual anode block.)",
                RuntimeWarning,
                stacklevel=2,
            )

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
        :param fiber: The :class:`~pyfibers.fiber.Fiber` object to stimulate.
        :param condition: Threshold condition
            (:attr:`ThresholdCondition.ACTIVATION` or :attr:`ThresholdCondition.BLOCK`).
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


class IntraStim(Stimulation):
    """Manage intracellular stimulation of model fibers.

    This class extends the :class:`Stimulation` class to provide intracellular stimulation capabilities.
    The intracellular stimulation is managed via a custom :class:`h.trainIClamp` mechanism.
    This mechanism allows for repeated square pulses of current to be injected into a fiber.
    Its arguments are provided as ``clamp_kws`` when creating an instance of this class.

    **Example Usage**

    .. code-block:: python

        # Assuming you have already created a fiber "my_fiber"
        clamp_kws = {
            "delay": 1,  # ms
            "pw": 0.1,  # ms
            "dur": 50,  # ms
            "freq": 100,  # Hz
            "amp": 1,  # nA, recommended to set 1 for scaling purposes
        }
        istim_ind = 0  # Stimulate the first section
        dt, tstop = 0.001, 50
        istim = IntraStim(dt=dt, tstop=tstop, istim_ind=istim_ind, clamp_kws=clamp_kws)

        # Run a simulation with a given stimulation amplitude
        stimamp = 2
        istim.run_sim(2, my_fiber)

        # Run a threshold search
        istim.find_threshold(my_fiber, condition="activation")
    """

    def __init__(
        self: IntraStim,
        dt: float = 0.001,
        tstop: float = 50,
        t_init_ss: float = -200,
        dt_init_ss: float = 5,
        istim_ind: int = None,
        istim_loc: float = None,
        clamp_kws: dict = None,
    ) -> None:
        """Initialize IntracellularStim class.

        :param dt: time step for simulation [seconds]
        :param tstop: time step for simulation [seconds]
        :param t_init_ss: the time (<=0ms) for the system to reach steady state before starting the simulation [ms]
        :param dt_init_ss: the time step used to reach steady state [ms]
        :param istim_ind: the :class:`~pyfibers.fiber.Fiber` section index (unmyelinated) or
            node of Ranvier number (myelinated) receiving stimulation
        :param istim_loc: node location along the  :class:`~pyfibers.fiber.Fiber` (using NEURON style indexing)
        :param clamp_kws: keyword arguments for the :class:`h.trainIClamp`.
            All optional, default given in parentheses.
            - 'delay': (0) the delay from the start of the simulation to the onset of the intracellular stimulation [ms]
            - 'pw': (1) the pulse duration of the intracellular stimulation [ms]
            - 'dur': (50) the duration from the start of the simulation to the end of the intracellular stimulation [ms]
            - 'freq': (100) the intracellular pulse repetition rate [Hz]
            - 'amp': (1) the intracellular stimulation amplitude [nA]
            Note that amp is scaled by the stimamp parameter in run_sim.
        :ivar istim: the NEURON :class:`h.trainIClamp` object for intracellular stimulation
        :ivar istim_record: the NEURON :class:`Vector <neuron:Vector>` recording the intracellular stimulation current
        :ivar istim_ind: the :class:`~pyfibers.fiber.Fiber` section index or node of Ranvier
            number receiving stimulation
        :ivar istim_loc: the node location along the :class:`~pyfibers.fiber.Fiber` receiving stimulation
        :ivar istim_params: the dictionary of intracellular stimulation parameters (see :meth:`_add_istim`)
        :raises ValueError: If both istim_ind and istim_loc are specified, or if neither is specified.
        """
        super().__init__(dt=dt, tstop=tstop, t_init_ss=t_init_ss, dt_init_ss=dt_init_ss)
        self.istim: h.trainIClamp = None
        self.istim_record = None
        self.istim_ind, self.istim_loc = istim_ind, istim_loc
        if (istim_ind is None) == (istim_loc is None):
            raise ValueError('Must specify either ind or loc, but not both')
        self.istim_params = {'delay': 0, 'pw': 1, 'dur': 50, 'freq': 100, 'amp': 1}
        self.istim_params.update(clamp_kws or {})  # add user specified clamp parameters

    def _cleanup_istim(self: IntraStim) -> None:
        """Clean up existing trainIClamp objects.

        This method removes any existing trainIClamp objects to prevent accumulation
        when a fiber is used with other Stimulation class instances.
        """
        self.istim = None
        self.istim_record = None

    def _add_istim(self: IntraStim, fiber: Fiber) -> IntraStim:
        """Create instance of :class:`h.trainIClamp` for intracellular stimulation.

        This method is not called by the user, see :class:IntraStim.
        Note that trainIClamp is a mod file included in this package. It is an
        extension of NEURON's built-in IClamp that allows repeated square pulses.

        :param fiber: The :class:`~pyfibers.fiber.Fiber` object to attach intracellular stimulation to.
        :return: The Stimulation instance (self).

        :meta public:
        """
        if self.istim_ind is not None:  # If ind was provided, use it directly
            ind = self.istim_ind
        else:  # If loc was provided, convert it to an index
            ind = fiber.loc_index(self.istim_loc)

        # Warn if we're about to stimulate a passive end node
        if ind < fiber.passive_end_nodes or ind >= len(fiber.nodes) - fiber.passive_end_nodes:
            warnings.warn(f'Intracellularly stimulating passive node at index {ind}.', stacklevel=2)

        self.istim = h.trainIClamp(fiber[ind](0.5))
        # Record the stimulus current over time
        self.istim_record = h.Vector().record(self.istim._ref_i)

        self.istim.delay = self.istim_params['delay']  # Delay before the stimulation starts (ms)
        self.istim.PW = self.istim_params['pw']  # Duration of each pulse (ms)
        self.istim.train = self.istim_params['dur']  # Total duration over which pulses occur (ms)
        self.istim.freq = self.istim_params['freq']  # Pulse repetition rate (Hz)
        self.istim.amp = self.istim_params['amp']  # Stimulation amplitude (nA)

        return self

    def run_sim(
        self: IntraStim,
        stimamp: float,
        fiber: Fiber,
        ap_detect_location: float = 0.9,
        exit_func: Callable = lambda *x, **y: False,
        exit_func_interval: int = 100,
        exit_func_kws: dict = None,
        use_exit_t: bool = False,
        fail_on_end_excitation: bool = True,
        ap_detect_threshold: float = -30,
    ) -> tuple[int, float]:
        """Run a simulation for a single stimulation amplitude.

        :param stimamp: Amplitude to be applied to extracellular stimulation
            - Should be a single float for one source
            - If stimamp is a single float and there are multiple sources, the same stimamp is applied to all sources
            - If stimamp is a list of floats, each float is applied to the corresponding source
        :param fiber: The :class:`~pyfibers.fiber.Fiber` to be stimulated.
        :param ap_detect_location: Location to detect action potentials (percent along fiber)
        :param exit_func: Function to call to check if simulation should be exited
        :param exit_func_interval: Interval to call exit_func
        :param exit_func_kws: Keyword arguments to pass to exit_func
        :param use_exit_t: If True, use the time returned by exit_func as the simulation end time
        :param fail_on_end_excitation: Behavior for end excitation detection
            - if True, raise error if end excitation is detected
            - if False, continue simulation if end excitation is detected
            - if None, do not check for end excitation
        :param ap_detect_threshold: Threshold for detecting action potentials (default: -30 mV)
        :raises RuntimeError: If NaNs are detected in fiber potentials
        :return: Number of detected APs and time of last detected AP.
        """
        self._add_istim(fiber)  # type: ignore
        self.istim.amp *= stimamp
        self._validate_inputs(stimamp, fiber)
        logger.info('Running: %s', np.array(stimamp).round(6))

        self.pre_run_setup(fiber, ap_detect_threshold=ap_detect_threshold)

        exit_func_kws = exit_func_kws or {}

        # Run simulation
        for i in range(self._n_timesteps):
            h.fadvance()

            # check for NaNs in fiber potentials
            if np.any(np.isnan([s.v for s in fiber.sections])):
                raise RuntimeError('NaN detected in fiber potentials at t =', h.t)
            if i % exit_func_interval == 0 and exit_func(fiber, ap_detect_location, **exit_func_kws):
                break
            if use_exit_t and self._exit_t and h.t >= self._exit_t:
                break

        # get precision from the number of decimal places in self.dt
        precision = len(str(self.dt).split('.')[1])

        # check for end excitation. None means don't check at all
        if fail_on_end_excitation is not None:
            self.end_excitation_checker(fiber, fail_on_end_excitation=fail_on_end_excitation)
        n_ap, time = self.ap_checker(fiber, ap_detect_location=ap_detect_location, precision=precision)
        logger.info('N aps: %s, time %s', int(n_ap), time)

        # Clean up trainIClamp at the end of simulation
        self._cleanup_istim()

        return n_ap, time

    def _validate_inputs(self: IntraStim, stimamp: float, fiber: Fiber) -> None:
        """Validate inputs for intracellular stimulation.

        :param stimamp: The stimulus amplitude.
        :param fiber: The :class:`~pyfibers.fiber.Fiber` to stimulate.
        :raises ValueError: If fiber potentials are not zero.
        :raises RuntimeError: If intracellular stimulation is not enabled.
        :raises TypeError: If stimamp is not a float or int.
        """
        # recompute timesteps
        self._n_timesteps = int(self.tstop / self.dt)
        if not np.all(fiber.potentials == 0):
            raise ValueError('Fiber potentials must be zero for IntracellularStim')
        if self.istim is None:
            raise RuntimeError('Intracellular stimulation is not enabled.')
        if stimamp < 0:
            warnings.warn('Negative intracellular stimulation amplitude.', stacklevel=2)
        if not isinstance(stimamp, (int, float)):
            raise TypeError('stimamp must be a single float or int')


class ScaledStim(Stimulation):
    """Manage extracellular stimulation of model fibers.

    This class takes one or more waveforms, each of which is expected to match one corresponding
    set of fiber potentials (i.e., one source) in the fiber being stimulated.
    Therefore, if you have N potential sets on the fiber, you must provide N waveforms,
    each describing the time course of stimulation for that source.

    The waveform can be either:
        - A callable that accepts a single float argument for the stimulation time (in ms) and returns
    the waveform value at that time.
        - A list of such callables with length N if there are N sources

    .. seealso::

        For more information on defining potentials, see :doc:`/extracellular_potentials`.

    ** Example Usage **

    .. code-block:: python

        # Assuming you have already created a fiber "my_fiber"
        # Add potentials to the fiber
        my_fiber.potentials = electrical_potentials_array


        # Create function which takes in a time and returns a waveform value
        def my_square_pulse(t: float) -> float:
            if t > 0.1 and t <= 0.2:  # if time is during the pulse
                return 1  # on
            else:  # if time is outside the pulse
                return 0  # off


        # Same waveform as above but using scipy.interpolate.interp1d
        my_square_pulse = interp1d(
            [0, 0.1, 0.2, 50],  # start, on, off, stop, in ms
            [0, 1, 0, 0],  # waveform values at those times
            kind="previous",
        )

        # Create a :class:`ScaledStim` object
        dt, tstop = 0.001, 50
        stim = ScaledStim(waveform=my_square_pulse, dt=dt, tstop=tstop)

        # Run the simulation
        stim.run_sim(-1, my_fiber)

        # Calculate threshold
        stim.find_threshold(my_fiber, condition="activation")
    """

    def __init__(
        self: ScaledStim,
        waveform: np.ndarray | list[Callable[[float], float]] | Callable[[float], float],
        dt: float = 0.001,
        tstop: float = 50,
        t_init_ss: float = -200,
        dt_init_ss: float = 5,
        pad_waveform: bool = True,
        truncate_waveform: bool = True,
    ) -> None:
        """Initialize the ScaledStim class.

        :param waveform: Callable or list of callables (e.g., function). Each callable should accept
            a single float argument for the simulation time (in ms) and return the waveform value at
            that time. It is recommended that you provide unit waveforms (maximum absolute value of 1).
            Support for waveform as a numpy array is retained for backwards compatibility, and will be
            removed in a future release.
        :param dt: Main simulation time step (ms).
        :param tstop: Total simulation duration (ms).
        :param t_init_ss: Time (<=0) to reach steady-state prior to the main simulation.
        :param dt_init_ss: Large time step used during the steady-state period (ms).
        :param pad_waveform: If True, extend the waveform with zeros to match simulation time.
        :param truncate_waveform: If True, truncate the waveform if it exceeds the simulation time.
        :ivar waveform: The waveform(s) to be applied to the fiber potentials.
            See :class:`ScaledStim` for more info.
        :ivar pad: If True, extend the waveform with zeros to match simulation time.
        :ivar truncate: If True, truncate the waveform if it exceeds the simulation time.
        """
        super().__init__(dt, tstop, t_init_ss, dt_init_ss)
        self.pad = pad_waveform
        self.truncate = truncate_waveform
        self._n_timesteps: int = None
        self.waveform = waveform
        self._prep_waveform()

    def _prep_potentials(self: ScaledStim, fiber: Fiber) -> None:
        """Prepare the fiber's potentials for scaled stimulation.

        Ensures they are in a suitable 2D numpy array form, one row per
        potential set. Each row must match the length of the fiber coordinates.

        :param fiber: The :class:`~pyfibers.fiber.Fiber` object containing the potentials to be prepared.
        :raises ValueError: If no potentials are found or mismatch in lengths of fiber coordinates.
        """
        # This should be moved to the Fiber class as a setter method
        if fiber.potentials is None:
            raise ValueError("No fiber potentials found.")
        fiber.potentials = np.array(fiber.potentials)

        # If it's just one 1D array, wrap it in a list for stacking
        if isinstance(fiber.potentials, np.ndarray) and fiber.potentials.ndim == 1:
            fiber.potentials = [fiber.potentials]

        # Convert each potential to a np.array and check lengths
        processed_potentials = [np.array(potential) for potential in fiber.potentials]
        if not all(len(potential) == len(fiber.coordinates) for potential in processed_potentials):
            raise ValueError("Potential arrays must match the length of fiber.coordinates.")

        # Stack them into a 2D array
        fiber.potentials = np.vstack(processed_potentials)

    def _prep_waveform(self: ScaledStim) -> None:
        """Process user-provided waveform(s) to match the simulation length.

        Accepts single waveform (1D numpy array or single callable) or
        multiple waveforms (2D numpy array or list of callables),
        processes each waveform independently, and saves the processed waveform(s) as a 2D numpy array.
        Also checks if the waveform has a max absolute value of 1 (recommended).

        :raises ValueError: if any processed waveform row length is not equal to the number of time steps
        :raises TypeError: if a combination of callables and lists of floats are provided as waveforms
        :raises RuntimeError: if an error is encountered while processing a callable into an array
        """
        # recompute timesteps
        self._n_timesteps = int(self.tstop / self.dt)

        prepped_waveform = self.waveform
        # wrap waveform in a list in not already a list
        if not isinstance(prepped_waveform, list | np.ndarray):
            prepped_waveform = [prepped_waveform]

        if any(callable(wf) for wf in prepped_waveform):
            # if a mix of callables and arrays were specified, throw an error
            if not all(callable(wf) for wf in prepped_waveform):
                raise TypeError("Waveform must be specified as either a callable or a list of callables.")

            # process callable(s) into 2d array with shape (len(prepped_waveform), self._n_timesteps)
            try:
                prepped_waveform = np.fromfunction(
                    np.vectorize(lambda i, j: prepped_waveform[int(i)](j * self.dt)),
                    (len(prepped_waveform), self._n_timesteps),
                )
            except Exception as e:  # noqa: B902
                # provide some information on where the error happened to the user
                raise RuntimeError(
                    """Error encountered while processing callable into array. Does the callable produce
                    valid output for all time steps?"""
                ) from e
        else:
            # process waveform provided as array for backwards compatibility
            warnings.warn(
                """Specifying waveforms using lists/arrays is deprecated. """
                """Please specify as callable or list of callables""",
                FutureWarning,
                stacklevel=2,
            )
            prepped_waveform = np.array(prepped_waveform)

            # Check if waveform is a single 1D numpy array and wrap it in a list
            if prepped_waveform.ndim == 1:
                prepped_waveform = [prepped_waveform]

            # Initialize list to store processed waveforms
            processed_waveforms = []

            # Process each waveform
            for row in prepped_waveform:
                row = np.array(row)  # Ensure row is a numpy array

                if self.pad and (self._n_timesteps > len(row)):
                    # Extend waveform row until it is of length tstop/dt
                    if row[-1] != 0:
                        warnings.warn("Padding a waveform that does not end with 0.", stacklevel=2)
                    row = np.hstack([row, [0] * (self._n_timesteps - len(row))])

                if self.truncate and (self._n_timesteps < len(row)):
                    # Truncate waveform row until it is of length tstop/dt
                    if any(row[self._n_timesteps :]):
                        warnings.warn("Truncating waveform removed non-zero values.", stacklevel=2)
                    row = row[: self._n_timesteps]

                if len(row) != self._n_timesteps:
                    raise ValueError("Processed waveform length must match the number of time steps (tstop / dt).")

                processed_waveforms.append(row)  # Append processed row to the list

            # Convert list of processed rows into a 2D numpy array
            prepped_waveform = np.vstack(processed_waveforms)

        # if max abs value is not 1, warn user
        if np.max(np.abs(prepped_waveform)) != 1:
            warnings.warn(
                'Waveform does not have a max absolute value of 1. '
                'This is recommended to simplify scaling of the waveform.',
                stacklevel=2,
            )

        # make sure that number of columns in the waveform matches the number of time steps
        if prepped_waveform.shape[1] != self._n_timesteps:
            raise RuntimeError("Processed waveform length must match the number of time steps (tstop / dt).")

        self._prepped_waveform = prepped_waveform

    def _potentials_at_time(
        self: ScaledStim, i: int, fiber: Fiber, stimamps: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.float64]:
        """Compute the total extracellular potential at time index i.

        Each row of the fiber's potentials is multiplied by the corresponding
        waveform value at time i and the corresponding stimamp, then summed.

        :param i: Current time index in the simulation.
        :param fiber: The :class:`~pyfibers.fiber.Fiber` object whose potentials are being scaled.
        :param stimamps: Array of amplitude scaling factors (one per waveform row).
        :return: 1D array of summed potentials along the fiber sections.
        """
        potentials = np.zeros(fiber.potentials.shape[1])
        # Multiply each potential row by the waveform and amplitude at this time
        for potential_set, waveform_row, amp in zip(fiber.potentials, self._prepped_waveform, stimamps, strict=True):
            potentials += amp * waveform_row[i] * potential_set
        return potentials

    def _validate_scaling_inputs(
        self: ScaledStim, fiber: Fiber, stimamps: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.float64]:
        """Validate scaling inputs before running simulation.

        :param fiber: Instance of :class:`~pyfibers.fiber.Fiber` to validate scaling inputs for.
        :param stimamps: Amplitude to be applied to extracellular stimulation.
        :return: Array of stimulation amplitudes to apply to each waveform.
        :raises ValueError: If validation checks fail for potentials, waveforms, or stimamps.
        """
        self._prep_waveform()
        self._prep_potentials(fiber)

        if len(fiber.potentials) != len(self._prepped_waveform):
            raise ValueError('Number of fiber potentials sets does not match number of waveforms')
        if np.all(fiber.potentials == 0):
            raise ValueError('Extracellular stimulation requires at least one non-zero fiber potential')
        if np.all(self._prepped_waveform == 0):
            raise ValueError('Extracellular stimulation requires at least one non-zero waveform')
        if len(stimamps.shape) == 0:  # if single float, apply to all sources
            return np.array([stimamps] * len(self._prepped_waveform))
        if len(stimamps) != len(self._prepped_waveform):
            raise ValueError("Number of stimamps must match the number of waveform rows.")

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
        fail_on_end_excitation: bool | None = True,
        ap_detect_threshold: float = -30,
    ) -> tuple[int, float]:
        """Run a simulation with a given stimulus amplitude(s).

        .. seealso::

            For more information on the underlying math, see the `Algorithms <algorithms>`_ docs page.

        :param stimamp: Amplitude to scale the product of extracellular potentials and waveform.
            - Should be a single float for one source
            - If stimamp is a single float and there are multiple sources, the same stimamp is applied to all sources
            - If stimamp is a list of floats, each float is applied to the corresponding source
        :param fiber: The :class:`~pyfibers.fiber.Fiber` object to stimulate.
        :param ap_detect_location: Normalized location in [0,1] to check for APs.
        :param exit_func: Callback to check if the simulation can be ended early (e.g., upon detection of an AP).
        :param exit_func_interval: How often (in time steps) to call exit_func.
        :param exit_func_kws: Additional arguments for exit_func.
        :param use_exit_t: If True, simulation will stop after self._exit_t (if set).
        :param fail_on_end_excitation: Behavior for end excitation detection
            - If True, raise error if end excitation is detected
            - If False, continue simulation if end excitation is detected
            - If None, do not check for end excitation
        :param ap_detect_threshold: Threshold for detecting action potentials (default: -30 mV)
        :raises RuntimeError: If NaNs are detected in membrane potentials or if required setup (e.g., istim) is missing.
        :return: Tuple (number_of_APs, time_of_last_AP).
        """
        stimamps = np.array(stimamp)
        logger.info("Running: %s", stimamps.round(6))

        stimamps = self._validate_scaling_inputs(fiber, stimamps)

        # Configure the simulation environment
        self.pre_run_setup(fiber, ap_detect_threshold=ap_detect_threshold)

        exit_func_kws = exit_func_kws or {}

        # Advance the simulation in small steps, updating extracellular potentials each time
        for i in range(self._n_timesteps):
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
        logger.info("N aps: %s, time %s", int(n_ap), time)
        return n_ap, time
