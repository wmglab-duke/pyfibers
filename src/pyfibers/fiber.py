"""Defines the :class:`Fiber` class and helper functions for building fiber models.

This module provides functionality for building and simulating
both 1D and 3D fiber models in the NEURON environment.
"""

from __future__ import annotations

import logging
import math
import typing
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from nd_line.nd_line import nd_line
from neuron import h

h.load_file('stdrun.hoc')

# Set up module-level logger
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .model_enum import FiberModel


def build_fiber(
    fiber_model: FiberModel,
    diameter: float,
    length: float = None,
    n_sections: int = None,
    n_nodes: int = None,
    enforce_odd_nodecount: bool = True,
    **kwargs,
) -> Fiber:
    """Generate a 1D (straight) fiber model in NEURON.

    This function creates a model fiber as an instance of the :class:`Fiber` class
    using the specific subclass specified from the :class:`FiberModel` enumerator.
    with user-specified diameter and length
    (from one of: number of sections, number of nodes, or length in microns).
    Additional keyword arguments are forwarded to the fiber model class constructor.

    By default, the first section of the fiber is located at the origin (0, 0, 0),
    and the fiber extends along the z-axis in the positive direction. To change the fiber's location,
    the method :meth:`Fiber.set_xyz` can be used to translate the fiber along the x, y, or z axes.
    To create fibers along a custom path in 3D space, use :func:`build_fiber_3d` instead.

    :param fiber_model: A :class:`FiberModel` enumerator specifying the type of fiber to instantiate.
    :param diameter: The fiber diameter in micrometers (µm).
    :param length: The total length of the fiber in micrometers (µm), if defining by length.
    :param n_sections: The total number of sections for discretizing the fiber, if defining by sections.
    :param n_nodes: The total number of nodes along the fiber, if defining by nodes.
    :param enforce_odd_nodecount: If ``True``, ensure that the number of nodes is odd.
    :param kwargs: Additional arguments forwarded to the underlying fiber model class.
    :raises ValueError: If more than one among ``length``, ``n_sections``, or ``n_nodes`` is specified
    :raises ValueError: If ``is_3d`` is specified in kwargs.
    :raises RuntimeError: If node count does not match number of nodes.
    :return: A :class:`Fiber` class instance.

    **Example Usage**

    .. code-block:: python

        from PyFibers import build_fiber, FiberModel

        fiber = build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10, n_nodes=25)
    """
    # must provide one of length, n_sections, or n_nodes, and only one
    if sum(x is not None for x in [length, n_sections, n_nodes]) != 1:
        raise ValueError("Must provide exactly one of length, n_sections, or n_nodes")

    if "is_3d" in kwargs:
        raise ValueError("is_3d is set automatically, try using build_fiber_3d() instead")

    fiber_class = fiber_model.value

    fiber_instance = fiber_class(diameter=diameter, fiber_model=fiber_model, **kwargs)

    fiber_instance.generate(
        n_nodes=n_nodes, n_sections=n_sections, length=length, enforce_odd_nodecount=enforce_odd_nodecount
    )

    # Set up coordinates and potentials
    fiber_instance.coordinates = np.concatenate(
        (
            np.zeros((len(fiber_instance.longitudinal_coordinates), 1)),
            np.zeros((len(fiber_instance.longitudinal_coordinates), 1)),
            fiber_instance.longitudinal_coordinates.reshape(-1, 1),
        ),
        axis=1,
    )
    fiber_instance.potentials = np.zeros(len(fiber_instance.longitudinal_coordinates))
    fiber_instance.time = h.Vector().record(h._ref_t)

    if len(fiber_instance) != fiber_instance.nodecount:
        raise RuntimeError("Node count does not match number of nodes")

    if fiber_instance.diameter > 3 and not fiber_instance.myelinated:
        warnings.warn(
            "Unmyelinated fibers are typically <=3 µm in diameter. "
            f"Received D={fiber_instance.diameter:.2f} µm. Proceed with caution.",
            stacklevel=2,
        )

    # Set all sections to the rest potentials (important for balancing currents)
    h.finitialize(fiber_instance.v_rest)

    # if fiber has balance method and not already balanced, balance fiber
    if hasattr(fiber_instance, 'balance'):
        fiber_instance.balance()
        fiber_instance.balanced = True

    return fiber_instance


def build_fiber_3d(
    fiber_model: FiberModel,
    diameter: float,
    path_coordinates: np.ndarray,
    shift: float = 0,
    shift_ratio: float = None,
    center: bool = False,
    **kwargs,
) -> Fiber:
    """Generate a 3D fiber model in NEURON based on a specified path.

    This function calculates the fiber's length from the user-supplied ``path_coordinates``
    and uses it internally to instantiate a 3D fiber model. The coordinates are a 2D numpy array of shape
    (number_of_points, 3), where each row represents a point in 3D space (x, y, z).

    The fiber model will be created by
    repeating sections along the path until no more nodes can be added without exceeding the path length.
    By default, the center of the first section is placed at the origin (0, 0, 0), and the fiber extends
    along the path.

    **Shifting Behavior**: For 3D fibers, shifting is handled during fiber creation (not during potential resampling).
    This is because 3D fiber geometry represents the physical position of the fiber in space, so shifting
    affects the actual fiber coordinates. You can apply shifts using either:
    - ``shift``: a shift in microns, OR
    - ``shift_ratio``: a fraction of ``delta_z`` (the internodal length).

    If ``center=True``, the shift is applied after the fiber is first centered about the midpoint of the 3D path.
    To test different 3D fiber positions, create a new fiber with the desired shift parameters.

    :param fiber_model: A :class:`FiberModel` enumerator specifying the type of fiber to instantiate.
    :param diameter: The fiber diameter in micrometers (µm).
    :param path_coordinates: A numpy array of shape (N, 3) specifying the 3D coordinates (x, y, z) of the fiber path.
    :param shift: A shift in microns to apply to the fiber coordinates.
    :param shift_ratio: Ratio of the internodal length to shift the fiber coordinates.
    :param center: If True, center the fiber before applying the shift.
    :param kwargs: Additional arguments forwarded to the underlying fiber model class.
    :raises ValueError: If ``path_coordinates`` is not provided, or if ``n_sections``, ``n_nodes``, or ``length``
        is specified (these are invalid in 3D mode).
    :return: A fully instantiated 3D fiber model :class:`Fiber` instance.

    **Example**:

    .. code-block:: python

        import numpy as np
        from PyFibers import build_fiber_3d, FiberModel

        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 3.0],
                [0.0, 2.0, 7.0],
                # ...
            ]
        )
        fiber = build_fiber_3d(
            fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10, path_coordinates=coords
        )
        print(fiber)
    """
    if path_coordinates is None:
        raise ValueError("path_coordinates must be provided for 3D fibers")

    if 'n_sections' in kwargs or 'n_nodes' in kwargs or 'length' in kwargs:
        raise ValueError("For 3D fibers, cannot specify n_sections, n_nodes, or length")

    # Calculate the length from path_coordinates
    fiber_path = nd_line(path_coordinates)

    # Apply shift if relevant
    shifted_start = _shift_fiber(
        fiber_path.length,
        build_fiber(  # call build_fiber to get delta_z
            fiber_model=fiber_model,
            diameter=diameter,
            length=fiber_path.length,
            **kwargs,
        ).delta_z,
        shift,
        shift_ratio,
        center,
    )

    # Create the fiber instance using the base class method
    fiber_instance = build_fiber(
        fiber_model=fiber_model,
        diameter=diameter,
        length=fiber_path.length - shifted_start,
        **kwargs,
    )

    # Make the 3D fiber coordinates an intrinsic property of the Fiber object.
    fiber_instance._set_3d()
    fiber_instance.coordinates = np.array(
        [fiber_path.interp(p + shifted_start) for p in fiber_instance.longitudinal_coordinates]
    )
    fiber_instance.path = fiber_path

    return fiber_instance


def _shift_fiber(
    fiber_length: float,
    delta_z: float,
    shift: float | None = None,
    shift_ratio: float | None = None,
    center: bool = False,
) -> float:
    """Shift a fiber's coordinates along its length by a specified amount.

    This is a unified helper function used by both 1D and 3D fiber shifting operations.
    The function calculates where to start placing fiber sections along a path or coordinate system.

    You can specify either:
    - ``shift``: a shift in microns, OR
    - ``shift_ratio``: a fraction of ``delta_z`` (the internodal length).

    If the shift or shift_ratio exceed the internodal length (delta_z),
    the extra lengths will be removed (only the modulus is applied).

    If ``center=True``, the shift is applied after the fiber
    is first centered about the midpoint of the fiber length.

    **Usage Context**:
    - For 1D fibers: Used in :meth:`Fiber.resample_potentials` for temporary testing of different alignments
    - For 3D fibers: Used in :func:`build_fiber_3d` for permanent fiber positioning

    :param fiber_length: The total length of the fiber (µm).
    :param delta_z: Internodal length of the fiber (µm).
    :param shift: Shift distance in microns (µm).
    :param shift_ratio: Shift as a ratio of ``delta_z``.
    :param center: If True, shift is applied after re-centering the
        fiber's length around the midpoint of the fiber path.
    :raises ValueError: If both ``shift`` and ``shift_ratio`` are provided.
    :return: The shifted start position of the fiber along its length.
    """
    if shift_ratio is not None and shift != 0:
        raise ValueError("Cannot specify both shift and shift_ratio")

    # Shift in microns
    shift_in_um = shift_ratio * delta_z if shift_ratio is not None else shift

    # Log informative message if modulo operation will change the shift
    if abs(shift_in_um) >= delta_z:
        logger.info(
            "Note: Requested shift of %.3f µm exceeds one internodal length "
            "(delta_z = %.3f µm). Using equivalent shift of %.3f µm instead.",
            shift_in_um,
            delta_z,
            shift_in_um % delta_z,
        )

    # Point to base shifting on
    point_to_shift = fiber_length / 2 if center else 0

    # Calculate final shift and run modulo operation
    final_shift = point_to_shift + shift_in_um
    return final_shift % delta_z


class Fiber:
    """Base class for model fibers.

    The :class:`Fiber` class provides functionality for constructing,
    configuring, and simulating fiber models. It encapsulates key
    methods for:

    - Generating sections specified by a fiber model subclass
    - Recording membrane voltage, current, and gating variables
    - Calculating extracellular potentials and single fiber action potentials
    - Measuring conduction velocity along the fiber
    - Handling 3D or 1D fiber geometry
    """

    def __init__(
        self: Fiber,
        fiber_model: FiberModel,
        diameter: float,
        temperature: float = 37,
        passive_end_nodes: int | bool = True,
        is_3d: bool = False,
    ) -> None:
        """Initialize the :class:`Fiber` class.

        :param fiber_model: The enumerator attribute (from :class:`FiberModel`) representing the type of fiber model.
        :param diameter: The diameter of the fiber (µm).
        :param temperature: The temperature at which the fiber will be simulated, in Celsius.
        :param passive_end_nodes: If ``True``, automatically assign passive properties to the end nodes.
            Can also be an integer specifying how many passive end nodes to include at each end.
        :param is_3d: If ``True``, fiber coordinates are treated as 3D.
            Usually set automatically by :func:`build_fiber_3d`.

        .. Intrinsic to the fiber model

        :ivar v_rest: The resting membrane potential of the fiber (in mV).
        :ivar gating_variables: A dictionary mapping gating variable
            names to their corresponding NEURON attribute names.
        :ivar myelinated: ``True`` if myelinated, ``False`` if unmyelinated.

        .. Intrinsic to the fiber

        :ivar diameter: The diameter of the fiber in micrometers.
        :ivar fiber_model: The :class:`FiberModel` attribute name.
        :ivar temperature: The temperature at which the fiber will be simulated [C].
        :ivar passive_end_nodes: The number of passive end nodes included at each end.
        :ivar delta_z: The center-to-center internodal length of the fiber.

        .. Intrinsic from build_fiber

        :ivar nodecount: The number of nodes in the fiber.
        :ivar length: The total length of the fiber in micrometers (end-to-end).
        :ivar sections: A list of NEURON :class:`Section <neuron:Section>` objects representing the fiber.
        :ivar nodes: A list of NEURON :class:`Section <neuron:Section>` objects representing
            the fiber nodes only (subset of ``Fiber.sections``).
        :ivar coordinates: A numpy array of 3D coordinates for
            the center of each section along the fiber.
        :ivar longitudinal_coordinates: A numpy array of 1D (arc-length) coordinates
            of the center of each section along the fiber.
        :ivar path: A :class:`nd_line` object representing the 3D path of the fiber.

        .. data recording

        :ivar apc: A list of NEURON :class:`APCount <neuron:APCount>` objects for
            detecting action potentials (from :meth:Fiber.apcounts()).
        :ivar vm: A list of NEURON :class:`Vector <neuron:Vector>` objects
            recording membrane voltage (from :meth:Fiber.record_vm()).
        :ivar im: A list of NEURON :class:`Vector <neuron:Vector>` objects
            recording membrane current (from :meth:Fiber.record_im()).
        :ivar vext: A list of NEURON :class:`Vector <neuron:Vector>` objects
            recording extracellular potential (from :meth:Fiber.record_vext()).
        :ivar time: A NEURON :class:`Vector <neuron:Vector>` recording the simulation time.
        :ivar gating: A dictionary mapping gating variable names to lists of
            recorded NEURON :class:`Vector <neuron:Vector>` objects
            (from :meth:Fiber.record_gating()).

        .. synapse

        :ivar nc: A NEURON :class:`NetCon <neuron:NetCon>` object for intrinsic activity.
        :ivar syn: A NEURON :class:`ExpSyn <neuron:ExpSyn>` object for intrinsic activity.
        :ivar stim: A NEURON :class:`NetStim <neuron:NetStim>` object for intrinsic activity.

        .. set by user

        :ivar potentials: A numpy array of extracellular potentials at each node along the fiber.
            For more info see the documentation on `extracellular potentials <extracellular potentials.md>` in PyFibers.
        """
        self.diameter = diameter
        self.fiber_model = fiber_model
        self.temperature = temperature
        self.passive_end_nodes = passive_end_nodes
        self.__is_3d = is_3d

        # recording
        self.gating: dict[str, h.Vector] = None
        self.gating_variables: dict[str, str] = {}
        self.vm: list = None
        self.apc: list = None
        self.im: list = None
        self.vext: list = None
        self.time: h.Vector = None

        # intrinsic activity
        self.nc: h.NetCon = None
        self.syn: h.ExpSyn = None
        self.stim: h.NetStim = None

        # fiber attributes
        self.myelinated: bool = None
        self.v_rest = None
        self.balanced = False
        self.nodecount: int = None
        self.delta_z: float = None
        self.sections: list = []
        self.nodes: list = []
        self.length: float = None
        self.coordinates: np.ndarray = np.array([])
        self.potentials: np.ndarray = np.array([])
        self.longitudinal_coordinates: np.ndarray = np.array([])
        self.path: nd_line = None

    # MAGIC METHODS #

    def __call__(self: Fiber, loc: float, target: str = 'nodes') -> h.Section:
        """Retrieve a node or section at a given normalized position along the fiber.

        Returns either the node or section nearest to ``loc * (len - 1)`` from the fiber.
        Note that the indexing is performed on the nodes or sections list.
        This means that the number represents the proportion along the
        list of nodes or sections, not necessarily along the physical fiber
        (though these are generally the same).

        :param loc: Location in the range [0, 1].
        :param target: Specifies whether to retrieve from ``'nodes'`` or ``'sections'``.
        :raises ValueError: If ``loc`` is not in [0, 1] or if ``target`` is not ``'nodes'`` or ``'sections'``.
        :return: The chosen node or section as a :class:`h.Section`.
        """
        if not (0 <= loc <= 1):
            raise ValueError("Location must be between 0 and 1")
        if target == 'sections':
            return self.sections[self.loc_index(loc, target=target)]
        if target != 'nodes':
            raise ValueError('target can either be "nodes" or "sections"')
        return self.nodes[self.loc_index(loc, target=target)]

    loc = __call__  # alias for backwards compatibility

    def __str__(self: Fiber) -> str:
        """Return a brief string representation of the fiber.

        :return: Description of the fiber model, diameter, length, node count, and section count.
        """
        return (
            f"{self.fiber_model.name} fiber of diameter {self.diameter} µm and length {self.length:.2f} µm "
            f"\n\tnode count: {len(self)}, section count: {len(self.sections)}. "
            f"\n\tFiber is {'' if self.is_3d() else 'not '}3d."
        )

    def __repr__(self: Fiber) -> str:
        """Return a detailed string representation of the fiber.

        :return: Detailed object information suitable for debugging.
        """
        return (
            f"{self.__class__.__name__}("
            f"fiber_model={self.fiber_model}, "
            f"diameter={self.diameter}, "
            f"temperature={self.temperature}, "
            f"passive_end_nodes={self.passive_end_nodes}, "
            f"is_3d={self.__is_3d}, "
            f"gating_variables={self.gating_variables}, "
            f"nc={self.nc}, "
            f"syn={self.syn}, "
            f"stim={self.stim}, "
            f"myelinated={self.myelinated}, "
            f"nodecount={self.nodecount}, "
            f"delta_z={self.delta_z}, "
            f"sections={self.sections}, "
            f"nodes={self.nodes}, "
            f"length={self.length}, "
            f"coordinates={self.coordinates.tolist()}, "
            f"potentials={self.potentials.tolist()}, "
            f"longitudinal_coordinates={self.longitudinal_coordinates.tolist()}"
            ")"
        )

    def __len__(self: Fiber, target: str = 'nodes') -> int:
        """Return the number of nodes or sections in the fiber.

        :param target: Can be either ``'nodes'`` or ``'sections'``.
        :return: The count of nodes or sections.
        :raises RuntimeError: If nodecount does not match the actual number of nodes.
        :raises ValueError: If target is not 'nodes' or 'sections'.
        """
        if self.nodecount != len(self.nodes):
            raise RuntimeError("Node count does not match number of nodes")
        if target == 'sections':
            return len(self.sections)
        if target != 'nodes':
            raise ValueError('target can either be "nodes" or "sections"')
        return len(self.nodes)

    def __getitem__(self: Fiber, item: int) -> h.Section:
        """Index into the fiber nodes by integer index.

        :param item: Zero-based index of the node to retrieve.
        :return: The node (:class:`h.Section`) at the specified index.
        """
        return self.nodes[item]

    def __iter__(self: Fiber) -> typing.Iterator[h.Section]:
        """Return an iterator over the fiber's nodes.

        :return: An iterator over the list of nodes.
        """
        return iter(self.nodes)

    def __contains__(self: Fiber, item: h.Section) -> bool:
        """Check if a given section is part of this fiber.

        :param item: A NEURON :class:`h.Section` to check for membership.
        :return: ``True`` if the section is in this fiber, else ``False``.
        """
        return item in self.sections

    # END MAGIC METHODS #

    def _connect_sections(self: Fiber) -> None:
        """Connect all fiber sections together."""
        for i in range(len(self.sections) - 1):
            self.sections[i + 1].connect(self.sections[i])

    def _calculate_coordinates(self: Fiber) -> None:
        """Generate the internal 1D coordinates for each section and compute fiber length.

        Uses the length (L) of each section to calculate cumulative
        longitudinal coordinates, then sets `Fiber.longitudinal_coordinates` and `Fiber.length`.

        :raises RuntimeError: If the computed center-to-center distance
            does not match the expected length (based on `Fiber.delta_z`).
        """
        start_coords = np.array([0] + [section.L for section in self.sections[:-1]])  # start of each section
        end_coords = np.array([section.L for section in self.sections])  # end of each section
        self.longitudinal_coordinates: np.ndarray = np.cumsum((start_coords + end_coords) / 2)  # type: ignore
        self.length = np.sum([section.L for section in self.sections])
        if not np.isclose(
            self.longitudinal_coordinates[-1] - self.longitudinal_coordinates[0],
            self.delta_z * (self.nodecount - 1),
        ):
            raise RuntimeError("Fiber length is not correct.")

    def loc_index(self: Fiber, loc: float, target: str = 'nodes') -> int:
        """Convert a normalized location [0, 1] into an integer index for nodes or sections.

        :param loc: Location in the fiber (from 0 to 1).
        :param target: Indicates whether to index into ``'nodes'`` or ``'sections'``.
        :raises ValueError: If ``loc`` is not in [0, 1] or if ``target`` is invalid.
        :return: The integer index corresponding to the node or section.
        """
        if not (0 <= loc <= 1):
            raise ValueError("Location must be between 0 and 1")
        if target == 'sections':
            return int(loc * (len(self.sections) - 1))
        if target != 'nodes':
            raise ValueError('target can either be "nodes" or "sections"')
        return int(loc * (len(self) - 1))

    def is_3d(self: Fiber) -> bool:
        """Check if the fiber is using 3D coordinates.

        :return: ``True`` if 3D, otherwise ``False``.
        """
        return self.__is_3d

    @property
    def last_shift_amount(self: Fiber) -> float:
        """Amount of coordinate shift from last resampling operation.

        :return: Shift amount in micrometers (µm). Positive values indicate
            shift in the positive coordinate direction.
        """
        return getattr(self, '_last_resample_shift', 0.0)

    @property
    def shifted_coordinates(self: Fiber) -> np.ndarray:
        """Coordinates after applying last resampling shift.

        These represent the effective position of the fiber sections
        after the last call to resample_potentials().

        :return: 1D array of shifted longitudinal coordinates in micrometers (µm).
        """
        return self.longitudinal_coordinates + self.last_shift_amount

    def _set_3d(self: Fiber) -> None:
        """Mark the fiber as 3D.

        Typically called internally by :func:`build_fiber_3d`.
        """
        self.__is_3d = True

    def resample_potentials(
        self: Fiber,
        potentials: np.ndarray,
        potential_coords: np.ndarray,
        center: bool = False,
        inplace: bool = False,
        shift: float = 0,
        shift_ratio: float = None,
    ) -> np.ndarray:
        """Use linear interpolation to resample external potentials onto the fiber's coordinate system (1D).

        This is used when extracellular potentials are calculated from an external source,
        such as a finite element model. The potentials provided by the user should be sampled at high
        resolution along the fiber's path and provided alongside the corresponding arc-length coordinates.

        **Shifting Behavior**: For 1D fibers, shifting is handled during potential resampling.
        This allows you to test various fiber positions relative to a given potential distribution without
        recreating the fiber. You can apply shifts using either:
        - ``shift``: a shift in microns, OR
        - ``shift_ratio``: a fraction of ``delta_z`` (the internodal length).

        If ``center=True``, both the input coordinates and the fiber's coordinates will be
        shifted such that their midpoints align. If a shift is also specified, the shift is applied after centering.

        **Note**: Unlike 3D fibers, 1D fiber shifting is temporary and only affects the potential resampling.
        The underlying fiber remains unchanged.

        :param potentials: 1D array of external potential values.
        :param potential_coords: 1D array of coordinates corresponding to ``potentials``.
        :param center: If ``True``, center the potentials around the midpoint of each domain.
            If a shift is also specified, the shift is applied after centering.
        :param inplace: If ``True``, update `Fiber.potentials` with the resampled values.
        :param shift: Shift distance in microns (µm).
        :param shift_ratio: Shift as a ratio of ``delta_z``.
        :return: Interpolated potential values aligned with `Fiber.longitudinal_coordinates`.
        :raises ValueError: If input array sizes or monotonicity checks fail, or if potential
            coordinates don't span the fiber coordinates.
        """
        potential_coords, potentials = np.array(potential_coords), np.array(potentials)

        if len(potential_coords.shape) != 1:
            raise ValueError("Potential coordinates must be a 1D array")
        if len(potentials.shape) != 1:
            raise ValueError("Potentials must be a 1D array")
        if len(potential_coords) != len(potentials):
            raise ValueError("Potentials and coordinates must be the same length")
        if len(potential_coords) < 2:
            raise ValueError("Must provide at least two points for resampling")
        if not np.all(np.diff(potential_coords) > 0):
            raise ValueError("Potential coordinates must be monotonically increasing")

        # Start with original coordinates
        target_coords = self.longitudinal_coordinates.copy()
        total_shift = 0.0

        # Apply centering first if requested
        if center:
            # Align center of fiber with center of potential coordinates
            fiber_center = (target_coords[0] + target_coords[-1]) / 2
            potential_center = (potential_coords[0] + potential_coords[-1]) / 2
            center_shift = potential_center - fiber_center
            target_coords = target_coords + center_shift
            total_shift += center_shift
        else:
            # When not centering, align potential coordinates to start at 0
            potential_coords = potential_coords - potential_coords[0]

        # Then apply shift if specified (note: center=False for _shift_fiber since we handled centering above)
        if shift != 0 or shift_ratio is not None:
            shifted_start = _shift_fiber(self.length, self.delta_z, shift, shift_ratio, center=False)
            target_coords = target_coords + shifted_start
            total_shift += shifted_start

        # Store the total shift amount
        self._last_resample_shift = total_shift

        newpotentials = np.interp(target_coords, potential_coords, potentials)

        # Check that potential coordinates span the target coordinates
        pot_min, pot_max = np.amin(potential_coords), np.amax(potential_coords)
        target_min, target_max = np.amin(target_coords), np.amax(target_coords)

        if not ((pot_max >= target_max) and (pot_min <= target_min)):
            raise ValueError(
                f"Potential coordinates must span the fiber coordinates. "
                f"Potential range: [{pot_min:.3f}, {pot_max:.3f}] µm, "
                f"Target range: [{target_min:.3f}, {target_max:.3f}] µm. "
                f"Missing coverage: {max(0, target_min - pot_min):.3f} µm at start, "
                f"{max(0, target_max - pot_max):.3f} µm at end. "
                f"Consider creating a shorter fiber to fit within the potential distribution."
            )

        if inplace:
            self.potentials = newpotentials
            if len(self.potentials) != len(self.longitudinal_coordinates):
                raise ValueError(
                    f"Potentials and coordinates must be the same length. "
                    f"Got {len(self.potentials)} potentials and {len(self.longitudinal_coordinates)} coordinates."
                )

        return newpotentials

    def apcounts(self: Fiber, thresh: float = -30) -> None:
        """Create NEURON :class:`APCount <neuron:APCount>` objects at each node to detect action potentials.

        :param thresh: Threshold voltage (mV) for an action potential.
        """
        self.apc = [h.APCount(node(0.5)) for node in self]
        for apc in self.apc:
            apc.thresh = thresh

    def record_values(
        self: Fiber,
        ref_attr: str,
        allsec: bool = False,
        indices: list[int] = None,
        allow_missing: bool = True,
        recording_dt: float = None,
        recording_tvec: h.Vector = None,
    ) -> list[h.Vector | None]:
        """Record NEURON variable references (e.g. membrane voltage) along the fiber.

        Note that ``recording_dt`` and ``recording_tvec`` are mutually exclusive.
        If both are None, the variable is recorded at every simulation timestep.
        For more info, see the NEURON docs:
        https://nrn.readthedocs.io/en/latest/python/programming/math/vector.html#Vector.record

        :param ref_attr: The NEURON attribute to record (e.g. ``'_ref_v'``).
        :param allsec: If ``True``, record from sections (including nodes). Otherwise, only record from nodes.
        :param indices: Specific indices to record from (if None, record from all).
        :param allow_missing: If ``True``, allows missing attributes without raising an error (returns None).
        :param recording_dt: The time step [ms] for recording the values (separate from simulation dt).
            Should be larger than the simulation dt.
        :param recording_tvec: A NEURON :class:`Vector <neuron:Vector>`
            of time points at which to record the values (ms).
            Note that the user MUST keep this Vector in memory for the duration of the simulation.
            This means you must assign it to a variable that is not overwritten or deleted.
            For example, to record at time points 0, 1, 2, and 3 ms:

            .. code-block:: python

                recording_tvec = h.Vector([0, 1, 2, 3])  # store times in a Vector
                fiber.record_values("_ref_v", recording_tvec=recording_tvec)  # pass Vector to record
                stimulation.find_threshold(fiber)  # run the simulation
                plt.plot(recording_tvec, fiber.vm[0])  # plot the recorded values

        :raises ValueError: If indices is an empty list.
        :return: A list of NEURON :class:`Vector <neuron:Vector>` objects or
            None (if allow_missing=True and the requested attribute is missing).
        """
        if recording_dt and recording_tvec:
            raise ValueError("Cannot specify both recording_dt and recording_tvec")

        def safe_record(section: h.Section, ref_attr: str) -> h.Vector | None:
            try:
                if recording_dt:
                    return h.Vector().record(getattr(section(0.5), ref_attr), recording_dt)
                if recording_tvec is not None:
                    return h.Vector().record(getattr(section(0.5), ref_attr), recording_tvec)
                return h.Vector().record(getattr(section(0.5), ref_attr))
            except AttributeError as e:
                if not allow_missing:
                    raise e
                return None

        if indices == []:
            raise ValueError(
                "Indices cannot be an empty list. If you want to record from all nodes, omit the indices argument."
            )
        if allsec:
            return [safe_record(self.sections[i], ref_attr) for i in indices or range(len(self.sections))]

        return [safe_record(self.nodes[i], ref_attr) for i in indices or range(len(self.nodes))]

    def record_vm(self: Fiber, **kwargs) -> list[h.Vector | None]:
        """Record membrane voltage (mV) along the fiber.

        :param kwargs: Additional arguments passed to :meth:`Fiber.record_values`.
        :return: A list of NEURON :class:`Vector <neuron:Vector>` objects recording membrane voltage.
        """
        self.vm = self.record_values('_ref_v', **kwargs)
        return self.vm

    def record_im(self: Fiber, **kwargs) -> list[h.Vector | None]:
        """Record membrane current (nA) along the fiber.

        :param kwargs: Additional arguments passed to :meth:`Fiber.record_values`.
        :return: A list of NEURON :class:`Vector <neuron:Vector>` objects recording membrane current.
        """
        self.im = self.record_values('_ref_i_membrane', **kwargs)
        return self.im

    def record_vext(self: Fiber) -> list[h.Vector]:
        """Record extracellular potential (mV) from each section along the fiber.

        :return: A list of NEURON :class:`Vector <neuron:Vector>` objects recording extracellular potential.
        """
        self.vext = [h.Vector().record(sec(0.5)._ref_vext[0]) for sec in self.sections]
        return self.vext

    def record_gating(self: Fiber, **kwargs) -> dict[str, list[h.Vector | None]]:
        """Record gating parameters (ion channel states) from axon nodes.

        The gating variables must be declared in ``Fiber.gating_variables`` within the fiber model class.

        :param kwargs: Additional arguments passed to :meth:`Fiber.record_values`.
        :return: A dictionary mapping gating variable names to
            lists of recorded NEURON :class:`Vector <neuron:Vector>` objects.
        :raises RuntimeError: If ``Fiber.gating_variables`` is empty.
        """
        if not self.gating_variables:
            raise RuntimeError("Gating variables not defined for this fiber type")

        self.gating = {}
        for name, var in self.gating_variables.items():
            self.gating[name] = self.record_values(f"_ref_{var}", **kwargs)
        return self.gating

    set_save_im = record_im  # alias for backwards compatibility
    set_save_vext = record_vext  # alias for backwards compatibility
    set_save_vm = record_vm  # alias for backwards compatibility
    set_save_gating = record_gating  # alias for backwards compatibility

    def point_source_potentials(
        self: Fiber,
        x: float,
        y: float,
        z: float,
        i0: float,
        sigma: float | tuple,
        inplace: bool = False,
    ) -> np.ndarray:
        """Compute extracellular potentials at the fiber's coordinates due to a point source stimulus.

        .. seealso::

            Documentation on `extracellular potentials <extracellular potentials.md>` in PyFibers.

        :param x: x-coordinate of the source in µm.
        :param y: y-coordinate of the source in µm.
        :param z: z-coordinate of the source in µm.
        :param i0: Magnitude of the point-source current (mA).
        :param sigma: Conductivity (S/m). A float for isotropic or a tuple (sigma_x, sigma_y, sigma_z) for anisotropic.
        :param inplace: If ``True``, update ``Fiber.potentials`` in-place.
        :return: Extracellular potentials at each fiber coordinate, in mV.
        """
        # Calculate distance between source and sections
        xs = x - self.coordinates[:, 0]
        ys = y - self.coordinates[:, 1]
        zs = z - self.coordinates[:, 2]
        # convert to meters
        xs *= 1e-6
        ys *= 1e-6
        zs *= 1e-6

        if isinstance(sigma, (float, int)):
            # Isotropic case
            potentials = i0 / (4 * np.pi * sigma * np.sqrt(xs**2 + ys**2 + zs**2))
        else:
            # Anisotropic case (approximate)
            sigma_x, sigma_y, sigma_z = sigma
            potentials = i0 / (
                4 * np.pi * np.sqrt(sigma_y * sigma_z * xs**2 + sigma_x * sigma_z * ys**2 + sigma_x * sigma_y * zs**2)
            )

        if inplace:
            self.potentials = potentials

        return potentials

    def measure_cv(self: Fiber, start: float = 0.25, end: float = 0.75, tolerance: float = 0.005) -> float:
        """Estimate conduction velocity (m/s) by measuring AP times at two points (start and end).

        This method calculates the conduction velocity by comparing the
        action potential times at two specified normalized locations (using NEURON indexing).
        It also checks for linear conduction between the two points, within a specified tolerance.

        :param start: Starting position for conduction velocity measurement (from 0 to 1).
        :param end: Ending position for conduction velocity measurement (from 0 to 1).
        :param tolerance: Tolerance (ms) for checking linearity of AP times.
        :raises ValueError: If conduction is not approximately linear between ``start`` and ``end``.
        :raises RuntimeError: If no APs are detected at one or both of the measurement nodes.
        :return: Conduction velocity in meters per second (m/s).
        """
        start_ind, end_ind = self.loc_index(start), self.loc_index(end)
        for ind in [start_ind, end_ind]:
            if self.apc[ind].n <= 0:
                raise RuntimeError(f"No detected APs at node {ind}.")

        # Check linearity
        aptimes = [self.apc[ind].time for ind in range(start_ind, end_ind + 1)]
        if not np.allclose(np.diff(aptimes), np.diff(aptimes)[0], atol=tolerance):
            raise ValueError("Conduction is not linear between the specified nodes.")

        # Calculate conduction velocity
        coords = [self.longitudinal_coordinates[i] for i, section in enumerate(self.sections) if section in self.nodes]
        distance = np.abs(coords[start_ind] - coords[end_ind])
        time = np.abs(aptimes[-1] - aptimes[0])
        distance *= 1e-6  # convert to meters
        time *= 1e-3  # convert to seconds

        return distance / time

    def add_intrinsic_activity(
        self: Fiber,
        loc: float = 0.1,
        loc_index: int = None,
        avg_interval: float = 1,
        num_stims: int = 1,
        start_time: float = 1,
        noise: float = 0,
        synapse_tau: float = 0.1,
        synapse_reversal_potential: float = 0,
        netcon_weight: float = 0.1,
    ) -> None:
        """Add a spike-generating synapse to the fiber for intrinsic activity.

        The synapse is generated via a :class:`NetStim <neuron:NetStim>` (spike generator), which
        is connected to an :class:`ExpSyn <neuron:ExpSyn>` (exponential synapse) on the chosen node.
        A :class:`NetCon <neuron:NetCon>` object links them together, injecting an exponentially decaying current
        upon each spike event.

        :param loc: Normalized location along the fiber where the synapse is placed ([0,1]).
        :param loc_index: Alternatively, specify an integer index of the node.
        :param avg_interval: Average interval between :class:`NetStim <neuron:NetStim>` spikes (ms).
        :param num_stims: Number of spikes to deliver.
        :param start_time: Time to start delivering spikes (ms).
        :param noise: Noise parameter for spike intervals (0 = regular, 1 = Poisson).
        :param synapse_tau: Time constant (ms) for synaptic current decay.
        :param synapse_reversal_potential: Reversal potential (mV) of the synapse.
        :param netcon_weight: Weight of the :class:`NetCon <neuron:NetCon>` between the spike generator and synapse.
        :raises ValueError: If neither ``loc`` nor ``loc_index`` is specified, or if both are specified.
        """
        if (loc is None) == (loc_index is None):
            raise ValueError("Must specify either loc or loc_index")
        node = self[loc_index] if loc_index is not None else self(loc)

        # Create spike generator
        self.stim = h.NetStim()
        self.stim.interval = avg_interval
        self.stim.number = num_stims
        self.stim.start = start_time
        self.stim.noise = noise

        # Create synapse
        self.syn = h.ExpSyn(node(0.5))
        self.syn.tau = synapse_tau
        self.syn.e = synapse_reversal_potential
        self.syn_current = h.Vector().record(self.syn._ref_i)

        # Connect generator to synapse
        self.nc = h.NetCon(self.stim, self.syn)
        self.nc.weight[0] = netcon_weight
        self.nc.delay = 0

    @staticmethod
    def calculate_periaxonal_current(from_sec: h.Section, to_sec: h.Section, vext_from: float, vext_to: float) -> float:
        """Compute the periaxonal current between two compartments for myelinated models.

        :param from_sec: The NEURON :class:`h.Section` from which current flows.
        :param to_sec: The NEURON :class:`h.Section` receiving current.
        :param vext_from: Extracellular (periaxonal) potential at ``from_sec`` (mV).
        :param vext_to: Extracellular (periaxonal) potential at ``to_sec`` (mV).
        :return: The periaxonal current in mA.
        """
        length_from = 1e-4 * from_sec.L  # [cm]
        xraxial_from = from_sec.xraxial[0]  # [megaOhm/cm]
        length_to = 1e-4 * to_sec.L  # [cm]
        xraxial_to = to_sec.xraxial[0]  # [megaOhm/cm]
        r_periaxonal = 1e6 * (xraxial_to * length_to / 2 + xraxial_from * length_from / 2)  # [Ohm]
        return (vext_from - vext_to) / r_periaxonal  # I = V/R [mA]

    def membrane_currents(self: Fiber, downsample: int = 1) -> np.ndarray:
        """Compute membrane currents at each section for each time point in the simulation.

        Uses the methods described in Pena et. al 2024: http://dx.doi.org/10.1371/journal.pcbi.1011833

        For myelinated fibers, this calculation includes periaxonal currents between adjacent
        sections (based on `h.Section.xraxial <neuron.Section.xraxial>`). The result is a matrix of shape:
        [``num_timepoints``, ``num_sections``].

        This method returns a tuple consisting of:
        1. A 2D array (time steps x number of sections) of membrane currents in mA.
        2. The array of time points corresponding to those currents (downsampled by the specified factor).

        :param downsample: Factor to reduce the temporal resolution (e.g., downsample=2 takes every 2nd time step).
        :return: A tuple (``i_membrane_matrix``, ``downsampled_time``). The matrix contains total currents (mA).
            The time array contains the corresponding simulation times (ms).
        :raises RuntimeError: If membrane currents or extracellular potentials were not recorded properly.
        """
        if self.im is None:
            raise RuntimeError("Membrane currents not saved. Call record_im(allsec=True) before running simulation.")
        if self.vext is None:
            raise RuntimeError("Extracellular potentials not saved. Call record_vext() before running simulation.")
        if len(self.im) != len(self.sections):
            raise RuntimeError(
                "Membrane currents not saved for all sections, call record_im(allsec=True) before running simulation."
            )
        if len(self.time.as_numpy()) == 0:
            raise RuntimeError("No record of simulation found. Run a simulation before calling this method.")

        # Extract the full time vector and compute downsample indices
        time_vector = np.array(self.time)
        downsampled_time = time_vector[::downsample]
        downsample_idx = np.arange(0, len(time_vector), downsample)

        # Precompute geometry factors for each section (convert from µm to cm)
        sections_length = np.array([1e-4 * sec.L for sec in self.sections])  # [cm]
        sections_diameter = np.array([1e-4 * sec.diam for sec in self.sections])  # [cm]

        # Initialize the output matrix for membrane currents
        i_membrane_matrix = np.zeros((len(downsampled_time), len(self.sections)))

        # Loop through each time index of interest (downsampled)
        for time_idx in downsample_idx:
            # Convert time index to downsampled row index
            downsampled_row = time_idx // downsample

            # specific_i_membrane is the membrane current density [mA/cm^2] for each section at this instant
            specific_i_membrane = np.array(
                [self.im[sec_idx][time_idx] for sec_idx in range(len(self.sections))]
            )  # shape: (num_sections,)

            # Multiply current density by surface area (π * diameter * length) to get total current in mA
            i_membrane = np.pi * sections_length * sections_diameter * specific_i_membrane  # [mA]

            # Extract the recorded extracellular potentials [mV] at this time point
            v_ext = np.array(
                [self.vext[sec_idx][time_idx] for sec_idx in range(len(self.sections))]
            )  # shape: (num_sections,)

            # If fiber is myelinated, include periaxonal currents between adjacent sections
            if self.myelinated:
                # Initialize left/right periaxonal currents for all sections
                peri_i_left = np.zeros(len(self.sections))
                peri_i_right = np.zeros(len(self.sections))

                # For each section (except the first), compute the current flowing from the previous section
                peri_i_left[1:] = np.array(
                    [
                        self.calculate_periaxonal_current(
                            self.sections[sec_idx - 1],
                            self.sections[sec_idx],
                            v_ext[sec_idx - 1],
                            v_ext[sec_idx],
                        )
                        for sec_idx in range(1, len(self.sections))
                    ]
                )

                # For each section (except the last), compute the current flowing from the next section
                peri_i_right[:-1] = np.array(
                    [
                        self.calculate_periaxonal_current(
                            self.sections[sec_idx + 1],
                            self.sections[sec_idx],
                            v_ext[sec_idx + 1],
                            v_ext[sec_idx],
                        )
                        for sec_idx in range(len(self.sections) - 1)
                    ]
                )

                # The net current at each section is the membrane current plus currents from both neighbors
                net_i_extra = i_membrane + peri_i_left + peri_i_right  # [mA]
            else:
                net_i_extra = i_membrane  # Unmyelinated: no periaxonal current

            # Store the results in the output matrix
            i_membrane_matrix[downsampled_row] = net_i_extra

        # Return the membrane current matrix along with the downsampled time vector
        return i_membrane_matrix, downsampled_time

    @staticmethod
    def sfap(current_matrix: np.ndarray, potentials: np.ndarray) -> np.ndarray:
        """Compute the Single-Fiber Action Potential (SFAP) by multiplying currents with recording potentials.

        This method uses the principle of reciprocity to calculate the SFAP.

        :param current_matrix: 2D array of shape [timepoints, sections], containing currents in mA.
        :param potentials: 1D array of potentials (mV) at each section, length = number of sections.
            These potentials should be for a 1 mA unit current source from the recording electrode.
        :raises ValueError: If the number of columns in the current matrix does not match the length of potentials.
        :return: The computed SFAP in microvolts (µV).
        """
        if len(potentials) != current_matrix.shape[1]:
            raise ValueError("Potentials and current matrix columns must have the same length")
        return 1e3 * np.dot(current_matrix, potentials)  # Convert to µV

    def record_sfap(self: Fiber, rec_potentials: list | np.ndarray, downsample: int = 1) -> np.ndarray:
        """Compute the SFAP time course at a given electrode location.

        :param rec_potentials: 1D array of precomputed potentials (mV)
            at each fiber section due to the electrode placement. These potentials should be calculated
            assuming a 1 mA unit current source from the recording electrode
            (e.g., using :meth:`Fiber.point_source_potentials` with ``i0=1``).
        :param downsample: Downsampling factor for the time vector (applies to the current matrix).
        :return: A tuple (``sfap_trace``, ``downsampled_time``) where ``sfap_trace`` is the computed single-fiber
            action potential in microvolts (µV) and ``downsampled_time`` is the corresponding time array (ms).
        """
        membrane_currents, downsampled_time = self.membrane_currents(downsample)
        return self.sfap(membrane_currents, rec_potentials), downsampled_time

    def set_xyz(self: Fiber, x: float = 0, y: float = 0, z: float = 0) -> None:
        """Assign new (x, y, z) shifts to a straight (1D) fiber.

        The fiber is assumed to be along the z-axis initially, with x=0 and y=0.
        This method sets the x and y coordinates to the specified values
        and shifts the z coordinate by the given amount.

        :param x: Shift in the x-direction (µm).
        :param y: Shift in the y-direction (µm).
        :param z: Shift in the z-direction (µm).
        :raises ValueError: If this fiber is a 3D fiber (since this method is for 1D only).
        """
        if self.__is_3d:
            raise ValueError("set_xyz() is not compatible with 3D fibers")
        if not np.allclose(self.coordinates[:, 0], self.coordinates[0, 0]) or not np.allclose(
            self.coordinates[:, 1], self.coordinates[0, 1]
        ):
            warnings.warn("X or Y coordinates vary, you may be operating on a 3D fiber path.", stacklevel=2)
        self.coordinates[:, 0] = x
        self.coordinates[:, 1] = y
        self.coordinates[:, 2] += z

    # 3D Fiber functionality #
    def resample_potentials_3d(
        self: Fiber,
        potentials: np.ndarray,
        potential_coords: np.ndarray,
        center: bool = False,
        inplace: bool = False,
    ) -> np.ndarray:
        """Interpolate external potentials onto a 3D fiber coordinate system.

        A wrapper around :meth:`Fiber.resample_potentials` that handles 3D coordinates by first computing
        the arc length of the provided coordinate array. As with the 1D version, this method
        is used to resample external potentials (e.g., from a finite element model) onto the fiber.

        **Note on shifting**: Unlike the 1D version, this method does not support shifting parameters
        (``shift``, ``shift_ratio``). For 3D fibers, shifting is handled during fiber creation via
        :func:`build_fiber_3d` parameters. If you need to test different fiber positions, create a new
        3D fiber with the desired shift parameters.

        At present, this does not check that the input coordinates lie exactly along the 3D fiber path.
        Therefore, it is recommended to use the same coordinates to construct the fiber as to use here.
        Alternatively, you can create a 1D fiber and calculate the coordinate arc lengths.
        For more information, see :doc:`/extracellular_potentials`.

        :param potentials: 1D array of external potential values.
        :param potential_coords: 2D array of shape (N, 3) representing the (x, y, z) coordinates
            where the potentials are measured or computed.
        :param center: If ``True``, center the potentials around the midpoint of each domain.
        :param inplace: If ``True``, update `Fiber.potentials` with the resampled values.
        :return: Interpolated potential values aligned with the fiber's 3D arc-length coordinates.
        :raises ValueError: If called on a non-3D fiber or if input coordinate shapes are invalid.
        """
        if not self.__is_3d:
            raise ValueError("resample_potentials_3d() is only compatible with 3D fibers")

        potential_coords, potentials = np.array(potential_coords), np.array(potentials)

        if len(potential_coords.shape) != 2:
            raise ValueError(
                "Potential coordinates must be a 2D array. " "If using arc lengths, use resample_potentials() instead."
            )
        if potential_coords.shape[1] != 3:
            raise ValueError("Must provide exactly 3 coordinates for x, y, z")

        # Convert (x, y, z) into cumulative arc length
        line = nd_line(potential_coords)
        arc_lengths = line.cumul

        return self.resample_potentials(
            potentials=potentials,
            potential_coords=arc_lengths,
            center=center,
            inplace=inplace,
        )

    # Fiber creation methods #

    def generate(
        self: Fiber,
        function_list: list[Callable],
        n_nodes: int = None,
        n_sections: int = None,
        length: float = None,
        enforce_odd_nodecount: bool = True,
    ) -> Fiber:
        """Build the fiber model sections in NEURON according to a specified generation strategy.

        This method either uses the desired number of nodes (``n_nodes``), the number of sections (``n_sections``),
        or determines the total number of sections from the fiber length and the spacing `Fiber.delta_z`.

        :param function_list: List of callable functions that each create a section.
            Typically structured as [node_func, myelin_func].
        :param n_nodes: Number of nodes of Ranvier.
        :param n_sections: Total number of sections in the fiber.
        :param length: Total length of the fiber (µm). Overrides n_sections if given.
        :param enforce_odd_nodecount: If ``True``, ensures that the fiber has an odd number of nodes.
        :return: The updated :class:`Fiber` instance after generation.
        :raises ValueError: If the computed number of sections does not align with the
            function_list-based pattern.
        """
        if n_nodes is not None:
            n_sections = (n_nodes - 1) * len(function_list) + 1
        elif length is not None:
            n_sections = math.floor(length / self.delta_z) * len(function_list) + 1
        else:
            if n_sections is None:
                raise ValueError("n_sections must be specified")

        if (n_sections - 1) % len(function_list) != 0:
            raise ValueError(f"n_sections must be 1 + {len(function_list)}*n, " "where n is (number_of_nodes - 1).")

        self.nodecount = int(1 + (n_sections - 1) / len(function_list))

        if self.nodecount % 2 == 0 and enforce_odd_nodecount:
            logger.info("Altering node count from %s to %s to enforce odd number.", self.nodecount, self.nodecount - 1)
            self.nodecount -= 1

        if self.nodecount < 3:
            warnings.warn("Fiber has fewer than 3 nodes. Consider increasing the fiber length.", stacklevel=2)

        self._create_sections(function_list)
        self._calculate_coordinates()

        return self

    def _create_sections(self: Fiber, function_list: list[Callable]) -> Fiber:
        """Create and connect NEURON sections for each node or internode in the fiber.

        The provided ``function_list`` starts with a function for a node, followed
        by each internodal section in order. Each node is optionally converted to a passive node
        if it is within the range of ``Fiber.passive_end_nodes``.

        :param function_list: A list of functions that each return a new NEURON :class:`h.Section`.
        :return: The updated :class:`Fiber` instance.

        :meta public:
        """
        nsec = (self.nodecount - 1) * len(function_list) + 1
        for ind in range(nsec):
            function = function_list[ind % len(function_list)]
            # If this is a node location (ind % len(function_list) == 0)
            if ind % len(function_list) == 0:
                # passive or active node
                node_type = 'active'
                # If within the range of passive end nodes
                if (
                    self.passive_end_nodes
                    and ind / len(function_list) < self.passive_end_nodes
                    or self.nodecount - 1 - ind / len(function_list) < self.passive_end_nodes
                ):
                    node_type = 'passive'
                section = function(ind, node_type)
                if node_type == 'passive':
                    section = self._make_passive(section)
                self.nodes.append(section)
            else:
                section = function(ind)
            self.sections.append(section)

        self._connect_sections()
        return self

    def _make_passive(self: Fiber, node: h.Section) -> h.Section:
        """Convert a node section to passive by removing all active mechanisms.

        For more info, see :doc:`/fiber_models`.

        :param node: The node :class:`h.Section` to be made passive.
        :return: The modified section with a passive mechanism inserted.
        :raises ValueError: If the node's name does not contain 'passive'.

        :meta public:
        """
        if 'passive' not in node.name():
            raise ValueError("Passive node name must contain 'passive'")
        mt = h.MechanismType(0)
        for mechanism in node.psection()['density_mechs']:
            if mechanism == 'extracellular':
                continue
            mt.select(mechanism)
            mt.remove(sec=node)
        node.insert('pas')
        node.g_pas = 0.0001  # [S/cm^2]
        node.e_pas = self.v_rest  # [mV]
        node.Ra = 1e10  # [Ohm*cm]
        node.cm = 1  # [uF/cm^2]
        return node

    def nodebuilder(self: Fiber, ind: int, node_type: str) -> h.Section:
        """Create a generic node of Ranvier.

        This method sets the node length, diameter, and inserts the extracellular mechanism,
        but does not add active ion channel mechanisms. Subclasses or external code should add
        the relevant channels to define the node's electrophysiological behavior.

        :param ind: Index of this node in the overall fiber construction.
        :param node_type: A string identifying the node as 'active' or 'passive'.
        :return: The newly created node as a NEURON :class:`h.Section`.
        """
        node = h.Section(name=f"{node_type} node {ind}")
        node.L = self.delta_z
        node.diam = self.diameter
        node.nseg = 1
        node.insert('extracellular')
        node.xc[0] = 0  # short circuit
        node.xg[0] = 1e10  # short circuit
        node.v = self.v_rest

        return node
