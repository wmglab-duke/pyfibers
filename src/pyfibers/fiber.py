"""Defines the Fiber class and helper functions for building fiber models.

This module provides functionality for building and simulating
both 1D and 3D fiber models in the NEURON environment.
"""

from __future__ import annotations

import math
import typing
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from nd_line.nd_line import nd_line
from neuron import h
from numpy import ndarray

h.load_file('stdrun.hoc')

if TYPE_CHECKING:
    from .model_enum import FiberModel


def build_fiber(
    fiber_model: FiberModel,
    diameter: float,
    length: float = None,
    n_sections: int = None,
    n_nodes: int = None,
    **kwargs,
) -> Fiber:
    """Generate a 1D (straight) fiber model in NEURON.

    This function creates a model fiber using the specified ``fiber_model`` class,
    with user specified diameter and length
    (from one of: number of sections, number of nodes, or length in microns).
    Additional keyword arguments are forwarded to the fiber model class constructor.

    By default, the first section of the fiber is located at the origin (0, 0, 0),
    and the fiber extends along the z-axis in the positive direction. To change the fiber's location,
    the method ``set_xyz()`` can be used to translate the fiber along the x, y, or z axes. To create
    fibers along a custom path in 3D space, use :func:`build_fiber_3d` instead.

    :param fiber_model: FiberModel enumerator specifying the type of fiber to instantiate.
    :param diameter: The fiber diameter in micrometers (µm).
    :param length: The total length of the fiber in micrometers (µm), if defining by length.
    :param n_sections: The total number of sections for discretizing the fiber, if defining by sections.
    :param n_nodes: The total number of nodes along the fiber, if defining by nodes.
    :param kwargs: Additional arguments forwarded to the underlying fiber model class.
    :raises ValueError: If more than one among ``length``, ``n_sections``, or ``n_nodes`` is specified.
    :return: A Fiber class instance.

    **Example**:

    .. code-block:: python

        from PyFibers import build_fiber, FiberModel

        fiber = build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10, n_nodes=25)
    """
    # must provide one of length, n_sections, or n_nodes, and only one
    if sum(x is not None for x in [length, n_sections, n_nodes]) != 1:
        raise ValueError("Must provide exactly one of length, n_sections, or n_nodes")

    assert "is_3d" not in kwargs, "is_3d is set automatically, try using build_fiber_3d() instead"

    fiber_class = fiber_model.value

    fiber_instance = fiber_class(diameter=diameter, fiber_model=fiber_model, **kwargs)

    fiber_instance.generate(n_nodes=n_nodes, n_sections=n_sections, length=length)

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

    assert len(fiber_instance) == fiber_instance.nodecount, "Node count does not match number of nodes"

    if fiber_instance.diameter > 3 and not fiber_instance.myelinated:
        warnings.warn(
            "Unmyelinated fibers are typically <=3 µm in diameter. "
            f"Received D={fiber_instance.diameter:.2f} µm. Proceed with caution.",
            stacklevel=2,
        )

    return fiber_instance


def build_fiber_3d(
    fiber_model: FiberModel,
    diameter: float,
    path_coordinates: ndarray,
    shift: float = 0,
    shift_ratio: float = None,
    center_shift: bool = False,
    **kwargs,
) -> Fiber:
    """Generate a 3D fiber model in NEURON based on a specified path.

    This function calculates the fiber's length from the user-supplied ``path_coordinates``
    and uses it internally to instantiate a 3D fiber model. The coordinates are a 2D numpy array of shape
    (number_of_points, 3),where each row represents a point in 3D space (x, y, z).

    The fiber model will be created by
    repeating sections along the path until no more nodes can be added without exceeding the path length.
    By default, the center of the first section is placed at the origin (0, 0, 0), and the fiber extends
    along the path.

    :param fiber_model: FiberModel enumerator specifying the type of fiber to instantiate.
    :param diameter: The fiber diameter in micrometers (µm).
    :param path_coordinates: Numpy array of shape (N, 3) specifying the 3D coordinates (x, y, z) of the fiber path.
    :param shift: shift in microns to apply to the fiber coordinates
    :param shift_ratio: ratio of the internodal length to shift the fiber coordinates
    :param center_shift: if True, center the fiber before applying the shift
    :param kwargs: Additional arguments forwarded to the underlying fiber model class.
    :raises ValueError: If ``path_coordinates`` is not provided, or if ``n_sections``, ``n_nodes``, or ``length``
        is specified (these are invalid in 3D mode).
    :return: A fully instantiated 3D fiber model class instance.

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
    shifted_start = _shift_fiber_3d(
        fiber_path,
        build_fiber(  # call build fiber to get delta_z
            fiber_model=fiber_model,
            diameter=diameter,
            length=fiber_path.length,
            **kwargs,
        ).delta_z,
        shift,
        shift_ratio,
        center_shift,
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


def _shift_fiber_3d(
    fiber_path: nd_line,
    dz: float,
    shift: float | None = None,
    shift_ratio: float | None = None,
    center: bool = False,
) -> float:
    """Shift a 3D fiber's coordinates along its path by a specified amount.

    You can specify either:
    - shift: a shift in microns, OR
    - shift_ratio: a fraction of self.delta_z (the internodal length).

    If the shift or shift_ratio exceed the internodal length (delta_z),
    the extra lengths will be removed (only the modulus is applied).

    Finally, if center=True, the shift is applied after the fiber
    is first centered about the midpoint of the 3D path.

    :param fiber_path: the 3D path of the fiber (nd_line object)
    :param dz: internodal length of the fiber (um)
    :param shift: shift distance in microns (um)
    :param shift_ratio: shift as a ratio of self.delta_z
    :param center: if True, shift is applied after re-centering
                the fiber's arc length around the midpoint of the 3D path
    :raises ValueError: if both shift and shift_ratio are provided
    :return: the shifted start position of the fiber along the 3D path
    """
    if shift_ratio is not None and shift != 0:
        raise ValueError("Cannot specify both shift and shift_ratio")

    # Shift in microns
    shift_in_um = shift_ratio * dz if shift_ratio is not None else shift

    # Point to base shifting on
    point_to_shift = fiber_path.length / 2 if center else 0

    # Return final shifted start position
    return (point_to_shift + shift_in_um) % dz


class Fiber:
    """Base class for model fibers.

    The ``Fiber`` class provides functionality for constructing,
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
        """Initialize the Fiber class.

        :param fiber_model: The enumerator representing the type of fiber model.
        :param diameter: The diameter of the fiber (µm).
        :param temperature: The temperature at which the fiber will be simulated, in Celsius.
        :param passive_end_nodes: If ``True``, automatically assign passive properties to the end nodes.
            Can also be an integer specifying how many passive end nodes to include at each end.
        :param is_3d: If ``True``, fiber coordinates are treated as 3D.
            Usually set automatically by :func:`build_fiber_3d`.
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
        self.nodecount: int = None
        self.delta_z: float = None
        self.sections: list = []
        self.nodes: list = []
        self.length: float = None
        self.coordinates: ndarray = np.array([])
        self.potentials: ndarray = np.array([])
        self.longitudinal_coordinates: ndarray = np.array([])
        self.path: nd_line = None

    # MAGIC METHODS #

    def __call__(self: Fiber, loc: float, target: str = 'nodes') -> h.Section:
        """Retrieve a node or section at a given normalized position along the fiber.

        Returns either the node or section nearest to ``loc * (len - 1)`` from the fiber.
        Note that the indexing is performed on the nodes or sections list.
        This means that a number is the proportion along the list of nodes or sections, not along the fiber
        (Though these are generally the same).

        :param loc: Location in the range [0, 1].
        :param target: Specifies whether to retrieve from ``'nodes'`` or ``'sections'``.
        :raises AssertionError: If loc is not in [0, 1] or if target is not ``'nodes'`` or ``'sections'``.
        :return: The chosen node or section.
        """
        assert 0 <= loc <= 1, "Location must be between 0 and 1"
        if target == 'sections':
            return self.sections[self.loc_index(loc, target=target)]
        assert target == 'nodes', 'target can either be "nodes" or "sections"'
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
        :raises AssertionError: If nodecount does not match the actual number of nodes.
        """
        assert self.nodecount == len(self.nodes), "Node count does not match number of nodes"
        if target == 'sections':
            return len(self.sections)
        assert target == 'nodes', 'target can either be "nodes" or "sections"'
        return len(self.nodes)

    def __getitem__(self: Fiber, item: int) -> h.Section:
        """Index into the fiber nodes by integer index.

        :param item: Zero-based index of the node to retrieve.
        :return: The node (h.Section) at the specified index.
        """
        return self.nodes[item]

    def __iter__(self: Fiber) -> typing.Iterator[h.Section]:
        """Return an iterator over the fiber's nodes.

        :return: Iterator over the node list.
        """
        return iter(self.nodes)

    def __contains__(self: Fiber, item: h.Section) -> bool:
        """Check if a given section is part of this fiber.

        :param item: NEURON Section to check for membership.
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
        longitudinal coordinates, then sets ``self.longitudinal_coordinates`` and ``self.length``.

        :raises AssertionError: If the computed center-to-center distance
            does not match the expected length (based on self.delta_z).
        """
        start_coords = np.array([0] + [section.L for section in self.sections[:-1]])  # start of each section
        end_coords = np.array([section.L for section in self.sections])  # end of each section
        self.longitudinal_coordinates: np.ndarray = np.cumsum((start_coords + end_coords) / 2)  # type: ignore
        self.length = np.sum([section.L for section in self.sections])
        assert np.isclose(
            self.longitudinal_coordinates[-1] - self.longitudinal_coordinates[0],
            self.delta_z * (self.nodecount - 1),
        ), "Fiber length is not correct."

    def loc_index(self: Fiber, loc: float, target: str = 'nodes') -> int:
        """Convert a normalized location [0, 1] into an integer index for nodes or sections.

        :param loc: Location in the fiber (from 0 to 1).
        :param target: Indicates whether to index into ``'nodes'`` or ``'sections'``.
        :raises AssertionError: If loc is not in [0, 1] or if target is invalid.
        :return: The integer index of the node or section.
        """
        assert 0 <= loc <= 1, "Location must be between 0 and 1"
        if target == 'sections':
            return int(loc * (len(self.sections) - 1))
        assert target == 'nodes', 'target can either be "nodes" or "sections"'
        return int(loc * (len(self) - 1))

    def is_3d(self: Fiber) -> bool:
        """Check if the fiber is using 3D coordinates.

        :return: ``True`` if 3D, otherwise ``False``.
        """
        return self.__is_3d

    def _set_3d(self: Fiber) -> None:
        """Mark the fiber as 3D.

        Typically called internally by ``build_fiber_3d``.
        """
        self.__is_3d = True

    def resample_potentials(
        self: Fiber,
        potentials: np.ndarray,
        potential_coords: np.ndarray,
        center: bool = False,
        inplace: bool = False,
    ) -> np.ndarray:
        """Use linear interpolation to resample external potentials onto the fiber's coordinate system (1D).

        This is used when extracellular potentials are calculated from an external source,
        such as a finite element model. The potentials provided by the user should be sampled at high
        resolution along the fiber's path and provided alongside the corresonding arc-length coordinates.

        If ``center=True``, both the input coordinates and the fiber's coordinates will be
        shifted such that their midpoints align.

        :param potentials: 1D array of external potential values.
        :param potential_coords: 1D array of coordinates corresponding to ``potentials``.
        :param center: If ``True``, center the potentials around the midpoint of each domain.
        :param inplace: If ``True``, update ``self.potentials`` with the resampled values.
        :return: Interpolated potential values aligned with ``self.longitudinal_coordinates``.
        :raises AssertionError: If input array sizes or monotonicity checks fail.
        """
        potential_coords, potentials = np.array(potential_coords), np.array(potentials)

        assert len(potential_coords.shape) == 1, "Potential coordinates must be a 1D array"
        assert len(potentials.shape) == 1, "Potentials must be a 1D array"
        assert len(potential_coords) == len(potentials), "Potentials and coordinates must be the same length"
        assert len(potential_coords) >= 2, "Must provide at least two points for resampling"
        assert np.all(np.diff(potential_coords) > 0), "Potential coordinates must be monotonically increasing"

        if not center:
            potential_coords = potential_coords - potential_coords[0]
            target_coords = self.longitudinal_coordinates
        else:
            target_coords = (
                self.longitudinal_coordinates
                - (self.longitudinal_coordinates[0] + self.longitudinal_coordinates[-1]) / 2
            )
            potential_coords = potential_coords - (potential_coords[0] + potential_coords[-1]) / 2

        newpotentials = np.interp(target_coords, potential_coords, potentials)

        assert (np.amax(potential_coords) >= np.amax(target_coords)) and (
            np.amin(potential_coords) <= np.amin(target_coords)
        ), "Potential coordinates must span the fiber"

        if inplace:
            self.potentials = newpotentials
            assert len(self.potentials) == len(
                self.longitudinal_coordinates
            ), "Potentials and coordinates must be the same length"

        return newpotentials

    def apcounts(self: Fiber, thresh: float = -30) -> None:
        """Create NEURON APCount objects at each node to detect action potentials.

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
        """Record NEURON variable references (e.g., membrane voltage) along the fiber.

        Note that recording_dit and recording_tvec are mutually exclusive.
        If both are None, the variable is recorded at every simulation timestep.
        For more info, see the NEURON docs:
        https://nrn.readthedocs.io/en/latest/python/programming/math/vector.html#Vector.record

        :param ref_attr: The NEURON attribute to record (e.g. ``'_ref_v'``).
        :param allsec: If ``True``, record from sections (including nodes). Otherwise, only record from nodes.
        :param indices: Specific indices to record from (if None, record from all).
        :param allow_missing: If ``True``, allows missing attributes without raising an error (returns None).
        :param recording_dt: The time step [ms] for recording the values (separate from simulation dt).
            Should be larger than the simulation dt.
        :param recording_tvec: NEURON vector of time points at which to record the values. [ms]
            Note that the user MUST keep this Vector in memory for the duration of the simulation.
            This means you must assign t to a variable, and that variable must not be overwritted or deleted.
            For example, to record at time points 0, 1, 2, and 3 ms:
            .. code-block:: python

                recording_tvec = h.Vector([0, 1, 2, 3])  # store times in a Vector
                fiber.record_values("_ref_v", recording_tvec=recording_tvec)  # pass Vector to record
                stimulation.find_threshold(fiber)  # run the simulation
                plt.plot(recording_tvec, fiber.vm[0])  # plot the recorded values
        :raises ValueError: If indices is an empty list.
        :return: A list of NEURON Vectors or None (if allow_missing=True and the requested attribute is missing).
        """
        assert not (recording_dt and recording_tvec), "Cannot specify both recording_dt and recording_tvec"

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

        :param kwargs: Additional arguments passed to ``record_values``.
        :return: List of NEURON vectors recording membrane voltage.
        """
        self.vm = self.record_values('_ref_v', **kwargs)
        return self.vm

    def record_im(self: Fiber, **kwargs) -> list[h.Vector | None]:
        """Record membrane current (nA) along the fiber.

        :param kwargs: Additional arguments passed to ``record_values``.
        :return: List of NEURON vectors recording membrane current.
        """
        self.im = self.record_values('_ref_i_membrane', **kwargs)
        return self.im

    def record_vext(self: Fiber) -> list[h.Vector]:
        """Record extracellular potential (mV) from each section along the fiber.

        :return: List of NEURON vectors recording ``vext``.
        """
        self.vext = [h.Vector().record(sec(0.5)._ref_vext[0]) for sec in self.sections]
        return self.vext

    def record_gating(self: Fiber, **kwargs) -> dict[str, list[h.Vector | None]]:
        """Record gating parameters (ion channel states) from axon nodes.

        The gating variables must be declared in ``self.gating_variables`` within the fiber model class.

        :param kwargs: Additional arguments passed to ``record_values``.
        :return: Dictionary mapping gating variable names to lists of recorded NEURON Vectors.
        :raises AssertionError: If ``self.gating_variables`` is empty.
        """
        assert self.gating_variables, "Gating variables not defined for this fiber type"

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

        :param x: x-coordinate of the source in µm.
        :param y: y-coordinate of the source in µm.
        :param z: z-coordinate of the source in µm.
        :param i0: Magnitude of the point-source current (mA).
        :param sigma: Conductivity (S/m). Float for isotropic or (sigma_x, sigma_y, sigma_z) for anisotropic.
        :param inplace: If ``True``, update the fiber's ``potentials`` in-place. #TODO update all attrs to use xref
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
        action potential times at two specified normalized locations (NEURON indexing).
        It also checks for linear conduction between the two points, with a specified tolerance.

        :param start: Starting position for conduction velocity measurement (from 0 to 1).
        :param end: Ending position for conduction velocity measurement (from 0 to 1).
        :param tolerance: Tolerance (ms) for checking linearity of AP times.
        :raises ValueError: If conduction is not approximately linear between ``start`` and ``end``.
        :raises AssertionError: If no APs are detected at one or both of the measurement nodes.
        :return: Conduction velocity in meters per second (m/s).
        """
        start_ind, end_ind = self.loc_index(start), self.loc_index(end)
        for ind in [start_ind, end_ind]:
            assert self.apc[ind].n > 0, f"No detected APs at node {ind}."

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

        The synapse is generated via a NetStim (spike generator), which
        is connected to an ExpSyn (exponential synapse) on the chosen node.
        A NetCon object links them together, injecting an exponentially decaying current
        upon each spike event.
        See the NEURON documentation for more information on these objects:
        - `NetStim <https://nrn.readthedocs.io/en/latest/hoc/modelspec/programmatic/mechanisms/mech.html#NetStim>`_
        - `ExpSyn <https://nrn.readthedocs.io/en/latest/hoc/modelspec/programmatic/mechanisms/mech.html#ExpSyn>`_
        - `NetCon <https://nrn.readthedocs.io/en/latest/hoc/modelspec/programmatic/network/netcon.html#>`_

        :param loc: Normalized location along the fiber where the synapse is placed ([0,1]).
        :param loc_index: Alternatively, specify an integer index of the node.
        :param avg_interval: Average interval between NetStim spikes (ms).
        :param num_stims: Number of spikes to deliver.
        :param start_time: Time to start delivering spikes (ms).
        :param noise: Noise parameter for spike intervals (0 = regular, 1 = Poisson).
        :param synapse_tau: Time constant (ms) for synaptic current decay.
        :param synapse_reversal_potential: Reversal potential (mV) of the synapse.
        :param netcon_weight: Weight of the NetCon between the spike generator and synapse.
        :raises AssertionError: If neither loc nor loc_index is specified, or if both are specified.
        """
        assert (loc is None) != (loc_index is None), "Must specify either loc or loc_index"
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

        :param from_sec: The NEURON Section from which current flows.
        :param to_sec: The NEURON Section receiving current.
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
        sections (based on ``xraxial``). The result is a matrix of shape:
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

        :param current_matrix: 2D array of shape [timepoints, sections], containing currents in mA.
        :param potentials: 1D array of potentials (mV) at each section, length = number of sections.
        :raises AssertionError: If matrix columns do not match the length of potentials.
        :return: The computed SFAP in microvolts (µV).
        """
        assert (
            len(potentials) == current_matrix.shape[1]
        ), "Potentials and current matrix columns must have the same length"
        return 1e3 * np.dot(current_matrix, potentials)  # Convert to µV

    def record_sfap(self: Fiber, rec_potentials: list | ndarray, downsample: int = 1) -> np.ndarray:
        """Compute the SFAP time course at a given electrode location.

        :param rec_potentials: 1D array of precomputed potentials (mV) at each fiber section
            due to the electrode placement.
        :param downsample: Downsampling factor for the time vector (applies to current matrix).
        :return: A tuple (``sfap_trace``, ``downsampled_time``). ``sfap_trace`` is the computed single-fiber
            action potential in microvolts (µV) across the downsampled time.
        """
        membrane_currents, downsampled_time = self.membrane_currents(downsample)
        return self.sfap(membrane_currents, rec_potentials), downsampled_time

    def set_xyz(self: Fiber, x: float = 0, y: float = 0, z: float = 0) -> None:
        """Assign new (x, y, z) shift to a straight (1D) fiber.

        The fiber is assumed to be along the z-axis initially, with x=0 and y=0.
        This method sets x and y coordinates to the specified values and shifts z by the given amount.

        :param x: Shift in the x-direction (µm).
        :param y: Shift in the y-direction (µm).
        :param z: Shift in the z-direction (µm).
        :raises AssertionError: If this fiber is a 3D fiber (since this method is for 1D only).
        """
        assert not self.__is_3d, "set_xyz() is not compatible with 3D fibers"
        if not np.allclose(self.coordinates[:, 0], x) or not np.allclose(self.coordinates[:, 1], y):
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

        A wrapper around ``resample_potentials`` but handles 3D coordinates by first computing
        the arc length of the provided coordinate array. As with the 1D version, this method
        is used to resample external potentials (e.g., from a finite element model) onto the fiber.

        At present, this does not check that the input coordinates are along the 3D fiber path.
        Therefore, it is recommended you use the same to construct the fiber as you use here.
        Alternatively, you can create a 1D fiber and calculate the coordinate arc lengths.
        For more information, see :doc:`/extracellular_potentials`.

        :param potentials: 1D array of external potential values.
        :param potential_coords: 2D array of shape (N, 3) representing the (x, y, z) coordinates
            where ``potentials`` are measured or computed.
        :param center: If ``True``, center the potentials around the midpoint of each domain.
        :param inplace: If ``True``, update ``self.potentials`` with the resampled values.
        :return: Interpolated potential values aligned with the fiber's 3D arc-length coordinates.
        :raises AssertionError: If called on a non-3D fiber or if input coordinate shapes are invalid.
        """
        assert self.__is_3d, "resample_potentials_3d() is only compatible with 3D fibers"

        potential_coords, potentials = np.array(potential_coords), np.array(potentials)

        assert len(potential_coords.shape) == 2, (
            "Potential coordinates must be a 2D array. " "If using arc lengths, use resample_potentials() instead."
        )
        assert potential_coords.shape[1] == 3, "Must provide exactly 3 coordinates for x,y,z"

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
    ) -> Fiber:
        """Build the fiber model sections in NEURON according to a specified generation strategy.

        This method either uses the desired number of nodes (``n_nodes``), the number of sections (``n_sections``),
        or determines the total number of sections from the fiber length and the spacing ``delta_z``.

        :param function_list: List of callable functions that each create a section.
            Typically structured as [node_func, myelin_func].
        :param n_nodes: Number of nodes of Ranvier.
        :param n_sections: Total number of sections in the fiber.
        :param length: Total length of the fiber (µm). Overrides n_sections if given.
        :return: The updated fiber instance after generation.
        :raises AssertionError: If the computed number of sections does not align with the
            function_list-based pattern.
        """
        if n_nodes is not None:
            n_sections = (n_nodes - 1) * len(function_list) + 1
        elif length is not None:
            n_sections = math.floor(length / self.delta_z) * len(function_list) + 1
        else:
            assert n_sections is not None

        assert (n_sections - 1) % len(function_list) == 0, (
            f"n_sections must be 1 + {len(function_list)}*n, " "where n is (number_of_nodes - 1)."
        )

        self.nodecount = int(1 + (n_sections - 1) / len(function_list))

        self._create_sections(function_list)
        self._calculate_coordinates()

        return self

    def _create_sections(self: Fiber, function_list: list[Callable]) -> Fiber:
        """Create and connect NEURON sections for each node or internode in the fiber.

        The provided ``function_list`` starts with a function for a node, followed
        by each internodal section in order. Each node is optionally converted to a passive node
        if it is within the range of ``passive_end_nodes``.

        :param function_list: A list of functions that each return a new NEURON Section.
        :return: The updated fiber instance.
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

        :param node: The node section to be made passive.
        :return: The modified section with a passive mechanism.
        :raises AssertionError: If the node's name does not contain 'passive'.
        """
        assert 'passive' in node.name(), "Passive node name must contain 'passive'"
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

        This method sets the node length, diameter, and extracellular mechanism
        but does not insert ion channel mechanisms. Subclasses or external code
        should add the relevant channels to define the node's electrophysiological behavior.

        :param ind: Index of this node in the overall fiber construction.
        :param node_type: A string identifying the node as 'active' or 'passive'.
        :return: The newly created node section.
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
