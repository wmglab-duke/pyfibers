from __future__ import annotations

from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest

from pyfibers.fiber import Fiber


def make_fiber(ap_times: list[float]) -> Fiber:
    fiber = create_autospec(Fiber)
    fiber.apc = [MagicMock(time=float(t)) for t in ap_times]
    return fiber


@pytest.fixture
def central_plateau_fiber():
    times = [10, 9, 8, 7, 6, 6, 7, 8, 9, 10]
    return make_fiber(times)


def test_initiation_nodes_requires_apc():
    fiber = create_autospec(Fiber)
    fiber.apc = None
    with pytest.raises(RuntimeError, match="AP counters not set up"):
        Fiber.initiation_nodes(fiber)


def test_initiation_nodes_return_types(central_plateau_fiber):
    init_nodes, n_sites, times = Fiber.initiation_nodes(central_plateau_fiber)
    assert isinstance(init_nodes, np.ndarray)
    assert isinstance(n_sites, int)
    assert isinstance(times, np.ndarray)
    assert times.shape == (len(central_plateau_fiber.apc),)


@pytest.mark.parametrize(
    ("ap_times", "expected_nodes", "expected_sites"),
    [
        pytest.param([10, 9, 8, 7, 6, 6, 7, 8, 9, 10], [4, 5], 1, id="central_plateau"),
        pytest.param([10, 9, 8, 7, 6, 7, 8, 9, 10], [4], 1, id="sharp_central_minimum"),
        pytest.param([10, 9, 8, 7, 6, 6, 6, 6, 9, 10], [4, 5, 6, 7], 1, id="wide_central_plateau"),
        pytest.param([10, 9, 4, 5, 6, 7, 8, 9, 10, 11], [2], 1, id="off_center"),
        pytest.param([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [9], 1, id="right_end"),
        pytest.param([1, 10, 9, 8, 7, 6, 5, 4, 3, 1], [0, 9], 2, id="dual_sites"),
        pytest.param([0] * 10, [], 0, id="no_aps"),
        pytest.param([5] * 10, list(range(10)), 1, id="all_nodes_same_time"),
        pytest.param([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0], 1, id="monotonic_from_left"),
        pytest.param([10, 9, 8, 3, 3, 8, 9, 10], [3, 4], 1, id="short_fiber_plateau"),
    ],
)
def test_initiation_nodes_scenarios(ap_times, expected_nodes, expected_sites):
    fiber = make_fiber(ap_times)
    init_nodes, n_sites, times = Fiber.initiation_nodes(fiber)
    assert list(init_nodes) == expected_nodes
    assert n_sites == expected_sites
    expected_times = np.array(ap_times, dtype=float)
    expected_times[expected_times == 0] = np.nan
    np.testing.assert_array_equal(times, expected_times)


def test_initiation_nodes_left_end_also_finds_right_local_minimum():
    """Decreasing times toward the far end can produce a second local minimum."""
    fiber = make_fiber([1, 15, 14, 13, 12, 11, 10, 9, 8, 7])
    init_nodes, n_sites, _ = Fiber.initiation_nodes(fiber)
    assert list(init_nodes) == [0, 9]
    assert n_sites == 2


def test_initiation_nodes_left_end_plateau():
    fiber = make_fiber([1, 1, 15, 14, 13, 12, 11, 10, 9, 8])
    init_nodes, n_sites, _ = Fiber.initiation_nodes(fiber)
    assert list(init_nodes) == [0, 1, 9]
    assert n_sites == 2


def test_initiation_nodes_single_interior_node():
    inf = float("inf")
    fiber = make_fiber([inf, inf, inf, inf, inf, 3, inf, inf, inf, inf])
    init_nodes, n_sites, _ = Fiber.initiation_nodes(fiber)
    assert list(init_nodes) == [5]
    assert n_sites == 1


def test_initiation_nodes_padding_maps_node_indices(central_plateau_fiber):
    """Plateau minimum at times[5:7] should map to nodes 4 and 5, not padded indices."""
    init_nodes, _, _ = Fiber.initiation_nodes(central_plateau_fiber)
    assert 4 in init_nodes
    assert 5 in init_nodes
    assert all(node < len(central_plateau_fiber.apc) for node in init_nodes)


def test_initiation_nodes_two_node_fiber():
    fiber = make_fiber([5, 10])
    init_nodes, n_sites, _ = Fiber.initiation_nodes(fiber)
    assert list(init_nodes) == [0]
    assert n_sites == 1


def test_initiation_nodes_single_node_fiber():
    fiber = make_fiber([3])
    init_nodes, n_sites, _ = Fiber.initiation_nodes(fiber)
    assert list(init_nodes) == [0]
    assert n_sites == 1


def test_initiation_nodes_rejects_padding_indices(monkeypatch):
    def fake_find_peaks(*args, **kwargs):
        return np.array([0]), {"left_edges": np.array([0]), "right_edges": np.array([1])}

    monkeypatch.setattr("pyfibers.fiber.find_peaks", fake_find_peaks)
    fiber = make_fiber([5, 10])
    with pytest.raises(RuntimeError, match="padding artifacts"):
        Fiber.initiation_nodes(fiber)


if __name__ == "__main__":
    pytest.main()
