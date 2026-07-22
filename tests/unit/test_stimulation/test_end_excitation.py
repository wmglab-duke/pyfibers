from __future__ import annotations

from unittest.mock import MagicMock, create_autospec

import pytest

from pyfibers import Fiber, Stimulation


def make_fiber(ap_times: list[float], passive_end_nodes: int | bool = False) -> Fiber:
    fiber = create_autospec(Fiber)
    fiber.apc = [MagicMock(time=float(t)) for t in ap_times]
    fiber.passive_end_nodes = passive_end_nodes
    fiber.sections = [MagicMock(name=MagicMock(return_value="section")) for _ in range(len(ap_times))]
    fiber.__len__.return_value = len(ap_times)
    fiber.initiation_nodes = lambda: Fiber.initiation_nodes(fiber)
    return fiber


@pytest.fixture
def fiber():
    return make_fiber([10, 9, 8, 7, 6, 6, 7, 8, 9, 10])


@pytest.mark.parametrize(
    ("ap_times", "passive_end_nodes", "expected"),
    [
        pytest.param([10, 9, 8, 7, 6, 6, 7, 8, 9, 10], False, False, id="central_initiation"),
        pytest.param([10, 9, 8, 7, 6, 6, 7, 8, 9, 10], 1, False, id="central_with_passive_ends"),
        pytest.param([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], False, True, id="left_end_pen0"),
        pytest.param([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], False, True, id="right_end_pen0"),
        pytest.param([1, 15, 14, 13, 12, 11, 10, 9, 8, 7], 1, True, id="left_and_right_artifacts_pen1"),
        pytest.param([10, 9, 8, 7, 6, 5, 4, 3, 2, 2], 1, True, id="right_plateau_pen1"),
        pytest.param([10, 9, 8, 7, 6, 5, 4, 3, 2, 10], 1, True, id="right_buffer_node_pen1"),
        pytest.param([0] * 10, False, False, id="no_aps"),
    ],
)
def test_end_excitation_checker_detected(ap_times, passive_end_nodes, expected):
    fiber = make_fiber(ap_times, passive_end_nodes=passive_end_nodes)
    result = Stimulation.end_excitation_checker(fiber, multi_site_check=False, fail_on_end_excitation=False)
    assert result is expected


def test_end_excitation_checker_no_excitation(fiber):
    result = Stimulation.end_excitation_checker(fiber, multi_site_check=False, fail_on_end_excitation=False)
    assert result is False


def test_end_excitation_checker_raises_on_left_end(fiber):
    fiber.apc[0].time = 1
    with pytest.raises(RuntimeError, match=r"End excitation detected on fiber.nodes\[0\]"):
        Stimulation.end_excitation_checker(fiber, multi_site_check=False, fail_on_end_excitation=True)


def test_end_excitation_checker_raises_on_right_end(fiber):
    fiber.apc[-1].time = 1
    with pytest.raises(RuntimeError, match=r"End excitation detected on fiber.nodes\[9\]"):
        Stimulation.end_excitation_checker(fiber, multi_site_check=False, fail_on_end_excitation=True)


def test_end_excitation_checker_warns_when_not_failing(fiber):
    fiber.apc[0].time = 1
    with pytest.warns(UserWarning, match=r"End excitation detected on fiber.nodes\[0\]"):
        result = Stimulation.end_excitation_checker(fiber, multi_site_check=False, fail_on_end_excitation=False)
    assert result is True


def test_end_excitation_checker_skips_raise_and_warn_when_none(fiber):
    fiber.apc[0].time = 1
    result = Stimulation.end_excitation_checker(fiber, fail_on_end_excitation=None)
    assert result is True


def test_end_excitation_checker_none_without_end_excitation(fiber):
    result = Stimulation.end_excitation_checker(fiber, fail_on_end_excitation=None)
    assert result is False


def test_end_excitation_checker_multiple_activation_sites_warns(fiber):
    fiber.apc[0].time = 1
    fiber.apc[9].time = 1
    with pytest.warns(RuntimeWarning, match=r"Multiple activation sites detected"):
        Stimulation.end_excitation_checker(fiber, multi_site_check=True, fail_on_end_excitation=False)


def test_end_excitation_checker_multiple_sites_silent_when_disabled(fiber):
    fiber.apc[0].time = 1
    fiber.apc[9].time = 1
    result = Stimulation.end_excitation_checker(fiber, multi_site_check=False, fail_on_end_excitation=False)
    assert result is True


def test_end_excitation_checker_left_end_plateau(fiber):
    fiber.passive_end_nodes = 2
    fiber.apc[0].time = fiber.apc[1].time = fiber.apc[2].time = 1
    with pytest.raises(RuntimeError, match=r"End excitation detected on fiber.nodes\[0 1 2\]"):
        Stimulation.end_excitation_checker(fiber, multi_site_check=False, fail_on_end_excitation=True)


@pytest.mark.parametrize("passive_end_nodes", [True, 1])
def test_passive_end_nodes_allows_first_active_node(passive_end_nodes):
    """Initiation one node past passive end should not count as end excitation."""
    times = [10, 9, 8, 7, 6, 6, 7, 8, 9, 10]
    fiber = make_fiber(times, passive_end_nodes=passive_end_nodes)
    fiber.apc[2].time = 1
    Stimulation.end_excitation_checker(fiber, multi_site_check=False, fail_on_end_excitation=True)


def test_passive_end_nodes_flags_initiation_inside_buffer():
    times = [10, 9, 8, 7, 6, 6, 7, 8, 9, 10]
    fiber = make_fiber(times, passive_end_nodes=2)
    fiber.apc[2].time = 1
    with pytest.raises(RuntimeError, match=r"End excitation detected on fiber.nodes\[2\]"):
        Stimulation.end_excitation_checker(fiber, multi_site_check=False, fail_on_end_excitation=True)


def test_passive_end_nodes_two_includes_left_plateau():
    fiber = make_fiber([1, 1, 15, 14, 13, 12, 11, 10, 9, 8], passive_end_nodes=2)
    with pytest.raises(RuntimeError, match=r"End excitation detected on fiber.nodes\[0 1 9\]"):
        Stimulation.end_excitation_checker(fiber, multi_site_check=False, fail_on_end_excitation=True)


def test_end_excitation_uses_initiation_nodes_not_raw_ap_times():
    """End nodes with APs that are not local time minima should not count as initiation."""
    fiber = make_fiber([5, 3, 8, 7, 1, 1, 7, 8, 9, 10])
    result = Stimulation.end_excitation_checker(fiber, multi_site_check=False, fail_on_end_excitation=False)
    assert result is False


def test_end_excitation_checker_return_type(fiber):
    result = Stimulation.end_excitation_checker(fiber, fail_on_end_excitation=None)
    assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main()
