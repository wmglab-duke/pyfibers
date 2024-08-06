from __future__ import annotations

from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest

from pyfibers import Fiber, Stimulation


@pytest.fixture
def fiber():
    default_times = np.array([10, 9, 8, 7, 6, 6, 7, 8, 9, 10]).astype(float)
    fiber = create_autospec(Fiber)
    fiber.apc = [MagicMock(time=default_times[i]) for i in range(10)]
    fiber.passive_end_nodes = False
    fiber.sections = [MagicMock(name=MagicMock(return_value='section')) for _ in range(10)]
    return fiber


def test_end_excitation_checker_no_excitation(fiber):
    result = Stimulation.end_excitation_checker(fiber, multi_site_check=False, fail_on_end_excitation=False)
    assert result == False  # noqa E712


def test_end_excitation_checker_end_excitation(fiber):
    fiber.apc[0].time = 1
    print([apc.time for apc in fiber.apc])
    with pytest.raises(RuntimeError, match=r'End excitation detected'):
        Stimulation.end_excitation_checker(fiber, multi_site_check=False, fail_on_end_excitation=True)


def test_end_excitation_checker_multiple_activation_sites(fiber):
    fiber.apc[0].time = 1
    fiber.apc[9].time = 1
    with pytest.warns(RuntimeWarning, match=r'Found multiple activation sites'):
        Stimulation.end_excitation_checker(fiber, multi_site_check=True, fail_on_end_excitation=False)


def test_end_excitation_checker_passive_nodes_excitation(fiber):
    fiber.passive_end_nodes = True
    fiber.apc[0].time = 1
    with pytest.warns(UserWarning, match=r'End excitation detected'):
        Stimulation.end_excitation_checker(fiber, multi_site_check=False, fail_on_end_excitation=False)


def test_end_excitation_checker_continue_on_end_excitation(fiber):
    fiber.apc[0].time = 1
    result = Stimulation.end_excitation_checker(fiber, multi_site_check=False, fail_on_end_excitation=False)
    assert result == True  # noqa E712


def test_no_check_if_passed_none(fiber):
    Stimulation.end_excitation_checker(fiber, fail_on_end_excitation=None)


def test_plateu_excitation(fiber):
    fiber.apc[0].time = fiber.apc[1].time = fiber.apc[2].time = 1
    with pytest.raises(RuntimeError, match=r'End excitation detected'):
        Stimulation.end_excitation_checker(fiber, multi_site_check=False, fail_on_end_excitation=True)


def test_passive_end_nodes(fiber):
    fiber.passive_end_nodes = True
    fiber.apc[2].time = 1  # should not raise an error
    Stimulation.end_excitation_checker(fiber, multi_site_check=False, fail_on_end_excitation=True)

    fiber.passive_end_nodes = 2  # now should raise an error
    with pytest.raises(RuntimeError, match=r'End excitation detected'):
        Stimulation.end_excitation_checker(fiber, multi_site_check=False, fail_on_end_excitation=True)


def test_right_side_excitation(fiber):
    fiber.apc[-1].time = 1
    with pytest.raises(RuntimeError, match=r'End excitation detected'):
        Stimulation.end_excitation_checker(fiber, multi_site_check=False, fail_on_end_excitation=True)


if __name__ == "__main__":
    pytest.main()
