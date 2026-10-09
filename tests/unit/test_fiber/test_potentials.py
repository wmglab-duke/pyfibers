"""Regression tests for potential assignment and NumPy mutation semantics.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

import numpy as np
import pytest

from pyfibers.fiber import Fiber


@pytest.fixture
def fiber():
    # These property operations need only coordinates and stored potentials.
    fiber = Fiber.__new__(Fiber)
    fiber.coordinates = np.zeros((3, 3))
    fiber.potentials = [[1, 2, 3], [10, 20, 30]]
    return fiber


def test_assignment_normalizes_single_source(fiber):
    values = np.array([1, 2, 3])
    fiber.potentials = values
    values[0] = 99
    np.testing.assert_array_equal(fiber.potentials, [[1, 2, 3]])


@pytest.mark.parametrize('other', [[[1, 2, 3], [4, 5, 6]], [1, 2, 3]])
def test_inplace_addition_superposes_values(fiber, other):
    expected = fiber.potentials.copy() + other
    fiber.potentials += other
    np.testing.assert_array_equal(fiber.potentials, expected)


def test_inplace_row_addition_preserves_other_sources(fiber):
    fiber.potentials[0] += [1, 2, 3]
    np.testing.assert_array_equal(fiber.potentials, [[2, 4, 6], [10, 20, 30]])


@pytest.mark.parametrize('operation', [lambda a: a.copy(), lambda a: a * 2])
def test_derived_array_addition_does_not_modify_fiber(fiber, operation):
    original = fiber.potentials.copy()
    derived = operation(fiber.potentials)
    expected = derived.copy() + 1
    derived += 1
    np.testing.assert_array_equal(derived, expected)
    np.testing.assert_array_equal(fiber.potentials, original)


@pytest.mark.parametrize(
    ('values', 'message'),
    [(None, 'No fiber potentials'), ([1, 2], 'match the length'), (np.ones((1, 3, 2)), '1D or 2D')],
)
def test_invalid_assignment_preserves_potentials(fiber, values, message):
    original = fiber.potentials.copy()
    with pytest.raises(ValueError, match=message):
        fiber.potentials = values
    np.testing.assert_array_equal(fiber.potentials, original)
