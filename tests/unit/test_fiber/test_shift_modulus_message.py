"""Tests for shift modulus message functionality.

The copyrights of this software are owned by Duke University. Please
refer to the LICENSE and README.md files for licensing instructions. The
source code can be found on the following GitHub repository:
https://github.com/wmglab-duke/ascent
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from pyfibers import FiberModel, build_fiber


class TestShiftModulusMessage:
    """Test suite for modulus operation warning messages."""

    @pytest.fixture
    def test_fiber(self):
        """Create a test fiber with known delta_z."""
        return build_fiber(
            fiber_model=FiberModel.MRG_INTERPOLATION,
            diameter=10.0,
            n_nodes=11,  # Use node count to ensure reasonable fiber
        )

    @pytest.fixture
    def test_potentials(self, test_fiber):
        """Create test potential distribution that spans the fiber."""
        # Create potential range that can accommodate fiber + largest expected shifts
        # We need to cover fiber length + potential large shifts (up to 3x delta_z)
        max_range = test_fiber.length + 3 * test_fiber.delta_z
        coords = np.linspace(0, max_range, 1000)
        potentials = np.ones(1000)  # Constant potentials for simplicity
        return potentials, coords

    def test_no_message_for_small_shifts(self, test_fiber, test_potentials):
        """Test that no message is printed for shifts smaller than delta_z."""
        potentials, coords = test_potentials

        # Capture stdout
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            # Shift smaller than delta_z should not trigger message
            test_fiber.resample_potentials(potentials, coords, shift=test_fiber.delta_z * 0.5)

        output = captured_output.getvalue()
        assert "Note: Requested shift" not in output

    def test_message_for_large_shifts(self, test_fiber, test_potentials):
        """Test that message is printed for shifts larger than delta_z."""
        potentials, coords = test_potentials

        # Capture stdout
        captured_output = io.StringIO()

        large_shift = test_fiber.delta_z + 50  # Definitely larger than delta_z

        with redirect_stdout(captured_output):
            test_fiber.resample_potentials(potentials, coords, shift=large_shift)

        output = captured_output.getvalue()
        assert "Note: Requested shift" in output
        assert f"{large_shift:.3f} µm" in output
        assert f"delta_z = {test_fiber.delta_z:.3f} µm" in output
        assert "Using equivalent shift" in output

    def test_message_content_accuracy(self, test_fiber, test_potentials):
        """Test that the message content is accurate."""
        potentials, coords = test_potentials

        # Use a shift that's exactly 1.5 times delta_z
        original_shift = test_fiber.delta_z * 1.5
        expected_equivalent = test_fiber.delta_z * 0.5  # Should be reduced by modulus

        # Capture stdout
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            test_fiber.resample_potentials(potentials, coords, shift=original_shift)

        output = captured_output.getvalue()

        # Check that the message contains the correct values
        assert f"{original_shift:.3f} µm" in output
        assert f"{expected_equivalent:.3f} µm" in output
        assert f"delta_z = {test_fiber.delta_z:.3f} µm" in output

    def test_message_for_exact_delta_z(self, test_fiber, test_potentials):
        """Test behavior when shift equals exactly delta_z."""
        potentials, coords = test_potentials

        # Capture stdout
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            # Shift exactly equal to delta_z
            test_fiber.resample_potentials(potentials, coords, shift=test_fiber.delta_z)

        output = captured_output.getvalue()

        # When shift equals delta_z exactly, modulo reduces it to 0, so should print message
        assert "Note: Requested shift" in output
        assert "Using equivalent shift of 0.000 µm" in output

    def test_message_for_shift_ratio(self, test_fiber, test_potentials):
        """Test that message works correctly with shift_ratio parameter."""
        potentials, coords = test_potentials

        # Capture stdout
        captured_output = io.StringIO()

        # Use shift_ratio > 1.0 to trigger modulus
        large_shift_ratio = 1.25

        with redirect_stdout(captured_output):
            test_fiber.resample_potentials(potentials, coords, shift_ratio=large_shift_ratio)

        output = captured_output.getvalue()

        # Should print message since effective shift > delta_z
        expected_shift = large_shift_ratio * test_fiber.delta_z
        assert "Note: Requested shift" in output
        assert f"{expected_shift:.3f} µm" in output

    def test_floating_point_precision_handling(self, test_fiber, test_potentials):
        """Test that floating point precision is handled correctly."""
        potentials, coords = test_potentials

        # Use a shift that's very close to but not exactly delta_z
        # Need to use a difference smaller than 1e-10 (the tolerance)
        # But since delta_z is large (>1000), the actual behavior will depend on
        # the modulus operation. Let's test with a small shift instead
        small_shift = 1e-12  # Very small shift that should not trigger message

        # Capture stdout
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            test_fiber.resample_potentials(potentials, coords, shift=small_shift)

        output = captured_output.getvalue()

        # Should not print message due to small shift
        assert "Note: Requested shift" not in output

    def test_message_with_centering(self, test_fiber, test_potentials):
        """Test that modulus message works correctly with centering."""
        potentials, coords = test_potentials

        # Capture stdout
        captured_output = io.StringIO()

        large_shift = test_fiber.delta_z + 100

        with redirect_stdout(captured_output):
            test_fiber.resample_potentials(potentials, coords, center=True, shift=large_shift)

        output = captured_output.getvalue()

        # Should still print modulus message even with centering
        assert "Note: Requested shift" in output
        assert f"{large_shift:.3f} µm" in output

    def test_multiple_resampling_messages(self, test_fiber, test_potentials):
        """Test that messages are printed for each resampling operation."""
        potentials, coords = test_potentials

        # Capture stdout for multiple operations
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            # First operation with large shift
            test_fiber.resample_potentials(potentials, coords, shift=test_fiber.delta_z + 50)

            # Second operation with different large shift
            test_fiber.resample_potentials(potentials, coords, shift=test_fiber.delta_z + 75)

        output = captured_output.getvalue()

        # Should have two messages
        message_count = output.count("Note: Requested shift")
        assert message_count == 2

        # Should contain both shift values
        assert f"{test_fiber.delta_z + 50:.3f} µm" in output
        assert f"{test_fiber.delta_z + 75:.3f} µm" in output

    def test_message_format_readability(self, test_fiber, test_potentials):
        """Test that the message format is user-friendly and informative."""
        potentials, coords = test_potentials

        # Capture stdout
        captured_output = io.StringIO()

        large_shift = test_fiber.delta_z * 2.3  # 2.3 times delta_z

        with redirect_stdout(captured_output):
            test_fiber.resample_potentials(potentials, coords, shift=large_shift)

        output = captured_output.getvalue()

        # Check message format and content
        assert "Note:" in output  # Should start with "Note:"
        assert "Requested shift" in output
        assert "exceeds one internodal length" in output
        assert "Using equivalent shift" in output
        assert "instead." in output

        # Should use µm symbol
        assert "µm" in output

        # Should format numbers to 3 decimal places
        lines = output.strip().split('\n')
        message_line = [line for line in lines if "Note:" in line][0]

        # Should contain properly formatted numbers
        assert f"{large_shift:.3f}" in message_line
        assert f"{test_fiber.delta_z:.3f}" in message_line
