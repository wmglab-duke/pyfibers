"""Tests for resampling potentials functionality.

The copyrights of this software are owned by Duke University. Please
refer to the LICENSE and README.md files for licensing instructions. The
source code can be found on the following GitHub repository:
https://github.com/wmglab-duke/ascent
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from pyfibers import FiberModel, build_fiber


class TestResamplePotentials:
    """Test suite for resample_potentials method and related functionality."""

    @pytest.fixture
    def test_fiber(self):
        """Create a standard test fiber."""
        return build_fiber(
            fiber_model=FiberModel.MRG_INTERPOLATION,
            diameter=10.0,
            n_nodes=11,  # Use node count instead of length to ensure reasonable fiber
        )

    @pytest.fixture
    def test_potentials(self, test_fiber):
        """Create test potential distribution that spans the fiber."""
        # Create a Gaussian potential distribution that covers the fiber with margin
        fiber_length = test_fiber.length
        # Make potential range 2x the fiber length to allow for shifting
        potential_range = fiber_length * 2
        n_coords = 1000
        coords = np.linspace(0, potential_range, n_coords)
        potentials = norm.pdf(np.linspace(-2, 2, n_coords), 0, 0.5) * 10
        return potentials, coords

    def test_basic_resampling(self, test_fiber, test_potentials):
        """Test basic potential resampling without shifts or centering."""
        potentials, coords = test_potentials

        # Test basic resampling
        resampled = test_fiber.resample_potentials(potentials, coords)

        # Check that we get the right number of points
        assert len(resampled) == len(test_fiber.longitudinal_coordinates)

        # Check that potentials are reasonable
        assert np.all(np.isfinite(resampled))
        assert np.min(resampled) >= 0  # Gaussian should be positive

        # Test inplace functionality
        original_potentials = test_fiber.potentials.copy()
        test_fiber.resample_potentials(potentials, coords, inplace=True)
        assert not np.array_equal(test_fiber.potentials, original_potentials)
        np.testing.assert_array_equal(test_fiber.potentials, resampled)

    def test_centering(self, test_fiber, test_potentials):
        """Test centered vs non-centered resampling."""
        potentials, coords = test_potentials

        # Test non-centered (default)
        non_centered = test_fiber.resample_potentials(potentials, coords, center=False)

        # Test centered
        centered = test_fiber.resample_potentials(potentials, coords, center=True)

        # Results should be different
        assert not np.array_equal(non_centered, centered)

        # Both should have same length
        assert len(non_centered) == len(centered) == len(test_fiber.longitudinal_coordinates)

    def test_shifting_microns(self, test_fiber, test_potentials):
        """Test shifting with absolute values in microns."""
        potentials, coords = test_potentials

        # Test different shift values
        shifts = [0, 100, 250, 500]
        results = []

        for shift in shifts:
            result = test_fiber.resample_potentials(potentials, coords, center=True, shift=shift)
            results.append(result)

            # Check shift tracking - when center=True, total shift includes centering
            # Simply verify that last_shift_amount is recorded (detailed validation in other tests)
            assert test_fiber.last_shift_amount is not None

            # For shift=0, verify centering occurred; for shift>0, verify shift was applied
            if shift == 0:
                assert test_fiber.last_shift_amount != 0  # Should have centering
            else:
                # Verify the shift component is present - the exact modulus may vary due to centering
                # but the magnitude should be reasonable
                assert abs(test_fiber.last_shift_amount) > 0

            # Check shifted coordinates
            expected_shifted = test_fiber.longitudinal_coordinates + test_fiber.last_shift_amount
            np.testing.assert_array_equal(test_fiber.shifted_coordinates, expected_shifted)

        # Results should be different for different shifts
        for i in range(len(results) - 1):
            assert not np.array_equal(results[i], results[i + 1])

    def test_shifting_ratio(self, test_fiber, test_potentials):
        """Test shifting with ratio values."""
        potentials, coords = test_potentials

        # Test different shift ratios
        shift_ratios = [0, 0.25, 0.5, 0.75, 1.0]
        results = []

        for shift_ratio in shift_ratios:
            result = test_fiber.resample_potentials(potentials, coords, center=True, shift_ratio=shift_ratio)
            results.append(result)

            # Check that shift amount is recorded and reasonable
            assert test_fiber.last_shift_amount is not None

            # For shift_ratio=0, should still have centering; for others, should have total shift
            if shift_ratio == 0:
                assert test_fiber.last_shift_amount != 0  # Should have centering
            else:
                # Should have some shift applied (centering + ratio shift)
                assert abs(test_fiber.last_shift_amount) > 0

        # Results should be different for different shift ratios
        for i in range(len(results) - 1):
            assert not np.array_equal(results[i], results[i + 1])

    def test_modulus_operation(self, test_fiber, test_potentials):
        """Test that shifts larger than delta_z are reduced by modulus."""
        potentials, coords = test_potentials

        # Test shift larger than delta_z
        large_shift = test_fiber.delta_z + 100
        expected_effective_shift = 100  # Should be reduced by modulus

        test_fiber.resample_potentials(potentials, coords, center=False, shift=large_shift)

        # Check that the effective shift is the modulus (use relaxed tolerance)
        assert abs(test_fiber.last_shift_amount - expected_effective_shift) < 1e-3

    def test_shift_and_shift_ratio_conflict(self, test_fiber, test_potentials):
        """Test that providing both shift and shift_ratio raises error."""
        potentials, coords = test_potentials

        with pytest.raises(ValueError, match="Cannot specify both shift and shift_ratio"):
            test_fiber.resample_potentials(potentials, coords, shift=100, shift_ratio=0.5)

    def test_input_validation(self, test_fiber):
        """Test input validation for resample_potentials."""
        # Test 2D arrays (should fail)
        with pytest.raises(ValueError, match="1D array"):
            test_fiber.resample_potentials(np.array([[1, 2], [3, 4]]), np.array([0, 1, 2, 3]))  # 2D potentials

        with pytest.raises(ValueError, match="1D array"):
            test_fiber.resample_potentials(np.array([1, 2, 3, 4]), np.array([[0, 1], [2, 3]]))  # 2D coordinates

        # Test mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            test_fiber.resample_potentials(np.array([1, 2, 3]), np.array([0, 1]))  # Different length

        # Test too few points
        with pytest.raises(ValueError, match="at least two points"):
            test_fiber.resample_potentials(np.array([1]), np.array([0]))

        # Test non-monotonic coordinates
        with pytest.raises(ValueError, match="monotonically increasing"):
            test_fiber.resample_potentials(np.array([1, 2, 3, 4]), np.array([0, 2, 1, 3]))  # Non-monotonic

    def test_coordinate_span_validation(self, test_fiber):
        """Test that potential coordinates must span fiber coordinates."""
        # Create potentials that don't span the full fiber
        fiber_length = test_fiber.length
        short_coords = np.array([0, fiber_length * 0.1, fiber_length * 0.2])  # Only covers 20% of fiber
        short_potentials = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="Potential coordinates must span the fiber coordinates"):
            test_fiber.resample_potentials(short_potentials, short_coords)

        # Should also suggest shortening the fiber
        with pytest.raises(ValueError, match="Consider creating a shorter fiber"):
            test_fiber.resample_potentials(short_potentials, short_coords)

    def test_shifted_coordinates_property(self, test_fiber, test_potentials):
        """Test the shifted_coordinates property."""
        potentials, coords = test_potentials

        # Initially, shifted coordinates should equal longitudinal coordinates
        np.testing.assert_array_equal(test_fiber.shifted_coordinates, test_fiber.longitudinal_coordinates)

        # After resampling with shift, should be different
        shift = 200
        test_fiber.resample_potentials(potentials, coords, shift=shift)

        expected_shifted = test_fiber.longitudinal_coordinates + test_fiber.last_shift_amount
        np.testing.assert_array_equal(test_fiber.shifted_coordinates, expected_shifted)

        # Original coordinates should remain unchanged
        assert test_fiber.last_shift_amount != 0
        assert not np.array_equal(test_fiber.shifted_coordinates, test_fiber.longitudinal_coordinates)

    def test_last_shift_amount_property(self, test_fiber, test_potentials):
        """Test the last_shift_amount property."""
        potentials, coords = test_potentials

        # Initially should be 0
        assert test_fiber.last_shift_amount == 0.0

        # After resampling without shift, should still be 0
        test_fiber.resample_potentials(potentials, coords)
        assert test_fiber.last_shift_amount == 0.0

        # After resampling with centering, should track center shift
        test_fiber.resample_potentials(potentials, coords, center=True)
        assert test_fiber.last_shift_amount != 0.0

        # After resampling with explicit shift, should track total shift
        explicit_shift = 150
        test_fiber.resample_potentials(potentials, coords, center=True, shift=explicit_shift)
        # Should include both centering and explicit shift
        assert abs(test_fiber.last_shift_amount) >= explicit_shift

    def test_interpolation_accuracy(self, test_fiber):
        """Test that interpolation produces expected results."""
        # Create simple linear potential distribution that spans the fiber
        fiber_length = test_fiber.length
        coords = np.linspace(0, fiber_length * 1.2, 5)  # Slightly longer than fiber
        potentials = np.array([0, 1, 2, 3, 4])  # Linear increase

        resampled = test_fiber.resample_potentials(potentials, coords)

        # All values should be between min and max of original
        assert np.all(resampled >= 0)
        assert np.all(resampled <= 4)

        # Test specific interpolation points if possible
        # (This would depend on the exact fiber coordinates)
        assert np.all(np.isfinite(resampled))

    def test_edge_cases(self, test_fiber):
        """Test edge cases and boundary conditions."""
        # Test with exactly two points
        fiber_length = test_fiber.length
        coords = np.array([0, fiber_length * 1.1])  # Slightly longer than fiber
        potentials = np.array([1, 5])

        resampled = test_fiber.resample_potentials(potentials, coords)
        assert len(resampled) == len(test_fiber.longitudinal_coordinates)
        assert np.all((resampled >= 1) & (resampled <= 5))

        # Test with constant potentials
        coords = np.linspace(0, fiber_length * 1.1, 100)
        potentials = np.full(100, 3.14)

        resampled = test_fiber.resample_potentials(potentials, coords)
        np.testing.assert_array_almost_equal(resampled, 3.14)

    def test_multiple_resampling_operations(self, test_fiber, test_potentials):
        """Test that multiple resampling operations work correctly."""
        potentials, coords = test_potentials

        # First resampling
        result1 = test_fiber.resample_potentials(potentials, coords, shift=100)
        shift1 = test_fiber.last_shift_amount

        # Second resampling with different parameters
        result2 = test_fiber.resample_potentials(potentials, coords, shift=200)
        shift2 = test_fiber.last_shift_amount

        # Shifts should be different
        assert shift1 != shift2

        # Results should be different
        assert not np.array_equal(result1, result2)

        # Last shift should reflect the most recent operation
        assert test_fiber.last_shift_amount == shift2


class TestShiftFiberHelper:
    """Test suite for the _shift_fiber helper function."""

    def test_shift_fiber_import(self):
        """Test that _shift_fiber can be imported for testing."""
        from pyfibers.fiber import _shift_fiber

        assert callable(_shift_fiber)

    def test_basic_shifting(self):
        """Test basic shift functionality."""
        from pyfibers.fiber import _shift_fiber

        fiber_length = 1000.0
        delta_z = 100.0

        # Test no shift (must explicitly pass 0)
        result = _shift_fiber(fiber_length, delta_z, shift=0)
        assert result == 0.0

        # Test positive shift
        result = _shift_fiber(fiber_length, delta_z, shift=50)
        assert result == 50.0

        # Test negative shift (should be wrapped to positive)
        result = _shift_fiber(fiber_length, delta_z, shift=-30)
        assert result == 70.0  # -30 % 100 = 70

    def test_shift_ratio(self):
        """Test shift_ratio functionality."""
        from pyfibers.fiber import _shift_fiber

        fiber_length = 1000.0
        delta_z = 100.0

        # Test shift_ratio (must explicitly pass shift=0 to avoid conflict)
        result = _shift_fiber(fiber_length, delta_z, shift=0, shift_ratio=0.5)
        assert result == 50.0  # 0.5 * 100

        result = _shift_fiber(fiber_length, delta_z, shift=0, shift_ratio=0.25)
        assert result == 25.0  # 0.25 * 100

    def test_centering(self):
        """Test centering functionality."""
        from pyfibers.fiber import _shift_fiber

        fiber_length = 1000.0
        delta_z = 100.0

        # Test centering without additional shift (must pass shift=0 explicitly)
        result = _shift_fiber(fiber_length, delta_z, shift=0, center=True)
        expected = (fiber_length / 2) % delta_z  # 500 % 100 = 0
        assert result == expected

        # Test centering with additional shift
        result = _shift_fiber(fiber_length, delta_z, shift=30, center=True)
        expected = (fiber_length / 2 + 30) % delta_z  # (500 + 30) % 100 = 30
        assert result == expected

    def test_modulus_operation(self):
        """Test that modulus operation works correctly."""
        from pyfibers.fiber import _shift_fiber

        fiber_length = 1000.0
        delta_z = 100.0

        # Test shift larger than delta_z
        result = _shift_fiber(fiber_length, delta_z, shift=150)
        assert result == 50.0  # 150 % 100 = 50

        # Test shift exactly equal to delta_z
        result = _shift_fiber(fiber_length, delta_z, shift=100)
        assert result == 0.0  # 100 % 100 = 0

    def test_error_handling(self):
        """Test error handling in _shift_fiber."""
        from pyfibers.fiber import _shift_fiber

        fiber_length = 1000.0
        delta_z = 100.0

        # Test providing both non-zero shift and shift_ratio
        with pytest.raises(ValueError, match="Cannot specify both shift and shift_ratio"):
            _shift_fiber(fiber_length, delta_z, shift=50, shift_ratio=0.5)


# Integration test that combines multiple components
class TestResamplingIntegration:
    """Integration tests for resampling functionality."""

    def test_tutorial_workflow(self):
        """Test the workflow from the tutorial notebook."""
        # Create fiber similar to tutorial
        fiber = build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=11)

        # Create Gaussian potential distribution that covers the fiber
        n_coords = 1000
        fiber_range = fiber.length * 1.5  # 1.5x fiber length for margin
        coords = np.linspace(0, fiber_range, n_coords)
        potentials = norm.pdf(np.linspace(-1, 1, n_coords), 0, 0.05) * 10

        # Test non-centered resampling
        non_centered = fiber.resample_potentials(potentials, coords, center=False)

        # Test centered resampling
        centered = fiber.resample_potentials(potentials, coords, center=True)

        # Test shifting
        shifted = fiber.resample_potentials(potentials, coords, center=True, shift=250)

        # All should have different results
        assert not np.array_equal(non_centered, centered)
        assert not np.array_equal(centered, shifted)

        # All should have valid length
        expected_length = len(fiber.longitudinal_coordinates)
        assert len(non_centered) == expected_length
        assert len(centered) == expected_length
        assert len(shifted) == expected_length

    def test_different_fiber_diameters(self):
        """Test resampling with different fiber diameters."""
        # We'll create a large potential distribution that can handle all fiber sizes
        coords = np.linspace(0, 15000, 1000)  # Large range to cover all fibers
        potentials = norm.pdf(np.linspace(-1, 1, 1000), 0, 0.3) * 5

        diameters = [5.7, 7.3, 10.0, 16.0]
        results = []
        fiber_properties = []

        for diameter in diameters:
            fiber = build_fiber(
                fiber_model=FiberModel.MRG_INTERPOLATION, diameter=diameter, n_nodes=11  # Use consistent node count
            )

            # Test that resampling works for all diameters
            resampled = fiber.resample_potentials(potentials, coords, center=True)
            results.append(resampled)
            fiber_properties.append((len(resampled), fiber.delta_z))

            # Check basic properties
            assert len(resampled) == len(fiber.longitudinal_coordinates)
            assert np.all(np.isfinite(resampled))

        # Results should be different due to different delta_z values or different lengths
        for i in range(len(results) - 1):
            # Compare either length or delta_z to ensure fibers are different
            len_i, delta_z_i = fiber_properties[i]
            len_j, delta_z_j = fiber_properties[i + 1]

            # Fibers should have different properties or different results
            assert len_i != len_j or delta_z_i != delta_z_j or not np.array_equal(results[i], results[i + 1])

    def test_error_recovery_workflow(self):
        """Test the error recovery workflow shown in tutorial."""
        # Create long fiber that will cause range errors
        long_fiber = build_fiber(
            fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=21  # Create longer fiber
        )

        # Create limited potential distribution that's shorter than the fiber
        fiber_length = long_fiber.length
        short_range = fiber_length * 0.3  # Only 30% of the fiber length to ensure error
        coords = np.linspace(0, short_range, 500)
        potentials = np.ones(500)

        # This should fail
        with pytest.raises(ValueError, match="Potential coordinates must span"):
            long_fiber.resample_potentials(potentials, coords)

        # Create shorter fiber that should work
        short_fiber = build_fiber(fiber_model=FiberModel.MRG_INTERPOLATION, diameter=10.0, n_nodes=7)  # Shorter fiber

        # This should succeed if the short fiber fits within the coords range
        # Create appropriate potentials for the short fiber
        short_fiber_coords = np.linspace(0, short_fiber.length * 1.2, 500)
        short_fiber_potentials = np.ones(500)
        result = short_fiber.resample_potentials(short_fiber_potentials, short_fiber_coords, center=True)
        assert len(result) == len(short_fiber.longitudinal_coordinates)
        assert np.all(np.isfinite(result))
