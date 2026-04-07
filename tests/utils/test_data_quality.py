"""Tests for rheojax.utils.data_quality."""

import numpy as np
import pytest

from rheojax.utils.data_quality import (
    check_wide_frequency_range,
    detect_data_range_decades,
)


@pytest.mark.smoke
class TestDetectDataRangeDecades:
    """Tests for detect_data_range_decades."""

    def test_known_range(self):
        x = np.array([1e-4, 1e4])
        decades = detect_data_range_decades(x)
        assert abs(decades - 8.0) < 0.01

    def test_single_decade(self):
        x = np.array([1.0, 10.0])
        decades = detect_data_range_decades(x)
        assert abs(decades - 1.0) < 0.01

    def test_no_positive_values(self):
        x = np.array([-1.0, -2.0, 0.0])
        decades = detect_data_range_decades(x)
        assert decades == 0.0

    def test_logspace_input(self):
        x = np.logspace(-3, 3, 100)
        decades = detect_data_range_decades(x)
        assert abs(decades - 6.0) < 0.1


@pytest.mark.smoke
class TestCheckWideFrequencyRange:
    """Tests for check_wide_frequency_range."""

    def test_narrow_range_not_wide(self):
        x = np.logspace(-1, 2, 50)
        result = check_wide_frequency_range(x, warn=False)
        assert result["is_wide_range"] is False

    def test_wide_range_detected(self):
        x = np.logspace(-8, 4, 100)
        result = check_wide_frequency_range(x, warn=False)
        assert result["is_wide_range"] is True
        assert result["decades"] > 8.0

    def test_custom_threshold(self):
        x = np.logspace(-3, 3, 50)  # 6 decades
        result = check_wide_frequency_range(x, threshold_decades=5.0, warn=False)
        assert result["is_wide_range"] is True

    def test_recommendation_in_result(self):
        x = np.logspace(-8, 4, 100)
        result = check_wide_frequency_range(x, warn=False)
        assert len(result["recommendation"]) > 0
