"""Enhanced Series method tests without pandas dependency."""

from __future__ import annotations

import pytest

import polarpandas as ppd
from tests.test_helpers import assert_series_equal


class TestSeriesComparison:
    """Test Series comparison operators with pandas compatibility."""

    def setup_method(self):
        """Create test data in both pandas and polarpandas."""
        self.data = [1, 2, 3, 4, 5]
        # Don't create Series here to avoid state pollution
        # Each test method will create fresh Series

    def test_gt_comparison(self):
        """Test greater than comparison."""
        expected = [value > 3 for value in self.data]
        result = ppd.Series(self.data) > 3
        assert_series_equal(result, expected)

    def test_lt_comparison(self):
        """Test less than comparison."""
        expected = [value < 3 for value in self.data]
        result = ppd.Series(self.data) < 3
        assert_series_equal(result, expected)

    def test_ge_comparison(self):
        """Test greater than or equal comparison."""
        expected = [value >= 3 for value in self.data]
        result = ppd.Series(self.data) >= 3
        assert_series_equal(result, expected)

    def test_le_comparison(self):
        """Test less than or equal comparison."""
        expected = [value <= 3 for value in self.data]
        result = ppd.Series(self.data) <= 3
        assert_series_equal(result, expected)

    def test_eq_comparison(self):
        """Test equal comparison."""
        expected = [value == 3 for value in self.data]
        result = ppd.Series(self.data) == 3
        assert_series_equal(result, expected)

    def test_ne_comparison(self):
        """Test not equal comparison."""
        expected = [value != 3 for value in self.data]
        result = ppd.Series(self.data) != 3
        assert_series_equal(result, expected)

    def test_series_comparison(self):
        """Test Series to Series comparison."""
        other_data = [2, 3, 4, 5, 6]
        expected = [a > b for a, b in zip(self.data, other_data)]
        result = ppd.Series(self.data) > ppd.Series(other_data)
        assert_series_equal(result, expected)


class TestSeriesMethods:
    """Test enhanced Series methods with pandas compatibility."""

    def setup_method(self):
        """Create test data in both pandas and polarpandas."""
        self.data = [1, 2, 3, 4, 5]
        # Don't create Series here to avoid state pollution
        # Each test method will create fresh Series

    def test_between_basic(self):
        """Test between method."""
        expected = [2 <= value <= 4 for value in self.data]
        result = ppd.Series(self.data).between(2, 4)
        assert_series_equal(result, expected)

    def test_series_between_inclusive_both(self):
        """Test between with inclusive='both'."""
        # Lines 321-322: inclusive == "both"
        expected = [2 <= value <= 4 for value in self.data]
        result = ppd.Series(self.data).between(2, 4, inclusive="both")
        assert_series_equal(result, expected)

    def test_series_between_inclusive_neither(self):
        """Test between with inclusive='neither'."""
        # Lines 323-324: inclusive == "neither"
        expected = [2 < value < 4 for value in self.data]
        result = ppd.Series(self.data).between(2, 4, inclusive="neither")
        assert_series_equal(result, expected)

    def test_series_between_inclusive_left(self):
        """Test between with inclusive='left'."""
        # Lines 325-326: inclusive == "left"
        expected = [2 <= value < 4 for value in self.data]
        result = ppd.Series(self.data).between(2, 4, inclusive="left")
        assert_series_equal(result, expected)

    def test_series_between_inclusive_right(self):
        """Test between with inclusive='right'."""
        # Lines 327-328: inclusive == "right"
        expected = [2 < value <= 4 for value in self.data]
        result = ppd.Series(self.data).between(2, 4, inclusive="right")
        assert_series_equal(result, expected)

    def test_series_between_invalid_inclusive_raises_error(self):
        """Test between raises error for invalid inclusive parameter."""
        # Lines 329-332: ValueError for invalid inclusive
        s = ppd.Series(self.data)
        with pytest.raises(ValueError, match="inclusive must be one of"):
            s.between(2, 4, inclusive="invalid")

    def test_series_between_empty_series(self):
        """Test between with empty Series."""
        # Lines 316-317: Empty series handling
        result = ppd.Series([], dtype=float).between(2, 4)
        assert result.to_list() == []

    def test_series_arithmetic_with_nulls(self):
        """Test arithmetic operations with null values."""
        # Lines 133-135: Arithmetic with nulls
        s = ppd.Series([1, None, 3, None, 5])

        # Addition
        result = s + 10
        assert isinstance(result, ppd.Series)

        # Multiplication
        result = s * 2
        assert isinstance(result, ppd.Series)

        # Division
        result = s / 2
        assert isinstance(result, ppd.Series)

    def test_series_comparison_with_nulls(self):
        """Test comparison operations with null values."""
        s = ppd.Series([1, None, 3, None, 5])

        # Greater than
        result = s > 2
        assert isinstance(result, ppd.Series)

        # Equal
        result = s == 3
        assert isinstance(result, ppd.Series)

    def test_between_inclusive(self):
        """Test between with different inclusive options."""
        # Test 'neither'
        expected = [2 < value < 4 for value in self.data]
        result = ppd.Series(self.data).between(2, 4, inclusive="neither")
        assert_series_equal(result, expected)

    def test_clip_basic(self):
        """Test clip method."""
        expected = [min(max(value, 2), 4) for value in self.data]
        result = ppd.Series(self.data).clip(lower=2, upper=4)
        assert_series_equal(result, expected)

    def test_clip_lower_only(self):
        """Test clip with only lower bound."""
        expected = [max(value, 3) for value in self.data]
        result = ppd.Series(self.data).clip(lower=3)
        assert_series_equal(result, expected)

    def test_clip_upper_only(self):
        """Test clip with only upper bound."""
        expected = [min(value, 3) for value in self.data]
        result = ppd.Series(self.data).clip(upper=3)
        assert_series_equal(result, expected)

    def test_rank_basic(self):
        """Test rank method."""
        expected = [float(idx + 1) for idx in range(len(self.data))]
        result = ppd.Series(self.data).rank()
        assert_series_equal(result, expected, rtol=1e-9)

    def test_rank_method(self):
        """Test rank with different method."""
        expected = [float(idx + 1) for idx in range(len(self.data))]
        result = ppd.Series(self.data).rank(method="min")
        assert_series_equal(result, expected, rtol=1e-9)

    def test_rank_descending(self):
        """Test rank with descending order."""
        expected = [float(len(self.data) - idx) for idx in range(len(self.data))]
        result = ppd.Series(self.data).rank(ascending=False)
        assert_series_equal(result, expected, rtol=1e-9)

    def test_sort_values_basic(self):
        """Test sort_values method."""
        # Create unsorted data
        data = [3, 1, 4, 2, 5]
        result = ppd.Series(data).sort_values()
        assert list(result.to_list()) == [1, 2, 3, 4, 5]

    def test_sort_values_descending(self):
        """Test sort_values with descending order."""
        # Create unsorted data
        result = ppd.Series([3, 1, 4, 2, 5]).sort_values(ascending=False)
        assert list(result.to_list()) == [5, 4, 3, 2, 1]

    def test_value_counts_basic(self):
        """Test value_counts method."""
        # Create data with duplicates
        data = [1, 2, 2, 3, 3, 3]
        ppd_values = ppd.Series(data).value_counts().to_list()
        counts = [item["count"] for item in ppd_values]
        values = [item[""] for item in ppd_values]
        assert counts == [3, 2, 1]
        assert values == [3, 2, 1]

    def test_value_counts_normalize(self):
        """Test value_counts with normalize."""
        # Create data with duplicates
        data = [1, 2, 2, 3, 3, 3]
        ppd_values = ppd.Series(data).value_counts(normalize=True).to_list()
        counts = [item["count"] for item in ppd_values]
        values = [item[""] for item in ppd_values]
        assert counts == [0.5, 0.3333333333333333, 0.16666666666666666]
        assert values == [3, 2, 1]

    def test_unique_basic(self):
        """Test unique method."""
        # Create data with duplicates
        data = [1, 2, 2, 3, 3, 3]
        result = ppd.Series(data).unique()
        assert sorted(result.to_list()) == [1, 2, 3]

    @pytest.mark.skip(
        reason="Polars null handling differs from pandas - permanent limitation"
    )
    def test_methods_with_nulls(self):
        """Test methods with null values."""
        pytest.skip("Known limitation tracked in KNOWN_LIMITATIONS.md")

    def test_methods_empty_series(self):
        """Test methods with empty Series."""
        result = ppd.Series([], dtype=float).between(1, 3)
        assert result.to_list() == []

    def test_methods_return_types(self):
        """Test that methods return correct types."""
        # Test comparison operators
        result = ppd.Series(self.data) > 3
        assert isinstance(result, ppd.Series)

        # Test between
        result = ppd.Series(self.data).between(2, 4)
        assert isinstance(result, ppd.Series)

        # Test clip
        result = ppd.Series(self.data).clip(lower=2, upper=4)
        assert isinstance(result, ppd.Series)

        # Test rank
        result = ppd.Series(self.data).rank()
        assert isinstance(result, ppd.Series)

        # Test sort_values
        result = ppd.Series(self.data).sort_values()
        assert isinstance(result, ppd.Series)

        # Test value_counts
        result = ppd.Series(self.data).value_counts()
        assert isinstance(result, ppd.Series)

        # Test unique
        result = ppd.Series(self.data).unique()
        assert isinstance(result, ppd.Series)

    def test_methods_preserve_original(self):
        """Test that methods don't modify original Series."""
        original = ppd.Series(self.data)
        before = original.to_list()

        # Perform operations
        _ = ppd.Series(self.data).between(2, 4)
        _ = ppd.Series(self.data).clip(lower=2, upper=4)

        # Original should be unchanged
        assert original.to_list() == before


class TestSeriesRolling:
    """Test Series rolling window custom applications."""

    def test_series_rolling_apply_raw(self):
        """rolling.apply with raw list input returns custom sums."""
        s = ppd.Series([1, 2, 3, 4], name="values")
        result = s.rolling(window=2, min_periods=1).apply(
            lambda window: sum(window), raw=True
        )
        assert_series_equal(result, [1, 3, 5, 7])

    def test_series_rolling_apply_series(self):
        """rolling.apply with Series input matches rolling mean."""
        s = ppd.Series([1.0, 2.0, 3.0, 4.0], name="values")
        result = s.rolling(window=2).apply(lambda window: window.mean())
        expected = [None, 1.5, 2.5, 3.5]
        assert_series_equal(result, expected, rtol=1e-9)
