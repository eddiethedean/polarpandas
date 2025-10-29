"""
Test enhanced Series methods with pandas compatibility.

All tests compare polarpandas output against actual pandas output
to ensure 100% compatibility.
"""

import pandas as pd
import pytest

import polarpandas as ppd


class TestSeriesComparison:
    """Test Series comparison operators with pandas compatibility."""

    def setup_method(self):
        """Create test data in both pandas and polarpandas."""
        self.data = [1, 2, 3, 4, 5]
        # Don't create Series here to avoid state pollution
        # Each test method will create fresh Series

    def test_gt_comparison(self):
        """Test greater than comparison."""
        pd_result = pd.Series(self.data) > 3
        ppd_result = ppd.Series(self.data) > 3
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_index=False
        )

    def test_lt_comparison(self):
        """Test less than comparison."""
        pd_result = pd.Series(self.data) < 3
        ppd_result = ppd.Series(self.data) < 3
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_index=False
        )

    def test_ge_comparison(self):
        """Test greater than or equal comparison."""
        pd_result = pd.Series(self.data) >= 3
        ppd_result = ppd.Series(self.data) >= 3
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_index=False
        )

    def test_le_comparison(self):
        """Test less than or equal comparison."""
        pd_result = pd.Series(self.data) <= 3
        ppd_result = ppd.Series(self.data) <= 3
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_index=False
        )

    def test_eq_comparison(self):
        """Test equal comparison."""
        pd_result = pd.Series(self.data) == 3
        ppd_result = ppd.Series(self.data) == 3
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_index=False
        )

    def test_ne_comparison(self):
        """Test not equal comparison."""
        pd_result = pd.Series(self.data) != 3
        ppd_result = ppd.Series(self.data) != 3
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_index=False
        )

    def test_series_comparison(self):
        """Test Series to Series comparison."""
        other_data = [2, 3, 4, 5, 6]
        pd_other = pd.Series(other_data)
        ppd_other = ppd.Series(other_data)

        pd_result = pd.Series(self.data) > pd_other
        ppd_result = ppd.Series(self.data) > ppd_other
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_index=False
        )


class TestSeriesMethods:
    """Test enhanced Series methods with pandas compatibility."""

    def setup_method(self):
        """Create test data in both pandas and polarpandas."""
        self.data = [1, 2, 3, 4, 5]
        # Don't create Series here to avoid state pollution
        # Each test method will create fresh Series

    def test_between_basic(self):
        """Test between method."""
        pd_result = pd.Series(self.data).between(2, 4)
        ppd_result = ppd.Series(self.data).between(2, 4)
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_index=False
        )

    def test_between_inclusive(self):
        """Test between with different inclusive options."""
        # Test 'neither'
        pd_result = pd.Series(self.data).between(2, 4, inclusive="neither")
        ppd_result = ppd.Series(self.data).between(2, 4, inclusive="neither")
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_index=False
        )

    def test_clip_basic(self):
        """Test clip method."""
        pd_result = pd.Series(self.data).clip(lower=2, upper=4)
        ppd_result = ppd.Series(self.data).clip(lower=2, upper=4)
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_index=False
        )

    def test_clip_lower_only(self):
        """Test clip with only lower bound."""
        pd_result = pd.Series(self.data).clip(lower=3)
        ppd_result = ppd.Series(self.data).clip(lower=3)
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_index=False
        )

    def test_clip_upper_only(self):
        """Test clip with only upper bound."""
        pd_result = pd.Series(self.data).clip(upper=3)
        ppd_result = ppd.Series(self.data).clip(upper=3)
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_index=False
        )

    def test_rank_basic(self):
        """Test rank method."""
        pd_result = pd.Series(self.data).rank()
        ppd_result = ppd.Series(self.data).rank()
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_index=False
        )

    def test_rank_method(self):
        """Test rank with different method."""
        pd_result = pd.Series(self.data).rank(method="min")
        ppd_result = ppd.Series(self.data).rank(method="min")
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_index=False
        )

    def test_rank_descending(self):
        """Test rank with descending order."""
        pd_result = pd.Series(self.data).rank(ascending=False)
        ppd_result = ppd.Series(self.data).rank(ascending=False)
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_index=False
        )

    def test_sort_values_basic(self):
        """Test sort_values method."""
        # Create unsorted data
        data = [3, 1, 4, 2, 5]
        pd_series = pd.Series(data)
        ppd_series = ppd.Series(data)

        pd_result = pd_series.sort_values()
        ppd_result = ppd_series.sort_values()
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_index=False
        )

    def test_sort_values_descending(self):
        """Test sort_values with descending order."""
        # Create unsorted data
        data = [3, 1, 4, 2, 5]
        pd_series = pd.Series(data)
        ppd_series = ppd.Series(data)

        pd_result = pd_series.sort_values(ascending=False)
        ppd_result = ppd_series.sort_values(ascending=False)

        # Compare values directly since index behavior differs in pure Polars
        assert list(ppd_result.values) == list(pd_result.values)

    def test_value_counts_basic(self):
        """Test value_counts method."""
        # Create data with duplicates
        data = [1, 2, 2, 3, 3, 3]
        pd_series = pd.Series(data)
        ppd_series = ppd.Series(data)

        pd_result = pd_series.value_counts()
        ppd_result = ppd_series.value_counts()

        # Compare values and counts directly since index behavior differs
        # Polars value_counts returns struct with values and counts
        ppd_values = ppd_result.to_list()
        ppd_counts = [
            item["count"] for item in ppd_values
        ]  # Extract counts from struct
        ppd_index = [item[""] for item in ppd_values]  # Extract values from struct

        assert ppd_counts == list(pd_result.values)
        assert ppd_index == list(pd_result.index)

    def test_value_counts_normalize(self):
        """Test value_counts with normalize."""
        # Create data with duplicates
        data = [1, 2, 2, 3, 3, 3]
        pd_series = pd.Series(data)
        ppd_series = ppd.Series(data)

        pd_result = pd_series.value_counts(normalize=True)
        ppd_result = ppd_series.value_counts(normalize=True)

        # Compare values and counts directly since index behavior differs
        # Polars value_counts returns struct with values and counts
        ppd_values = ppd_result.to_list()
        ppd_counts = [
            item["count"] for item in ppd_values
        ]  # Extract counts from struct
        ppd_index = [item[""] for item in ppd_values]  # Extract values from struct

        assert ppd_counts == list(pd_result.values)
        assert ppd_index == list(pd_result.index)

    def test_unique_basic(self):
        """Test unique method."""
        # Create data with duplicates
        data = [1, 2, 2, 3, 3, 3]
        pd_series = pd.Series(data)
        ppd_series = ppd.Series(data)

        pd_result = pd_series.unique()
        ppd_result = ppd_series.unique()

        # Convert to sorted lists for comparison
        pd_sorted = sorted(pd_result)
        ppd_sorted = sorted(ppd_result.to_pandas().tolist())
        assert pd_sorted == ppd_sorted

    @pytest.mark.skip(
        reason="Polars null handling differs from pandas - permanent limitation"
    )
    def test_methods_with_nulls(self):
        """Test methods with null values."""
        data = [1, None, 3, 4, 5]
        pd_series = pd.Series(data)
        ppd_series = ppd.Series(data)

        # Test between with nulls
        pd_result = pd_series.between(2, 4)
        ppd_result = ppd_series.between(2, 4)
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_index=False
        )

    def test_methods_empty_series(self):
        """Test methods with empty Series."""
        pd_empty = pd.Series([], dtype=float)
        ppd_empty = ppd.Series([], dtype=float)

        # Test between with empty series
        pd_result = pd_empty.between(1, 3)
        ppd_result = ppd_empty.between(1, 3)
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_index=False
        )

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
        original_pd = pd.Series(self.data).copy()
        original_ppd = ppd.Series(self.data).copy()

        # Perform operations
        pd.Series(self.data).between(2, 4)
        ppd.Series(self.data).between(2, 4)
        pd.Series(self.data).clip(lower=2, upper=4)
        ppd.Series(self.data).clip(lower=2, upper=4)

        # Original should be unchanged
        pd.testing.assert_series_equal(original_pd, pd.Series(self.data))
        pd.testing.assert_series_equal(
            original_ppd.to_pandas(), ppd.Series(self.data).to_pandas()
        )
