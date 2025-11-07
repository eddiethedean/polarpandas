"""
Test statistical methods for DataFrame.

All tests use pre-computed expected values that were generated using pandas.
This allows testing without a pandas runtime dependency.
"""

import pytest

import polarpandas as ppd
from tests.test_helpers import assert_frame_equal, load_expected


class TestDataFrameStatistical:
    """Test statistical methods."""

    def setup_method(self) -> None:
        """Create test data."""
        self.data = {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": [1.1, 2.2, 3.3, 4.4, 5.5],
        }

    def test_nlargest_basic(self) -> None:
        """Test nlargest method."""
        expected = load_expected("test_dataframe_statistical", "test_nlargest_basic")
        ppd_result = ppd.DataFrame(self.data).nlargest(3, "A")
        assert_frame_equal(ppd_result, expected)

    def test_nlargest_multiple_columns(self) -> None:
        """Test nlargest with multiple columns."""
        expected = load_expected(
            "test_dataframe_statistical", "test_nlargest_multiple_columns"
        )
        ppd_result = ppd.DataFrame(self.data).nlargest(3, ["A", "B"])
        assert_frame_equal(ppd_result, expected)

    def test_nsmallest_basic(self) -> None:
        """Test nsmallest method."""
        expected = load_expected("test_dataframe_statistical", "test_nsmallest_basic")
        ppd_result = ppd.DataFrame(self.data).nsmallest(3, "A")
        assert_frame_equal(ppd_result, expected)

    def test_nsmallest_multiple_columns(self) -> None:
        """Test nsmallest with multiple columns."""
        expected = load_expected(
            "test_dataframe_statistical", "test_nsmallest_multiple_columns"
        )
        ppd_result = ppd.DataFrame(self.data).nsmallest(3, ["A", "B"])
        assert_frame_equal(ppd_result, expected)

    def test_corr_basic(self) -> None:
        """Test correlation matrix."""
        expected = load_expected("test_dataframe_statistical", "test_corr_basic")
        ppd_result = ppd.DataFrame(self.data).corr()
        assert_frame_equal(ppd_result, expected, rtol=1e-5)

    def test_corr_method(self) -> None:
        """Test correlation with different method."""
        # Only test pearson (spearman not yet implemented)
        expected = load_expected("test_dataframe_statistical", "test_corr_method")
        ppd_result = ppd.DataFrame(self.data).corr(method="pearson")
        assert_frame_equal(ppd_result, expected, rtol=1e-5)

        # Test that spearman raises NotImplementedError
        with pytest.raises(NotImplementedError):
            ppd.DataFrame(self.data).corr(method="spearman")

    def test_cov_basic(self) -> None:
        """Test covariance matrix."""
        expected = load_expected("test_dataframe_statistical", "test_cov_basic")
        ppd_result = ppd.DataFrame(self.data).cov()
        assert_frame_equal(ppd_result, expected, rtol=1e-5)

    def test_rank_basic(self) -> None:
        """Test ranking."""
        expected = load_expected("test_dataframe_statistical", "test_rank_basic")
        ppd_result = ppd.DataFrame(self.data).rank()
        assert_frame_equal(ppd_result, expected)

    def test_rank_method(self) -> None:
        """Test ranking with different method."""
        expected = load_expected("test_dataframe_statistical", "test_rank_method")
        ppd_result = ppd.DataFrame(self.data).rank(method="min")
        assert_frame_equal(ppd_result, expected)

    def test_rank_numeric_only(self) -> None:
        """Test ranking with numeric_only=True."""
        # Add a string column
        data_with_str = self.data.copy()
        data_with_str["D"] = ["a", "b", "c", "d", "e"]

        expected = load_expected("test_dataframe_statistical", "test_rank_numeric_only")
        ppd_result = ppd.DataFrame(data_with_str).rank(numeric_only=True)
        assert_frame_equal(ppd_result, expected)

    def test_diff_basic(self) -> None:
        """Test difference calculation."""
        expected = load_expected("test_dataframe_statistical", "test_diff_basic")
        ppd_result = ppd.DataFrame(self.data).diff()
        assert_frame_equal(ppd_result, expected)

    def test_diff_periods(self) -> None:
        """Test difference with different periods."""
        expected = load_expected("test_dataframe_statistical", "test_diff_periods")
        ppd_result = ppd.DataFrame(self.data).diff(periods=2)
        assert_frame_equal(ppd_result, expected)

    def test_pct_change_basic(self) -> None:
        """Test percentage change."""
        expected = load_expected("test_dataframe_statistical", "test_pct_change_basic")
        ppd_result = ppd.DataFrame(self.data).pct_change()
        assert_frame_equal(ppd_result, expected, rtol=1e-5)

    def test_pct_change_periods(self) -> None:
        """Test percentage change with different periods."""
        expected = load_expected(
            "test_dataframe_statistical", "test_pct_change_periods"
        )
        ppd_result = ppd.DataFrame(self.data).pct_change(periods=2)
        assert_frame_equal(ppd_result, expected, rtol=1e-5)

    def test_cumsum_basic(self) -> None:
        """Test cumulative sum."""
        expected = load_expected("test_dataframe_statistical", "test_cumsum_basic")
        ppd_result = ppd.DataFrame(self.data).cumsum()
        assert_frame_equal(ppd_result, expected)

    def test_cumprod_basic(self) -> None:
        """Test cumulative product."""
        expected = load_expected("test_dataframe_statistical", "test_cumprod_basic")
        ppd_result = ppd.DataFrame(self.data).cumprod()
        assert_frame_equal(ppd_result, expected)

    def test_cummax_basic(self) -> None:
        """Test cumulative maximum."""
        expected = load_expected("test_dataframe_statistical", "test_cummax_basic")
        ppd_result = ppd.DataFrame(self.data).cummax()
        assert_frame_equal(ppd_result, expected)

    def test_cummin_basic(self) -> None:
        """Test cumulative minimum."""
        expected = load_expected("test_dataframe_statistical", "test_cummin_basic")
        ppd_result = ppd.DataFrame(self.data).cummin()
        assert_frame_equal(ppd_result, expected)

    def test_statistical_with_nulls(self) -> None:
        """Test statistical methods with null values."""
        data_with_nulls = {
            "A": [1, None, 3, 4, 5],
            "B": [10, 20, None, 40, 50],
            "C": [1.1, 2.2, 3.3, None, 5.5],
        }
        ppd_df = ppd.DataFrame(data_with_nulls)

        # Test correlation with nulls
        expected = load_expected(
            "test_dataframe_statistical", "test_statistical_with_nulls"
        )
        ppd_result = ppd_df.corr()
        assert_frame_equal(ppd_result, expected, rtol=1e-5)

    def test_statistical_empty_dataframe(self) -> None:
        """Test statistical methods with empty DataFrame."""
        ppd_empty = ppd.DataFrame()

        # These should raise appropriate errors
        with pytest.raises((ValueError, KeyError)):
            ppd_empty.nlargest(3, "A")

    def test_statistical_single_row(self) -> None:
        """Test statistical methods with single row."""
        data_single = {"A": [1], "B": [10], "C": [1.1]}
        ppd_df = ppd.DataFrame(data_single)

        # Test nlargest with single row
        expected = load_expected(
            "test_dataframe_statistical", "test_statistical_single_row"
        )
        ppd_result = ppd_df.nlargest(1, "A")
        assert_frame_equal(ppd_result, expected)

    def test_statistical_return_types(self) -> None:
        """Test that statistical methods return correct types."""
        result = ppd.DataFrame(self.data).nlargest(3, "A")
        assert isinstance(result, ppd.DataFrame)

        result = ppd.DataFrame(self.data).corr()
        assert isinstance(result, ppd.DataFrame)

        result = ppd.DataFrame(self.data).cumsum()
        assert isinstance(result, ppd.DataFrame)

    def test_statistical_preserves_original(self) -> None:
        """Test that statistical methods don't modify original DataFrame."""
        original_data = self.data.copy()
        df = ppd.DataFrame(self.data)

        # Perform statistical operations
        df.nlargest(3, "A")
        df.corr()

        # Original data should be unchanged
        assert self.data == original_data
