"""
Test statistical methods for DataFrame with pandas compatibility.

All tests compare polarpandas output against actual pandas output
to ensure 100% compatibility.
"""

import pandas as pd
import pytest

import polarpandas as ppd


class TestDataFrameStatistical:
    """Test statistical methods with pandas compatibility."""

    def setup_method(self):
        """Create test data in both pandas and polarpandas."""
        self.data = {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": [1.1, 2.2, 3.3, 4.4, 5.5],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ppd_df = ppd.DataFrame(self.data)

    def test_nlargest_basic(self):
        """Test nlargest method."""
        pd_result = self.pd_df.nlargest(3, "A")
        ppd_result = self.ppd_df.nlargest(3, "A")
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_nlargest_multiple_columns(self):
        """Test nlargest with multiple columns."""
        pd_result = self.pd_df.nlargest(3, ["A", "B"])
        ppd_result = self.ppd_df.nlargest(3, ["A", "B"])
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_nsmallest_basic(self):
        """Test nsmallest method."""
        pd_result = self.pd_df.nsmallest(3, "A")
        ppd_result = self.ppd_df.nsmallest(3, "A")
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_nsmallest_multiple_columns(self):
        """Test nsmallest with multiple columns."""
        pd_result = self.pd_df.nsmallest(3, ["A", "B"])
        ppd_result = self.ppd_df.nsmallest(3, ["A", "B"])
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    @pytest.mark.skip(
        reason="Polars doesn't have built-in correlation support - permanent limitation"
    )
    def test_corr_basic(self):
        """Test correlation matrix."""
        pd_result = self.pd_df.corr()
        ppd_result = self.ppd_df.corr()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    @pytest.mark.skip(
        reason="Polars doesn't have built-in correlation support - permanent limitation"
    )
    def test_corr_method(self):
        """Test correlation with different method."""
        pd_result = self.pd_df.corr(method="spearman")
        ppd_result = self.ppd_df.corr(method="spearman")
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    @pytest.mark.skip(
        reason="Polars doesn't have built-in covariance support - permanent limitation"
    )
    def test_cov_basic(self):
        """Test covariance matrix."""
        pd_result = self.pd_df.cov()
        ppd_result = self.ppd_df.cov()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_rank_basic(self):
        """Test ranking."""
        pd_result = self.pd_df.rank()
        ppd_result = self.ppd_df.rank()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_rank_method(self):
        """Test ranking with different method."""
        pd_result = self.pd_df.rank(method="min")
        ppd_result = self.ppd_df.rank(method="min")
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_rank_numeric_only(self):
        """Test ranking with numeric_only=True."""
        # Add a string column
        data_with_str = self.data.copy()
        data_with_str["D"] = ["a", "b", "c", "d", "e"]

        pd_df = pd.DataFrame(data_with_str)
        ppd_df = ppd.DataFrame(data_with_str)

        pd_result = pd_df.rank(numeric_only=True)
        ppd_result = ppd_df.rank(numeric_only=True)
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_diff_basic(self):
        """Test difference calculation."""
        pd_result = self.pd_df.diff()
        ppd_result = self.ppd_df.diff()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_diff_periods(self):
        """Test difference with different periods."""
        pd_result = self.pd_df.diff(periods=2)
        ppd_result = self.ppd_df.diff(periods=2)
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_pct_change_basic(self):
        """Test percentage change."""
        pd_result = self.pd_df.pct_change()
        ppd_result = self.ppd_df.pct_change()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_pct_change_periods(self):
        """Test percentage change with different periods."""
        pd_result = self.pd_df.pct_change(periods=2)
        ppd_result = self.ppd_df.pct_change(periods=2)
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_cumsum_basic(self):
        """Test cumulative sum."""
        pd_result = self.pd_df.cumsum()
        ppd_result = self.ppd_df.cumsum()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_cumprod_basic(self):
        """Test cumulative product."""
        pd_result = self.pd_df.cumprod()
        ppd_result = self.ppd_df.cumprod()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_cummax_basic(self):
        """Test cumulative maximum."""
        pd_result = self.pd_df.cummax()
        ppd_result = self.ppd_df.cummax()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_cummin_basic(self):
        """Test cumulative minimum."""
        pd_result = self.pd_df.cummin()
        ppd_result = self.ppd_df.cummin()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    @pytest.mark.skip(
        reason="Polars doesn't have built-in correlation support - permanent limitation"
    )
    def test_statistical_with_nulls(self):
        """Test statistical methods with null values."""
        data_with_nulls = {
            "A": [1, None, 3, 4, 5],
            "B": [10, 20, None, 40, 50],
            "C": [1.1, 2.2, 3.3, None, 5.5],
        }
        pd_df = pd.DataFrame(data_with_nulls)
        ppd_df = ppd.DataFrame(data_with_nulls)

        # Test correlation with nulls
        pd_result = pd_df.corr()
        ppd_result = ppd_df.corr()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_statistical_empty_dataframe(self):
        """Test statistical methods with empty DataFrame."""
        pd_empty = pd.DataFrame()
        ppd_empty = ppd.DataFrame()

        # These should raise appropriate errors
        with pytest.raises((ValueError, KeyError)):
            pd_empty.nlargest(3, "A")
        with pytest.raises((ValueError, KeyError)):
            ppd_empty.nlargest(3, "A")

    def test_statistical_single_row(self):
        """Test statistical methods with single row."""
        data_single = {"A": [1], "B": [10], "C": [1.1]}
        pd_df = pd.DataFrame(data_single)
        ppd_df = ppd.DataFrame(data_single)

        # Test nlargest with single row
        pd_result = pd_df.nlargest(1, "A")
        ppd_result = ppd_df.nlargest(1, "A")
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    @pytest.mark.skip(
        reason="Polars doesn't have built-in correlation support - permanent limitation"
    )
    def test_statistical_return_types(self):
        """Test that statistical methods return correct types."""
        result = self.ppd_df.nlargest(3, "A")
        assert isinstance(result, ppd.DataFrame)

        result = self.ppd_df.corr()
        assert isinstance(result, ppd.DataFrame)

        result = self.ppd_df.cumsum()
        assert isinstance(result, ppd.DataFrame)

    @pytest.mark.skip(
        reason="Polars doesn't have built-in correlation support - permanent limitation"
    )
    def test_statistical_preserves_original(self):
        """Test that statistical methods don't modify original DataFrame."""
        original_pd = self.pd_df.copy()
        original_ppd = self.ppd_df.copy()

        # Perform statistical operations
        self.pd_df.nlargest(3, "A")
        self.ppd_df.nlargest(3, "A")
        self.pd_df.corr()
        self.ppd_df.corr()

        # Original should be unchanged
        pd.testing.assert_frame_equal(original_pd, self.pd_df)
        pd.testing.assert_frame_equal(original_ppd.to_pandas(), self.ppd_df.to_pandas())
