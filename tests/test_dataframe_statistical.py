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
        # Don't create DataFrames here to avoid state pollution
        # Each test method will create fresh DataFrames

    def test_nlargest_basic(self):
        """Test nlargest method."""
        pd_result = pd.DataFrame(self.data).nlargest(3, "A")
        ppd_result = ppd.DataFrame(self.data).nlargest(3, "A")
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_nlargest_multiple_columns(self):
        """Test nlargest with multiple columns."""
        pd_result = pd.DataFrame(self.data).nlargest(3, ["A", "B"])
        ppd_result = ppd.DataFrame(self.data).nlargest(3, ["A", "B"])
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_nsmallest_basic(self):
        """Test nsmallest method."""
        pd_result = pd.DataFrame(self.data).nsmallest(3, "A")
        ppd_result = ppd.DataFrame(self.data).nsmallest(3, "A")
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_nsmallest_multiple_columns(self):
        """Test nsmallest with multiple columns."""
        pd_result = pd.DataFrame(self.data).nsmallest(3, ["A", "B"])
        ppd_result = ppd.DataFrame(self.data).nsmallest(3, ["A", "B"])
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_corr_basic(self):
        """Test correlation matrix."""
        pd_result = pd.DataFrame(self.data).corr()
        ppd_result = ppd.DataFrame(self.data).corr()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_corr_method(self):
        """Test correlation with different method."""
        # Only test pearson (spearman not yet implemented)
        import pytest

        pd_result = pd.DataFrame(self.data).corr(method="pearson")
        ppd_result = ppd.DataFrame(self.data).corr(method="pearson")
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

        # Test that spearman raises NotImplementedError
        with pytest.raises(NotImplementedError):
            ppd.DataFrame(self.data).corr(method="spearman")

    def test_cov_basic(self):
        """Test covariance matrix."""
        pd_result = pd.DataFrame(self.data).cov()
        ppd_result = ppd.DataFrame(self.data).cov()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_rank_basic(self):
        """Test ranking."""
        pd_result = pd.DataFrame(self.data).rank()
        ppd_result = ppd.DataFrame(self.data).rank()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_rank_method(self):
        """Test ranking with different method."""
        pd_result = pd.DataFrame(self.data).rank(method="min")
        ppd_result = ppd.DataFrame(self.data).rank(method="min")
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
        pd_result = pd.DataFrame(self.data).diff()
        ppd_result = ppd.DataFrame(self.data).diff()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_diff_periods(self):
        """Test difference with different periods."""
        pd_result = pd.DataFrame(self.data).diff(periods=2)
        ppd_result = ppd.DataFrame(self.data).diff(periods=2)
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_pct_change_basic(self):
        """Test percentage change."""
        pd_result = pd.DataFrame(self.data).pct_change()
        ppd_result = ppd.DataFrame(self.data).pct_change()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_pct_change_periods(self):
        """Test percentage change with different periods."""
        pd_result = pd.DataFrame(self.data).pct_change(periods=2)
        ppd_result = ppd.DataFrame(self.data).pct_change(periods=2)
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_cumsum_basic(self):
        """Test cumulative sum."""
        pd_result = pd.DataFrame(self.data).cumsum()
        ppd_result = ppd.DataFrame(self.data).cumsum()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_cumprod_basic(self):
        """Test cumulative product."""
        pd_result = pd.DataFrame(self.data).cumprod()
        ppd_result = ppd.DataFrame(self.data).cumprod()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_cummax_basic(self):
        """Test cumulative maximum."""
        pd_result = pd.DataFrame(self.data).cummax()
        ppd_result = ppd.DataFrame(self.data).cummax()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_cummin_basic(self):
        """Test cumulative minimum."""
        pd_result = pd.DataFrame(self.data).cummin()
        ppd_result = ppd.DataFrame(self.data).cummin()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

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

    def test_statistical_return_types(self):
        """Test that statistical methods return correct types."""
        result = ppd.DataFrame(self.data).nlargest(3, "A")
        assert isinstance(result, ppd.DataFrame)

        result = ppd.DataFrame(self.data).corr()
        assert isinstance(result, ppd.DataFrame)

        result = ppd.DataFrame(self.data).cumsum()
        assert isinstance(result, ppd.DataFrame)

    def test_statistical_preserves_original(self):
        """Test that statistical methods don't modify original DataFrame."""
        original_pd = pd.DataFrame(self.data).copy()
        original_ppd = ppd.DataFrame(self.data).copy()

        # Perform statistical operations
        pd.DataFrame(self.data).nlargest(3, "A")
        ppd.DataFrame(self.data).nlargest(3, "A")
        pd.DataFrame(self.data).corr()
        ppd.DataFrame(self.data).corr()

        # Original should be unchanged
        pd.testing.assert_frame_equal(original_pd, pd.DataFrame(self.data))
        pd.testing.assert_frame_equal(
            original_ppd.to_pandas(), ppd.DataFrame(self.data).to_pandas()
        )
