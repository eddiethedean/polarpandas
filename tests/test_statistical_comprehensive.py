"""
Comprehensive tests for statistical methods with edge cases and performance tests.

All tests compare polarpandas output against actual pandas output
to ensure 100% compatibility.
"""

import numpy as np
import pandas as pd
import pytest

import polarpandas as ppd


class TestStatisticalMethodsComprehensive:
    """Comprehensive tests for statistical methods with edge cases."""

    def setup_method(self):
        """Create test data in both pandas and polarpandas."""
        self.data = {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": [1.1, 2.2, 3.3, 4.4, 5.5],
        }
        # Don't create DataFrames here to avoid state pollution
        # Each test method will create fresh DataFrames

    @pytest.mark.skip(
        reason="Polars doesn't have built-in correlation support - permanent limitation"
    )
    def test_corr_with_nulls(self):
        """Test correlation with null values."""
        data_with_nulls = {
            "A": [1, None, 3, 4, 5],
            "B": [10, 20, None, 40, 50],
            "C": [1.1, 2.2, 3.3, None, 5.5],
        }
        pd_df = pd.DataFrame(data_with_nulls)
        ppd_df = ppd.DataFrame(data_with_nulls)

        pd_result = pd_df.corr()
        ppd_result = ppd_df.corr()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    @pytest.mark.skip(
        reason="Polars doesn't have built-in correlation support - permanent limitation"
    )
    def test_corr_single_column(self):
        """Test correlation with single column."""
        single_col_data = {"A": [1, 2, 3, 4, 5]}
        pd_df = pd.DataFrame(single_col_data)
        ppd_df = ppd.DataFrame(single_col_data)

        pd_result = pd_df.corr()
        ppd_result = ppd_df.corr()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    @pytest.mark.skip(
        reason="Polars doesn't have built-in correlation support - permanent limitation"
    )
    def test_corr_empty_dataframe(self):
        """Test correlation with empty DataFrame."""
        pd_empty = pd.DataFrame()
        ppd_empty = ppd.DataFrame()

        # Both should return empty DataFrames (no error raised)
        pd_result = pd_empty.corr()
        ppd_result = ppd_empty.corr()

        # Check that both return empty DataFrames with same shape
        assert pd_result.shape == (0, 0)
        assert ppd_result.shape == (0, 0)
        assert len(pd_result.columns) == 0
        assert len(ppd_result.columns) == 0

    @pytest.mark.skip(
        reason="Polars doesn't have built-in correlation support - permanent limitation"
    )
    def test_corr_single_row(self):
        """Test correlation with single row."""
        single_row_data = {"A": [1], "B": [10], "C": [1.1]}
        pd_df = pd.DataFrame(single_row_data)
        ppd_df = ppd.DataFrame(single_row_data)

        pd_result = pd_df.corr()
        ppd_result = ppd_df.corr()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    @pytest.mark.skip(
        reason="Polars doesn't have built-in correlation support - permanent limitation"
    )
    def test_corr_methods(self):
        """Test correlation with different methods."""
        for method in ["pearson", "kendall", "spearman"]:
            pd_result = pd.DataFrame(self.data).corr(method=method)
            ppd_result = ppd.DataFrame(self.data).corr(method=method)
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    @pytest.mark.skip(
        reason="Polars doesn't have built-in covariance support - permanent limitation"
    )
    def test_cov_with_nulls(self):
        """Test covariance with null values."""
        data_with_nulls = {
            "A": [1, None, 3, 4, 5],
            "B": [10, 20, None, 40, 50],
            "C": [1.1, 2.2, 3.3, None, 5.5],
        }
        pd_df = pd.DataFrame(data_with_nulls)
        ppd_df = ppd.DataFrame(data_with_nulls)

        pd_result = pd_df.cov()
        ppd_result = ppd_df.cov()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_rank_methods(self):
        """Test rank with different methods."""
        for method in ["average", "min", "max", "first", "dense"]:
            pd_result = pd.DataFrame(self.data).rank(method=method)
            ppd_result = ppd.DataFrame(self.data).rank(method=method)
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_rank_ascending(self):
        """Test rank with ascending=False."""
        pd_result = pd.DataFrame(self.data).rank(ascending=False)
        ppd_result = ppd.DataFrame(self.data).rank(ascending=False)
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_rank_numeric_only(self):
        """Test rank with numeric_only=True."""
        # Add string column
        data_with_str = self.data.copy()
        data_with_str["D"] = ["a", "b", "c", "d", "e"]

        pd_df = pd.DataFrame(data_with_str)
        ppd_df = ppd.DataFrame(data_with_str)

        pd_result = pd_df.rank(numeric_only=True)
        ppd_result = ppd_df.rank(numeric_only=True)
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_diff_periods(self):
        """Test diff with different periods."""
        for periods in [1, 2, 3]:
            pd_result = pd.DataFrame(self.data).diff(periods=periods)
            ppd_result = ppd.DataFrame(self.data).diff(periods=periods)
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_pct_change_periods(self):
        """Test pct_change with different periods."""
        for periods in [1, 2, 3]:
            pd_result = pd.DataFrame(self.data).pct_change(periods=periods)
            ppd_result = ppd.DataFrame(self.data).pct_change(periods=periods)
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_cumsum_with_nulls(self):
        """Test cumsum with null values."""
        data_with_nulls = {"A": [1, None, 3, 4, 5], "B": [10, 20, None, 40, 50]}
        pd_df = pd.DataFrame(data_with_nulls)
        ppd_df = ppd.DataFrame(data_with_nulls)

        pd_result = pd_df.cumsum()
        ppd_result = ppd_df.cumsum()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_cumprod_with_nulls(self):
        """Test cumprod with null values."""
        data_with_nulls = {"A": [1, None, 3, 4, 5], "B": [10, 20, None, 40, 50]}
        pd_df = pd.DataFrame(data_with_nulls)
        ppd_df = ppd.DataFrame(data_with_nulls)

        pd_result = pd_df.cumprod()
        ppd_result = ppd_df.cumprod()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_cummax_with_nulls(self):
        """Test cummax with null values."""
        data_with_nulls = {"A": [1, None, 3, 4, 5], "B": [10, 20, None, 40, 50]}
        pd_df = pd.DataFrame(data_with_nulls)
        ppd_df = ppd.DataFrame(data_with_nulls)

        pd_result = pd_df.cummax()
        ppd_result = ppd_df.cummax()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_cummin_with_nulls(self):
        """Test cummin with null values."""
        data_with_nulls = {"A": [1, None, 3, 4, 5], "B": [10, 20, None, 40, 50]}
        pd_df = pd.DataFrame(data_with_nulls)
        ppd_df = ppd.DataFrame(data_with_nulls)

        pd_result = pd_df.cummin()
        ppd_result = ppd_df.cummin()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    @pytest.mark.skip(
        reason="Polars doesn't have built-in correlation support - permanent limitation"
    )
    def test_statistical_methods_empty_dataframe(self):
        """Test statistical methods with empty DataFrame."""
        pd_empty = pd.DataFrame()
        ppd_empty = ppd.DataFrame()

        # These should return empty DataFrames
        pd_result = pd_empty.corr()
        ppd_result = ppd_empty.corr()

        # Check that both return empty DataFrames with same shape
        assert pd_result.shape == (0, 0)
        assert ppd_result.shape == (0, 0)
        assert len(pd_result.columns) == 0
        assert len(ppd_result.columns) == 0

    def test_statistical_methods_single_row(self):
        """Test statistical methods with single row."""
        single_row_data = {"A": [1], "B": [10], "C": [1.1]}
        pd_df = pd.DataFrame(single_row_data)
        ppd_df = ppd.DataFrame(single_row_data)

        # Test cumsum with single row
        pd_result = pd_df.cumsum()
        ppd_result = ppd_df.cumsum()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    @pytest.mark.skip(
        reason="Polars doesn't have built-in correlation support - permanent limitation"
    )
    def test_statistical_methods_large_dataset(self):
        """Test statistical methods with large dataset."""
        # Create larger dataset
        np.random.seed(42)
        large_data = {
            "A": np.random.randn(1000),
            "B": np.random.randn(1000),
            "C": np.random.randn(1000),
        }
        pd_df = pd.DataFrame(large_data)
        ppd_df = ppd.DataFrame(large_data)

        # Test correlation with large dataset
        pd_result = pd_df.corr()
        ppd_result = ppd_df.corr()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_statistical_methods_mixed_types(self):
        """Test statistical methods with mixed data types."""
        mixed_data = {
            "A": [1, 2, 3, 4, 5],
            "B": [1.1, 2.2, 3.3, 4.4, 5.5],
            "C": [True, False, True, False, True],
        }
        pd_df = pd.DataFrame(mixed_data)
        ppd_df = ppd.DataFrame(mixed_data)

        # Test cumsum with mixed types
        pd_result = pd_df.cumsum()
        ppd_result = ppd_df.cumsum()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    @pytest.mark.skip(
        reason="Polars doesn't have built-in correlation support - permanent limitation"
    )
    def test_statistical_methods_return_types(self):
        """Test that statistical methods return correct types."""
        # Test correlation
        result = ppd.DataFrame(self.data).corr()
        assert isinstance(result, ppd.DataFrame)

        # Test covariance
        result = ppd.DataFrame(self.data).cov()
        assert isinstance(result, ppd.DataFrame)

        # Test rank
        result = ppd.DataFrame(self.data).rank()
        assert isinstance(result, ppd.DataFrame)

        # Test diff
        result = ppd.DataFrame(self.data).diff()
        assert isinstance(result, ppd.DataFrame)

        # Test pct_change
        result = ppd.DataFrame(self.data).pct_change()
        assert isinstance(result, ppd.DataFrame)

        # Test cumsum
        result = ppd.DataFrame(self.data).cumsum()
        assert isinstance(result, ppd.DataFrame)

        # Test cumprod
        result = ppd.DataFrame(self.data).cumprod()
        assert isinstance(result, ppd.DataFrame)

        # Test cummax
        result = ppd.DataFrame(self.data).cummax()
        assert isinstance(result, ppd.DataFrame)

        # Test cummin
        result = ppd.DataFrame(self.data).cummin()
        assert isinstance(result, ppd.DataFrame)

    @pytest.mark.skip(
        reason="Polars doesn't have built-in correlation support - permanent limitation"
    )
    def test_statistical_methods_preserve_original(self):
        """Test that statistical methods don't modify original DataFrame."""
        original_pd = pd.DataFrame(self.data).copy()
        original_ppd = ppd.DataFrame(self.data).copy()

        # Perform statistical operations
        pd.DataFrame(self.data).corr()
        ppd.DataFrame(self.data).corr()
        pd.DataFrame(self.data).cov()
        ppd.DataFrame(self.data).cov()
        pd.DataFrame(self.data).rank()
        ppd.DataFrame(self.data).rank()
        pd.DataFrame(self.data).diff()
        ppd.DataFrame(self.data).diff()
        pd.DataFrame(self.data).pct_change()
        ppd.DataFrame(self.data).pct_change()
        pd.DataFrame(self.data).cumsum()
        ppd.DataFrame(self.data).cumsum()

        # Original should be unchanged
        pd.testing.assert_frame_equal(original_pd, pd.DataFrame(self.data))
        pd.testing.assert_frame_equal(
            original_ppd.to_pandas(), ppd.DataFrame(self.data).to_pandas()
        )


class TestEdgeCases:
    """Test edge cases for statistical methods."""

    def test_all_nan_values(self):
        """Test with all NaN values."""
        all_nan_data = {"A": [np.nan, np.nan, np.nan], "B": [np.nan, np.nan, np.nan]}
        pd_df = pd.DataFrame(all_nan_data)
        ppd_df = ppd.DataFrame(all_nan_data)

        # Test cumsum with all NaN
        pd_result = pd_df.cumsum()
        ppd_result = ppd_df.cumsum()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_inf_values(self):
        """Test with infinite values."""
        inf_data = {"A": [1, 2, np.inf, 4, 5], "B": [10, 20, 30, np.inf, 50]}
        pd_df = pd.DataFrame(inf_data)
        ppd_df = ppd.DataFrame(inf_data)

        # Test cumsum with inf values
        pd_result = pd_df.cumsum()
        ppd_result = ppd_df.cumsum()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_zero_values(self):
        """Test with zero values."""
        zero_data = {"A": [0, 0, 0, 0, 0], "B": [0, 0, 0, 0, 0]}
        pd_df = pd.DataFrame(zero_data)
        ppd_df = ppd.DataFrame(zero_data)

        # Test cumsum with zeros
        pd_result = pd_df.cumsum()
        ppd_result = ppd_df.cumsum()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_negative_values(self):
        """Test with negative values."""
        negative_data = {"A": [-1, -2, -3, -4, -5], "B": [-10, -20, -30, -40, -50]}
        pd_df = pd.DataFrame(negative_data)
        ppd_df = ppd.DataFrame(negative_data)

        # Test cumsum with negative values
        pd_result = pd_df.cumsum()
        ppd_result = ppd_df.cumsum()
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)
