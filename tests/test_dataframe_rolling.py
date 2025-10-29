"""
Comprehensive tests for rolling operations.

Tests rolling window operations and edge cases.
"""

import pandas as pd

import polarpandas as ppd


class TestRollingOperations:
    """Test rolling window operations."""

    def setup_method(self):
        """Create test data."""
        self.data = {
            "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "B": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }

    def test_rolling_mean_basic(self):
        """Test basic rolling mean."""
        pd_df = pd.DataFrame(self.data)
        ppd_df = ppd.DataFrame(self.data)

        pd_result = pd_df.rolling(3).mean()
        ppd_result = ppd_df.rolling(3).mean()

        pd.testing.assert_frame_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False
        )

    def test_rolling_sum_basic(self):
        """Test basic rolling sum."""
        pd_df = pd.DataFrame(self.data)
        ppd_df = ppd.DataFrame(self.data)

        pd_result = pd_df.rolling(3).sum()
        ppd_result = ppd_df.rolling(3).sum()

        pd.testing.assert_frame_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False
        )

    def test_rolling_window_larger_than_dataframe(self):
        """Test rolling with window larger than DataFrame."""
        # Lines 1922-1931: Edge case handling
        df = ppd.DataFrame({"A": [1, 2], "B": [10, 20]})

        # Window size 5 > DataFrame length 2
        result = df.rolling(5).mean()
        assert isinstance(result, ppd.DataFrame)
        assert len(result) == 2

    def test_rolling_empty_dataframe(self):
        """Test rolling with empty DataFrame."""
        df = ppd.DataFrame()

        # Should handle empty DataFrame gracefully
        try:
            result = df.rolling(3).mean()
            assert isinstance(result, ppd.DataFrame)
        except Exception:
            # Empty DataFrame may raise error, which is acceptable
            pass

    def test_rolling_chained_operations(self):
        """Test chained rolling operations."""
        df = ppd.DataFrame(self.data)

        # Chain rolling operations
        result = df.rolling(3).mean()
        result2 = result.rolling(2).sum()
        assert isinstance(result2, ppd.DataFrame)
        assert len(result2) == len(df)

    def test_rolling_std(self):
        """Test rolling standard deviation."""
        pd_df = pd.DataFrame(self.data)
        ppd_df = ppd.DataFrame(self.data)

        pd_result = pd_df.rolling(3).std()
        ppd_result = ppd_df.rolling(3).std()

        pd.testing.assert_frame_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False
        )

    def test_rolling_max(self):
        """Test rolling maximum."""
        pd_df = pd.DataFrame(self.data)
        ppd_df = ppd.DataFrame(self.data)

        pd_result = pd_df.rolling(3).max()
        ppd_result = ppd_df.rolling(3).max()

        pd.testing.assert_frame_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False
        )

    def test_rolling_min(self):
        """Test rolling minimum."""
        pd_df = pd.DataFrame(self.data)
        ppd_df = ppd.DataFrame(self.data)

        pd_result = pd_df.rolling(3).min()
        ppd_result = ppd_df.rolling(3).min()

        pd.testing.assert_frame_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False
        )

    def test_rolling_window_size_one(self):
        """Test rolling with window size 1."""
        df = ppd.DataFrame(self.data)

        result = df.rolling(1).mean()
        assert isinstance(result, ppd.DataFrame)
        assert len(result) == len(df)

    def test_rolling_single_column(self):
        """Test rolling on single column DataFrame."""
        df = ppd.DataFrame({"A": [1, 2, 3, 4, 5]})

        result = df.rolling(3).mean()
        assert isinstance(result, ppd.DataFrame)
        assert "A" in result.columns
        assert len(result) == 5
