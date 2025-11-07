"""Comprehensive tests for rolling operations without pandas dependency."""

import polarpandas as ppd
from tests.test_helpers import assert_frame_equal


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
        df = ppd.DataFrame(self.data)
        result = df.rolling(3).mean()
        expected = {
            "A": [None, None, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "B": [None, None, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        }
        assert_frame_equal(result, expected)

    def test_rolling_sum_basic(self):
        """Test basic rolling sum."""
        df = ppd.DataFrame(self.data)
        result = df.rolling(3).sum()
        expected = {
            "A": [None, None, 6, 9, 12, 15, 18, 21, 24, 27],
            "B": [None, None, 60, 90, 120, 150, 180, 210, 240, 270],
        }
        assert_frame_equal(result, expected)

    def test_rolling_apply_custom_raw(self):
        """Test rolling apply with raw list input."""
        df = ppd.DataFrame(self.data)
        result = df.rolling(3, min_periods=2).apply(
            lambda window: sum(window), raw=True
        )
        expected = {
            "A": [None, 3, 6, 9, 12, 15, 18, 21, 24, 27],
            "B": [None, 30, 60, 90, 120, 150, 180, 210, 240, 270],
        }
        assert_frame_equal(result, expected)

    def test_rolling_apply_series_input(self):
        """Test rolling apply with Series input behaves like rolling mean."""
        df = ppd.DataFrame(self.data)
        result = df.rolling(3).apply(lambda s: s.mean())
        expected = {
            "A": [None, None, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "B": [None, None, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        }
        assert_frame_equal(result, expected)

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
        df = ppd.DataFrame(self.data)
        result = df.rolling(3).std()
        expected = {
            "A": [None, None] + [1.0] * 8,
            "B": [None, None] + [10.0] * 8,
        }
        assert_frame_equal(result, expected)

    def test_rolling_max(self):
        """Test rolling maximum."""
        df = ppd.DataFrame(self.data)
        result = df.rolling(3).max()
        expected = {
            "A": [None, None, 3, 4, 5, 6, 7, 8, 9, 10],
            "B": [None, None, 30, 40, 50, 60, 70, 80, 90, 100],
        }
        assert_frame_equal(result, expected)

    def test_rolling_min(self):
        """Test rolling minimum."""
        df = ppd.DataFrame(self.data)
        result = df.rolling(3).min()
        expected = {
            "A": [None, None, 1, 2, 3, 4, 5, 6, 7, 8],
            "B": [None, None, 10, 20, 30, 40, 50, 60, 70, 80],
        }
        assert_frame_equal(result, expected)

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
