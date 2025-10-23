"""
Test enhanced string and datetime accessors with pandas compatibility.

All tests compare polarpandas output against actual pandas output
to ensure 100% compatibility.
"""

from datetime import datetime

import pandas as pd
import pytest

import polarpandas as ppd


class TestStringAccessorEnhanced:
    """Test enhanced string accessor methods with pandas compatibility."""

    def setup_method(self):
        """Create test data in both pandas and polarpandas."""
        self.data = ["hello world", "test string", "another example", "final test"]
        self.pd_series = pd.Series(self.data)
        self.ppd_series = ppd.Series(self.data)

    def test_split_basic(self):
        """Test split method."""
        pd_result = self.pd_series.str.split()
        ppd_result = self.ppd_series.str.split()
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_split_with_separator(self):
        """Test split with specific separator."""
        pd_result = self.pd_series.str.split(" ")
        ppd_result = self.ppd_series.str.split(" ")
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_split_expand(self):
        """Test split with expand=True."""
        pd_result = self.pd_series.str.split(" ", expand=True)
        ppd_result = self.ppd_series.str.split(" ", expand=True)
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_extract_basic(self):
        """Test extract method."""
        # Create data with patterns
        data = ["hello123", "test456", "world789"]
        pd_series = pd.Series(data)
        ppd_series = ppd.Series(data)

        pd_result = pd_series.str.extract(r"(\d+)")
        ppd_result = ppd_series.str.extract(r"(\d+)")
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_slice_basic(self):
        """Test slice method."""
        pd_result = self.pd_series.str.slice(0, 5)
        ppd_result = self.ppd_series.str.slice(0, 5)
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_slice_with_step(self):
        """Test slice with step."""
        pd_result = self.pd_series.str.slice(0, 10, 2)
        ppd_result = self.ppd_series.str.slice(0, 10, 2)
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_contains_basic(self):
        """Test contains method."""
        pd_result = self.pd_series.str.contains("test")
        ppd_result = self.ppd_series.str.contains("test")
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_startswith_basic(self):
        """Test startswith method."""
        pd_result = self.pd_series.str.startswith("h")
        ppd_result = self.ppd_series.str.startswith("h")
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_endswith_basic(self):
        """Test endswith method."""
        pd_result = self.pd_series.str.endswith("d")
        ppd_result = self.ppd_series.str.endswith("d")
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_string_methods_with_nulls(self):
        """Test string methods with null values."""
        data = ["hello", None, "world", ""]
        pd_series = pd.Series(data)
        ppd_series = ppd.Series(data)

        # Test contains with nulls
        pd_result = pd_series.str.contains("o")
        ppd_result = ppd_series.str.contains("o")
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_string_methods_empty_series(self):
        """Test string methods with empty Series."""
        pd_empty = pd.Series([], dtype=str)
        ppd_empty = ppd.Series([], dtype=str)

        # Test contains with empty series
        pd_result = pd_empty.str.contains("test")
        ppd_result = ppd_empty.str.contains("test")
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )


class TestDatetimeAccessorEnhanced:
    """Test enhanced datetime accessor methods with pandas compatibility."""

    def setup_method(self):
        """Create test data in both pandas and polarpandas."""
        self.data = [
            datetime(2023, 1, 15, 10, 30, 45),
            datetime(2023, 6, 20, 14, 15, 30),
            datetime(2023, 12, 31, 23, 59, 59),
        ]
        self.pd_series = pd.Series(self.data)
        self.ppd_series = ppd.Series(self.data)

    @pytest.mark.skip(
        reason="Known limitation: Polars doesn't have direct equivalent to pandas date property that returns date objects"
    )
    def test_date_property(self):
        """Test date property."""
        pd_result = self.pd_series.dt.date
        ppd_result = self.ppd_series.dt.date
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_time_property(self):
        """Test time property."""
        pd_result = self.pd_series.dt.time
        ppd_result = self.ppd_series.dt.time
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_dayofweek_property(self):
        """Test dayofweek property."""
        pd_result = self.pd_series.dt.dayofweek
        ppd_result = self.ppd_series.dt.dayofweek
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_dayofyear_property(self):
        """Test dayofyear property."""
        pd_result = self.pd_series.dt.dayofyear
        ppd_result = self.ppd_series.dt.dayofyear
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_quarter_property(self):
        """Test quarter property."""
        pd_result = self.pd_series.dt.quarter
        ppd_result = self.ppd_series.dt.quarter
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_is_month_start_property(self):
        """Test is_month_start property."""
        pd_result = self.pd_series.dt.is_month_start
        ppd_result = self.ppd_series.dt.is_month_start
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_is_month_end_property(self):
        """Test is_month_end property."""
        pd_result = self.pd_series.dt.is_month_end
        ppd_result = self.ppd_series.dt.is_month_end
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_is_quarter_start_property(self):
        """Test is_quarter_start property."""
        pd_result = self.pd_series.dt.is_quarter_start
        ppd_result = self.ppd_series.dt.is_quarter_start
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_is_quarter_end_property(self):
        """Test is_quarter_end property."""
        pd_result = self.pd_series.dt.is_quarter_end
        ppd_result = self.ppd_series.dt.is_quarter_end
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_is_year_start_property(self):
        """Test is_year_start property."""
        pd_result = self.pd_series.dt.is_year_start
        ppd_result = self.ppd_series.dt.is_year_start
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_is_year_end_property(self):
        """Test is_year_end property."""
        pd_result = self.pd_series.dt.is_year_end
        ppd_result = self.ppd_series.dt.is_year_end
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_floor_method(self):
        """Test floor method."""
        pd_result = self.pd_series.dt.floor("D")
        ppd_result = self.ppd_series.dt.floor("D")
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_ceil_method(self):
        """Test ceil method."""
        pd_result = self.pd_series.dt.ceil("D")
        ppd_result = self.ppd_series.dt.ceil("D")
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_round_method(self):
        """Test round method."""
        pd_result = self.pd_series.dt.round("H")
        ppd_result = self.ppd_series.dt.round("H")
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_datetime_methods_with_nulls(self):
        """Test datetime methods with null values."""
        data = [
            datetime(2023, 1, 15, 10, 30, 45),
            None,
            datetime(2023, 12, 31, 23, 59, 59),
        ]
        pd_series = pd.Series(data)
        ppd_series = ppd.Series(data)

        # Test year with nulls
        pd_result = pd_series.dt.year
        ppd_result = ppd_series.dt.year
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    @pytest.mark.skip(
        reason="Polars datetime accessor with empty series differs from pandas - permanent limitation"
    )
    def test_datetime_methods_empty_series(self):
        """Test datetime methods with empty Series."""
        pd_empty = pd.Series([], dtype="datetime64[ns]")
        ppd_empty = ppd.Series([], dtype="datetime64[ns]")

        # Test year with empty series
        pd_result = pd_empty.dt.year
        ppd_result = ppd_empty.dt.year
        pd.testing.assert_series_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False, check_names=False
        )

    def test_datetime_methods_return_types(self):
        """Test that datetime methods return correct types."""
        # Test properties
        result = self.ppd_series.dt.year
        assert isinstance(result, ppd.Series)

        result = self.ppd_series.dt.month
        assert isinstance(result, ppd.Series)

        result = self.ppd_series.dt.day
        assert isinstance(result, ppd.Series)

        # Test methods
        result = self.ppd_series.dt.floor("D")
        assert isinstance(result, ppd.Series)

        result = self.ppd_series.dt.ceil("D")
        assert isinstance(result, ppd.Series)

        result = self.ppd_series.dt.round("H")
        assert isinstance(result, ppd.Series)

    def test_datetime_methods_preserve_original(self):
        """Test that datetime methods don't modify original Series."""
        original_pd = self.pd_series.copy()
        original_ppd = self.ppd_series.copy()

        # Perform operations
        _ = self.pd_series.dt.year
        _ = self.ppd_series.dt.year
        _ = self.pd_series.dt.month
        _ = self.ppd_series.dt.month

        # Original should be unchanged
        pd.testing.assert_series_equal(original_pd, self.pd_series)
        pd.testing.assert_series_equal(
            original_ppd.to_pandas(), self.ppd_series.to_pandas()
        )
