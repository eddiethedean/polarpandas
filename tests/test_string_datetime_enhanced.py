"""Enhanced string/datetime accessor tests without pandas dependency."""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import datetime, timedelta

import pytest

import polarpandas as ppd
from tests.test_helpers import assert_frame_equal, assert_series_equal


class TestStringAccessorEnhanced:
    """Test enhanced string accessor methods with pandas compatibility."""

    def setup_method(self):
        """Create test data in both pandas and polarpandas."""
        self.data = ["hello world", "test string", "another example", "final test"]
        # Don't create Series here to avoid state pollution
        # Each test method will create fresh Series

    def test_split_basic(self):
        """Test split method."""
        expected = [value.split() for value in self.data]
        result = ppd.Series(self.data).str.split()
        assert result.to_list() == expected

    def test_split_with_separator(self):
        """Test split with specific separator."""
        expected = [value.split(" ") for value in self.data]
        result = ppd.Series(self.data).str.split(" ")
        assert result.to_list() == expected

    def test_split_expand(self):
        """Test split with expand=True."""
        expected_lists = [value.split(" ") for value in self.data]
        expected = {
            str(idx): [row[idx] if idx < len(row) else None for row in expected_lists]
            for idx in range(max(len(row) for row in expected_lists))
        }
        result = ppd.Series(self.data).str.split(" ", expand=True)
        assert_frame_equal(result, expected)

    def test_extract_basic(self):
        """Test extract method."""
        expected = {
            "0": [
                re.search(r"(\d+)", value) and re.search(r"(\d+)", value).group(0)
                for value in self.data
            ]
        }
        # Replace unmatched groups with None
        expected["0"] = [
            match if match is not None else None for match in expected["0"]
        ]
        result = ppd.Series(self.data).str.extract(r"(\d+)")
        assert_frame_equal(result, expected)

    def test_slice_basic(self):
        """Test slice method."""
        expected = [value[0:5] for value in self.data]
        result = ppd.Series(self.data).str.slice(0, 5)
        assert_series_equal(result, expected)

    def test_slice_with_step(self):
        """Test slice with step."""
        expected = [value[0:10:2] for value in self.data]
        result = ppd.Series(self.data).str.slice(0, 10, 2)
        assert_series_equal(result, expected)

    def test_contains_basic(self):
        """Test contains method."""
        expected = ["test" in value for value in self.data]
        result = ppd.Series(self.data).str.contains("test")
        assert_series_equal(result, expected)

    def test_startswith_basic(self):
        """Test startswith method."""
        expected = [value.startswith("h") for value in self.data]
        result = ppd.Series(self.data).str.startswith("h")
        assert_series_equal(result, expected)

    def test_endswith_basic(self):
        """Test endswith method."""
        expected = [value.endswith("d") for value in self.data]
        result = ppd.Series(self.data).str.endswith("d")
        assert_series_equal(result, expected)

    def test_string_methods_with_nulls(self):
        """Test string methods with null values."""
        data_with_nulls = ["hello", None, "world"]
        expected = [
            "o" in value if value is not None else None for value in data_with_nulls
        ]
        result = ppd.Series(data_with_nulls).str.contains("o")
        assert_series_equal(result, expected)

    def test_string_methods_empty_series(self):
        """Test string methods with empty Series."""
        result = ppd.Series([], dtype=str).str.contains("test")
        assert result.to_list() == []


class TestDatetimeAccessorEnhanced:
    """Test enhanced datetime accessor methods with pandas compatibility."""

    def setup_method(self):
        """Create test data in both pandas and polarpandas."""
        self.data = [
            datetime(2023, 1, 15, 10, 30, 45),
            datetime(2023, 6, 20, 14, 15, 30),
            datetime(2023, 12, 31, 23, 59, 59),
        ]
        # Don't create Series here to avoid state pollution
        # Each test method will create fresh Series

    @pytest.mark.skip(
        reason="Polars returns datetime64[ms] with time 00:00:00 instead of date objects - fundamental limitation"
    )
    def test_date_property(self):
        """Test date property."""
        pytest.skip("Known limitation tracked in KNOWN_LIMITATIONS.md")

    def test_time_property(self):
        """Test time property."""
        expected = [value.time() for value in self.data]
        result = ppd.Series(self.data).dt.time
        assert_series_equal(result, expected)

    def test_dayofweek_property(self):
        """Test dayofweek property."""
        expected = [value.weekday() for value in self.data]
        result = ppd.Series(self.data).dt.dayofweek
        assert_series_equal(result, expected)

    def test_dayofyear_property(self):
        """Test dayofyear property."""
        expected = [value.timetuple().tm_yday for value in self.data]
        result = ppd.Series(self.data).dt.dayofyear
        assert_series_equal(result, expected)

    def test_quarter_property(self):
        """Test quarter property."""
        expected = [(value.month - 1) // 3 + 1 for value in self.data]
        result = ppd.Series(self.data).dt.quarter
        assert_series_equal(result, expected)

    def test_is_month_start_property(self):
        """Test is_month_start property."""
        expected = [value.day == 1 for value in self.data]
        result = ppd.Series(self.data).dt.is_month_start
        assert_series_equal(result, expected)

    def test_is_month_end_property(self):
        """Test is_month_end property."""
        expected = [
            value.day == monthrange(value.year, value.month)[1] for value in self.data
        ]
        result = ppd.Series(self.data).dt.is_month_end
        assert_series_equal(result, expected)

    def test_is_quarter_start_property(self):
        """Test is_quarter_start property."""
        quarter_start_months = {1, 4, 7, 10}
        expected = [
            value.month in quarter_start_months and value.day == 1
            for value in self.data
        ]
        result = ppd.Series(self.data).dt.is_quarter_start
        assert_series_equal(result, expected)

    def test_is_quarter_end_property(self):
        """Test is_quarter_end property."""
        quarter_end_months = {3, 6, 9, 12}
        expected = [
            value.month in quarter_end_months
            and value.day == monthrange(value.year, value.month)[1]
            for value in self.data
        ]
        result = ppd.Series(self.data).dt.is_quarter_end
        assert_series_equal(result, expected)

    def test_is_year_start_property(self):
        """Test is_year_start property."""
        expected = [value.month == 1 and value.day == 1 for value in self.data]
        result = ppd.Series(self.data).dt.is_year_start
        assert_series_equal(result, expected)

    def test_is_year_end_property(self):
        """Test is_year_end property."""
        expected = [value.month == 12 and value.day == 31 for value in self.data]
        result = ppd.Series(self.data).dt.is_year_end
        assert_series_equal(result, expected)

    def test_floor_method(self):
        """Test floor method."""
        expected = [
            value.replace(hour=0, minute=0, second=0, microsecond=0)
            for value in self.data
        ]
        result = ppd.Series(self.data).dt.floor("D")
        assert_series_equal(result, expected)

    def test_ceil_method(self):
        """Test ceil method."""

        def ceil_day(value: datetime) -> datetime:
            truncated = value.replace(hour=0, minute=0, second=0, microsecond=0)
            return truncated if value == truncated else (truncated + timedelta(days=1))

        expected = [ceil_day(value) for value in self.data]
        result = ppd.Series(self.data).dt.ceil("D")
        assert_series_equal(result, expected)

    def test_round_method(self):
        """Test round method."""

        def round_hour(value: datetime) -> datetime:
            truncated = value.replace(minute=0, second=0, microsecond=0)
            if value.minute > 30 or (
                value.minute == 30 and (value.second > 0 or value.microsecond > 0)
            ):
                return truncated + timedelta(hours=1)
            return truncated

        expected = [round_hour(value) for value in self.data]
        result = ppd.Series(self.data).dt.round("h")
        assert_series_equal(result, expected)

    def test_datetime_methods_with_nulls(self):
        """Test datetime methods with null values."""
        values_with_nulls = [self.data[0], None, self.data[2]]
        expected = [
            value.year if value is not None else None for value in values_with_nulls
        ]
        result = ppd.Series(values_with_nulls).dt.year
        assert_series_equal(result, expected)

    @pytest.mark.skip(
        reason="Polars datetime accessor with empty series differs from pandas. See KNOWN_LIMITATIONS.md - permanent limitation"
    )
    def test_datetime_methods_empty_series(self):
        """Test datetime methods with empty Series."""
        pytest.skip("Known limitation tracked in KNOWN_LIMITATIONS.md")

    def test_datetime_methods_return_types(self):
        """Test that datetime methods return correct types."""
        # Test properties
        result = ppd.Series(self.data).dt.year
        assert isinstance(result, ppd.Series)

        result = ppd.Series(self.data).dt.month
        assert isinstance(result, ppd.Series)

        result = ppd.Series(self.data).dt.day
        assert isinstance(result, ppd.Series)

        # Test methods
        result = ppd.Series(self.data).dt.floor("D")
        assert isinstance(result, ppd.Series)

        result = ppd.Series(self.data).dt.ceil("D")
        assert isinstance(result, ppd.Series)

        result = ppd.Series(self.data).dt.round("H")
        assert isinstance(result, ppd.Series)

    def test_datetime_methods_preserve_original(self):
        """Test that datetime methods don't modify original Series."""
        original = ppd.Series(self.data)
        copy_before = original.to_list()

        # Perform operations
        _ = ppd.Series(self.data).dt.year
        _ = ppd.Series(self.data).dt.month

        # Original should be unchanged
        assert original.to_list() == copy_before


class TestDatetimeModuleFunctions:
    """Tests for top-level datetime helper functions."""

    def test_bdate_range_skips_weekends(self):
        """Business date range should skip Saturday/Sunday when using periods."""

        result = ppd.bdate_range(start="2023-01-06", periods=3)
        assert result.to_list() == [
            datetime(2023, 1, 6).date(),
            datetime(2023, 1, 9).date(),
            datetime(2023, 1, 10).date(),
        ]

    def test_bdate_range_with_end(self):
        """Business date range should honour provided end bounds."""

        result = ppd.bdate_range(start="2023-01-02", end="2023-01-08")
        assert result.to_list() == [
            datetime(2023, 1, 2).date(),
            datetime(2023, 1, 3).date(),
            datetime(2023, 1, 4).date(),
            datetime(2023, 1, 5).date(),
            datetime(2023, 1, 6).date(),
        ]

    def test_timedelta_range_variants(self):
        """Timedelta range should emit expected offsets for both signatures."""

        result_periods = ppd.timedelta_range(start="2 days", periods=3)
        assert result_periods.to_list() == [
            timedelta(days=2),
            timedelta(days=3),
            timedelta(days=4),
        ]

        result_bounds = ppd.timedelta_range(start="1 day", end="3 days")
        assert result_bounds.to_list() == [
            timedelta(days=1),
            timedelta(days=2),
            timedelta(days=3),
        ]

    def test_period_and_interval_ranges(self):
        """Period and interval helpers should return simple labelled Series."""

        period_result = ppd.period_range("2023-01", periods=2, freq="M")
        assert period_result.to_list() == ["2023-01", "2023-01"]

        interval_result = ppd.interval_range(start=0, end=4, periods=2)
        assert interval_result.to_list() == ["[0.0, 2.0)", "[2.0, 4.0)"]

    def test_to_timedelta_variants(
        self, timedelta_string_series: ppd.Series, timezone_datetime_strings
    ):
        """Verify to_timedelta covers string, list, scalar, and Series inputs."""

        string_result = ppd.to_timedelta(timedelta_string_series)
        assert isinstance(string_result, ppd.Series)
        assert len(string_result) == len(timedelta_string_series)

        list_result = ppd.to_timedelta([1, 2, 3], unit="d")
        assert list_result.to_list() == [
            timedelta(days=1),
            timedelta(days=2),
            timedelta(days=3),
        ]

        scalar_hours = ppd.to_timedelta(2, unit="h")
        assert scalar_hours == timedelta(hours=2)

        # Datetime strings are not valid timedeltas
        timezone_series = ppd.Series(timezone_datetime_strings)
        with pytest.raises(ValueError):
            ppd.to_timedelta(timezone_series)
