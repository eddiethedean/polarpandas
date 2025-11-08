from datetime import datetime, timedelta

import polars as pl

import polarpandas as ppd


def test_date_range_basic():
    result = ppd.date_range(start="2021-01-01", periods=3)
    values = result._series.to_list()
    assert values == [
        datetime(2021, 1, 1),
        datetime(2021, 1, 2),
        datetime(2021, 1, 3),
    ]


def test_date_range_with_freq_hours():
    result = ppd.date_range(start="2021-01-01 00:00", periods=3, freq="2H")
    values = result._series.to_list()
    assert values[1] - values[0] == timedelta(hours=2)


def test_bdate_range_excludes_weekends():
    result = ppd.bdate_range(start="2021-01-01", periods=5)
    weekdays = [value.weekday() for value in result._series.to_list()]
    assert all(day < 5 for day in weekdays)


def test_to_datetime_strings():
    series = ppd.to_datetime(["2024-01-01", "2024-01-02"])
    assert isinstance(series, ppd.Series)
    assert series._series.dtype == pl.Datetime("ns")
    assert series._series.to_list()[0] == datetime(2024, 1, 1)


def test_to_datetime_numeric_seconds():
    series = ppd.to_datetime([0, 1, 2], unit="s")
    expected = [
        datetime(1970, 1, 1),
        datetime(1970, 1, 1, 0, 0, 1),
        datetime(1970, 1, 1, 0, 0, 2),
    ]
    assert series._series.to_list() == expected


def test_to_datetime_coerce_errors():
    series = ppd.to_datetime(["2024-01-01", "invalid"], errors="coerce")
    assert series._series.to_list()[1] is None


def test_to_datetime_scalar():
    value = ppd.to_datetime("2024-02-01")
    assert value == datetime(2024, 2, 1)


def test_to_timedelta_strings():
    series = ppd.to_timedelta(["1 day", "2 days"])
    assert series._series.dtype == pl.Duration("us")
    assert series._series.to_list()[0] == timedelta(days=1)


def test_to_timedelta_numeric_seconds():
    series = ppd.to_timedelta([1, 2], unit="s")
    assert series._series.dtype == pl.Duration("ns")
    assert series._series.to_list()[0] == timedelta(seconds=1)

