"""
Datetime utilities for PolarPandas.

Provides pandas-style helpers implemented with Polars primitives.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import polars as pl

from .frame import DataFrame
from .series import Series

_FREQ_PATTERN = re.compile(r"^\s*(?P<count>-?\d+)?\s*(?P<unit>[A-Za-z]+)\s*$")
_FREQ_UNIT_MAP = {
    "D": "d",
    "DAY": "d",
    "DAYS": "d",
    "H": "h",
    "HOUR": "h",
    "HOURS": "h",
    "T": "m",
    "MIN": "m",
    "MINS": "m",
    "MINUTE": "m",
    "MINUTES": "m",
    "S": "s",
    "SEC": "s",
    "SECS": "s",
    "SECOND": "s",
    "SECONDS": "s",
    "L": "ms",
    "MS": "ms",
    "MILLISECOND": "ms",
    "MILLISECONDS": "ms",
    "U": "us",
    "US": "us",
    "MICROSECOND": "us",
    "MICROSECONDS": "us",
    "N": "ns",
    "NS": "ns",
    "NANOSECOND": "ns",
    "NANOSECONDS": "ns",
    "W": "w",
    "B": "d",
    "BUS": "d",
    "WEEK": "w",
    "WEEKS": "w",
}

_DELTA_STRING_MAP = {
    "D": "days",
    "DAY": "days",
    "DAYS": "days",
    "H": "hours",
    "HOUR": "hours",
    "HOURS": "hours",
    "T": "minutes",
    "MIN": "minutes",
    "MINS": "minutes",
    "MINUTE": "minutes",
    "MINUTES": "minutes",
    "S": "seconds",
    "SEC": "seconds",
    "SECS": "seconds",
    "SECOND": "seconds",
    "SECONDS": "seconds",
    "MS": "milliseconds",
    "MILLISECOND": "milliseconds",
    "MILLISECONDS": "milliseconds",
    "US": "microseconds",
    "MICROSECOND": "microseconds",
    "MICROSECONDS": "microseconds",
}

_POLARS_TO_DELTA = {
    "d": "days",
    "h": "hours",
    "m": "minutes",
    "s": "seconds",
    "ms": "milliseconds",
    "us": "microseconds",
    "w": "weeks",
}

_UNIT_TO_NS = {
    "ns": 1,
    "nanosecond": 1,
    "nanoseconds": 1,
    "us": 1_000,
    "µs": 1_000,
    "microsecond": 1_000,
    "microseconds": 1_000,
    "ms": 1_000_000,
    "millisecond": 1_000_000,
    "milliseconds": 1_000_000,
    "s": 1_000_000_000,
    "sec": 1_000_000_000,
    "second": 1_000_000_000,
    "seconds": 1_000_000_000,
}


def _parse_frequency(freq: str) -> tuple[int, str]:
    match = _FREQ_PATTERN.match(freq or "D")
    if not match:
        raise ValueError(f"Unsupported frequency string: {freq!r}")
    count_str = match.group("count")
    unit_key = match.group("unit").upper()
    if unit_key not in _FREQ_UNIT_MAP:
        raise ValueError(f"Unsupported frequency unit: {freq!r}")
    count = int(count_str) if count_str else 1
    if count == 0:
        raise ValueError("Frequency step must be non-zero.")
    return count, _FREQ_UNIT_MAP[unit_key]


def _interval_string(freq: str) -> str:
    count, unit = _parse_frequency(freq)
    prefix = "-" if count < 0 else ""
    magnitude = abs(count)
    return f"{prefix}{magnitude}{unit}"


def _parse_timedelta_string(value: str) -> timedelta:
    match = _FREQ_PATTERN.match(value)
    if not match:
        raise ValueError(f"Unsupported timedelta string: {value!r}")
    count = int(match.group("count") or 1)
    unit_key = match.group("unit").upper()
    if unit_key not in _DELTA_STRING_MAP:
        raise ValueError(f"Unsupported timedelta unit: {value!r}")
    unit_name = _DELTA_STRING_MAP[unit_key]
    return timedelta(**{unit_name: count})


def _ensure_datetime(value: Union[str, datetime]) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"Unsupported datetime type: {type(value)!r}")


def _to_polars_datetime_series(
    series: pl.Series,
    *,
    fmt: Optional[str],
    unit: Optional[str],
    strict: bool,
    exact: bool,
    utc: bool,
    time_zone: Optional[str],
    errors: str,
) -> pl.Series:
    time_unit = unit if unit in {"ns", "us", "µs", "ms"} else "ns"
    target_tz = "UTC" if utc else time_zone

    dtype = series.dtype

    if hasattr(dtype, "is_temporal") and dtype.is_temporal():
        result = series.dt.cast_time_unit(time_unit)
        if target_tz:
            if series.dtype.time_zone is None:
                result = result.dt.replace_time_zone(target_tz)
            else:
                result = result.dt.convert_time_zone(target_tz)
        elif series.dtype.time_zone:
            result = result.dt.replace_time_zone(None)
        return result

    if series.dtype == pl.Utf8:
        return series.str.to_datetime(
            format=fmt,
            strict=strict,
            exact=exact,
            time_unit=time_unit,
            time_zone=target_tz,
        )

    if (hasattr(dtype, "is_integer") and dtype.is_integer()) or (
        hasattr(dtype, "is_float") and dtype.is_float()
    ):
        base_unit = unit or "ns"
        multiplier = _UNIT_TO_NS.get(base_unit.lower())
        if multiplier is None:
            raise ValueError(f"Unsupported numeric unit {unit!r} for to_datetime")
        if dtype.is_float():
            int_series = series.cast(pl.Float64).round(0).cast(pl.Int64)
        else:
            int_series = series.cast(pl.Int64)
        ns_series = int_series * multiplier
        result = ns_series.cast(pl.Datetime(time_unit="ns"))
        if target_tz:
            result = result.dt.replace_time_zone(target_tz)
        return result

    if errors == "coerce":
        return pl.Series(series.name, [None] * len(series)).cast(
            pl.Datetime(time_unit=time_unit, time_zone=target_tz)
        )
    raise TypeError(f"Cannot convert dtype {series.dtype} to datetime")


def date_range(
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
    periods: Optional[int] = None,
    freq: str = "D",
    **kwargs: Any,
) -> Series:
    """
    Create a date range using Polars ``date_range``.
    """
    if start is None and end is None:
        raise ValueError("Must specify either 'start' and 'end', or 'periods'.")

    if periods is None and start is not None and end is None:
        raise ValueError("Must specify either 'start' and 'end', or 'periods'.")
    if periods is None and end is not None and start is None:
        raise ValueError("Must specify either 'start' and 'end', or 'periods'.")
    if periods is not None and start is None and end is None:
        raise ValueError("Must specify either 'start' and 'end', or 'periods'.")

    interval = _interval_string(freq)
    closed = kwargs.pop("closed", kwargs.pop("inclusive", "both"))
    time_zone = kwargs.pop("tz", kwargs.pop("time_zone", None))
    name = kwargs.pop("name", "date_range")
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported keyword arguments: {unsupported}")

    step_delta = timedelta(0)
    if interval is not None:
        count, unit = _parse_frequency(freq)
        if unit not in _POLARS_TO_DELTA:
            raise ValueError(f"Unsupported frequency {freq!r}")
        step_delta = timedelta(**{_POLARS_TO_DELTA[unit]: count})

    if periods is not None and end is None:
        start_dt = _ensure_datetime(start)
        values = [start_dt + step_delta * i for i in range(periods)]
        series = pl.Series(name, values)
        return Series(series)

    if end is None and periods is None:
        raise ValueError("Must specify either 'start' and 'end', or 'periods'.")

    call_kwargs: Dict[str, Any] = {
        "interval": interval,
        "closed": closed,
        "eager": True,
    }
    if time_zone is not None:
        call_kwargs["tz"] = time_zone

    series = pl.date_range(
        start=start,
        end=end,
        **call_kwargs,
    )
    if periods is not None:
        series = series[:periods]
    return Series(series.rename(name))


def to_datetime(
    arg: Any,
    *,
    format: Optional[str] = None,
    unit: Optional[str] = None,
    errors: str = "raise",
    exact: bool = True,
    utc: bool = False,
    dayfirst: bool = False,  # parity placeholder
    **kwargs: Any,
) -> Any:
    """
    Convert the input argument to datetime using Polars parsing.
    """
    if errors not in {"raise", "coerce", "ignore"}:
        raise ValueError("errors must be one of {'raise', 'coerce', 'ignore'}")
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported keyword arguments: {unsupported}")

    if errors == "ignore":
        return arg

    strict = errors == "raise"
    index: Optional[Sequence[Any]] = None

    if isinstance(arg, DataFrame):
        pl_df = arg._df.clone()
        converted_cols = {}
        for column in pl_df.columns:
            converted_cols[column] = _to_polars_datetime_series(
                pl_df[column],
                fmt=format,
                unit=unit,
                strict=strict,
                exact=exact,
                utc=utc,
                time_zone=None,
                errors=errors,
            )
        new_df = pl.DataFrame(converted_cols)
        return DataFrame(new_df, index=arg._index, index_name=arg._index_name)

    scalar_input = False
    name = "datetime"

    if isinstance(arg, Series):
        pl_series = arg._series.clone()
        index = arg._index
    elif isinstance(arg, pl.Series):
        pl_series = arg.clone()
    elif isinstance(arg, (str, bytes)):
        scalar_input = True
        pl_series = pl.Series(name, [arg])
    elif isinstance(arg, (int, float)):
        scalar_input = True
        if unit is None:
            raise ValueError(
                "Unsupported type for to_datetime when 'unit' is not provided."
            )
        pl_series = pl.Series(name, [arg])
    elif isinstance(arg, Iterable):
        pl_series = pl.Series(name, list(arg))
    else:
        scalar_input = True
        pl_series = pl.Series(name, [arg])

    converted = _to_polars_datetime_series(
        pl_series,
        fmt=format,
        unit=unit,
        strict=strict,
        exact=exact,
        utc=utc,
        time_zone=None,
        errors=errors,
    )

    if scalar_input:
        return converted[0]
    return Series(converted, index=index)


def bdate_range(
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
    periods: Optional[int] = None,
    freq: str = "B",
    **kwargs: Any,
) -> Series:
    """
    Business day range that excludes weekends.
    """
    name = kwargs.pop("name", "bdate_range")
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported keyword arguments: {unsupported}")

    if periods is not None:
        if start is None:
            raise ValueError("start must be provided when using periods in bdate_range.")
        current = _ensure_datetime(start)
        values: list[datetime] = []
        day_step = timedelta(days=1)
        direction = 1 if day_step >= timedelta(0) else -1
        while len(values) < periods:
            if current.weekday() < 5:
                values.append(current)
            current = current + direction * day_step
        return Series(pl.Series(name, values))

    full_range = date_range(start=start, end=end, freq="D")
    filtered = [
        value for value in full_range._series.to_list() if value.weekday() < 5
    ]
    return Series(pl.Series(name, filtered))


def timedelta_range(
    start: Optional[str] = None,
    end: Optional[str] = None,
    periods: Optional[int] = None,
    freq: str = "D",
    **kwargs: Any,
) -> Series:
    """
    Generate timedelta values at a fixed frequency.
    """
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported keyword arguments: {unsupported}")

    step = timedelta(0)
    count, unit = _parse_frequency(freq)
    if unit not in _POLARS_TO_DELTA:
        raise ValueError(f"Unsupported timedelta frequency: {freq!r}")
    unit_name = _POLARS_TO_DELTA[unit]
    step = timedelta(**{unit_name: count})

    values: list[timedelta] = []
    if periods is not None:
        base = timedelta(0) if start is None else _parse_timedelta_string(start)
        for i in range(periods):
            values.append(base + step * i)
    else:
        if start is None or end is None:
            raise ValueError("Must specify either (start and end) or (start and periods)")
        current = _parse_timedelta_string(start)
        target = _parse_timedelta_string(end)
        cmp = (lambda a, b: a <= b) if step >= timedelta(0) else (lambda a, b: a >= b)
        while cmp(current, target):
            values.append(current)
            current += step

    return Series(values)


def period_range(
    start: Optional[str] = None,
    end: Optional[str] = None,
    periods: Optional[int] = None,
    freq: str = "D",
    **kwargs: Any,
) -> Series:
    """
    Placeholder period range implementation.
    """
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported keyword arguments: {unsupported}")
    if periods is not None:
        base = start or "P0"
        return Series([f"{base}+{i}{freq}" for i in range(periods)])
    if start is not None and end is not None:
        return Series([start, end])
        raise ValueError("Must specify either (start and end) or (start and periods)")


def interval_range(
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    periods: Optional[int] = None,
    freq: Optional[str] = None,
    **kwargs: Any,
) -> Series:
    """
    Simplified interval range generation.
    """
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported keyword arguments: {unsupported}")
    if periods is not None and start is not None and end is not None:
        step = (end - start) / periods
        intervals = [
            f"[{start + i * step}, {start + (i + 1) * step})" for i in range(periods)
        ]
        return Series(intervals)
    if start is not None and end is not None:
        return Series([f"[{start}, {end})"])
    raise ValueError("Must specify either (start and end) or (start and periods)")


def to_timedelta(arg: Any, unit: str = "ns", **kwargs: Any) -> Any:
    """
    Convert argument to timedeltas.
    """
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported keyword arguments: {unsupported}")

    if isinstance(arg, Series):
        data = arg._series
        index = arg._index
    elif isinstance(arg, pl.Series):
        data = arg
        index = None
    elif isinstance(arg, Iterable) and not isinstance(arg, (str, bytes)):
        data = pl.Series("timedelta", list(arg))
        index = None
    elif isinstance(arg, (int, float, str)):
        data = pl.Series("timedelta", [arg])
        index = None
    else:
        raise TypeError(f"Unsupported type for to_timedelta: {type(arg)!r}")

    if data.dtype == pl.Utf8:
        parsed = [
            _parse_timedelta_string(val) if val is not None else None for val in data
        ]
        result = pl.Series(data.name, parsed)
    else:
        multiplier = _UNIT_TO_NS.get(unit.lower())
        if multiplier is None:
            raise ValueError(f"Unsupported unit for to_timedelta: {unit!r}")
        ns_series = data.cast(pl.Int64) * multiplier
        result = ns_series.cast(pl.Duration(time_unit="ns"))

    if isinstance(arg, (int, float, str)):
        return result[0]
    return Series(result, index=index)

