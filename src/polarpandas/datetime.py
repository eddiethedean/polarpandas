"""
Datetime utilities for PolarPandas implemented without a pandas dependency.

This module covers a small, well-defined subset of the pandas datetime helpers
that PolarPandas exposes: generating date/business-date/timedelta/period/
interval ranges as well as `to_datetime`/`to_timedelta` conversions.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import polars as pl
from dateutil import parser as du_parser
from dateutil import rrule
from dateutil.relativedelta import relativedelta
from dateutil.tz import gettz

from .frame import DataFrame
from .series import Series

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _validate_range_params(feature: str, start: Any, end: Any, periods: Optional[int]) -> None:
    provided = sum(param is not None for param in (start, end, periods))
    if provided < 2:
        raise ValueError("Must specify either (start and end) or (start and periods)")


def _coerce_datetime(value: Any, tz: Optional[str] = None) -> Optional[datetime]:
    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    else:
        dt = du_parser.isoparse(str(value))

    if tz is not None:
        tzinfo = gettz(tz)
        if tzinfo is None:
            raise ValueError(f"Unknown timezone '{tz}'.")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tzinfo)
        else:
            dt = dt.astimezone(tzinfo)

    return dt


def _split_freq(raw_freq: str) -> Tuple[int, str]:
    freq = raw_freq.strip()
    if not freq:
        return 1, "D"

    matched = re.match(r"^([+-]?\d+)?([A-Za-z]+)$", freq)
    if matched is None:
        raise ValueError(f"Unsupported frequency string '{raw_freq}'.")

    multiple_str, code = matched.groups()
    multiple = int(multiple_str) if multiple_str is not None else 1
    normalized = code.lower()

    if normalized in {"min", "minute", "minutes"}:
        normalized = "min"
    elif normalized == "t":
        normalized = "min"
    elif normalized in {"sec", "secs"}:
        normalized = "s"

    return multiple, normalized


def _freq_to_delta(freq_code: str, interval: int) -> Union[timedelta, relativedelta]:
    fc = freq_code.lower()
    if fc == "d":
        return timedelta(days=interval)
    if fc == "b":
        return timedelta(days=interval)
    if fc == "h":
        return timedelta(hours=interval)
    if fc in {"min"}:
        return timedelta(minutes=interval)
    if fc == "s":
        return timedelta(seconds=interval)
    if fc in {"ms", "l"}:
        return timedelta(milliseconds=interval)
    if fc in {"us", "µs", "u"}:
        return timedelta(microseconds=interval)
    if fc in {"ns", "n"}:
        raise ValueError("Nanosecond frequency is not supported without pandas.")
    if fc in {"w"}:
        return timedelta(weeks=interval)
    if fc in {"m", "ms"}:  # month / month start
        return relativedelta(months=interval)
    if fc in {"q", "qs"}:
        return relativedelta(months=3 * interval)
    if fc in {"y", "a", "ys", "as"}:
        return relativedelta(years=interval)

    raise ValueError(f"Unsupported frequency code '{freq_code}'.")


def _scale_delta(delta: Union[timedelta, relativedelta], factor: int) -> Union[timedelta, relativedelta]:
    if isinstance(delta, relativedelta):
        return relativedelta(
            years=delta.years * factor,
            months=delta.months * factor,
            days=delta.days * factor,
            hours=delta.hours * factor,
            minutes=delta.minutes * factor,
            seconds=delta.seconds * factor,
            microseconds=delta.microseconds * factor,
        )
    return delta * factor


def _apply_delta(dt: datetime, delta: Union[timedelta, relativedelta], factor: int = 1) -> datetime:
    scaled = _scale_delta(delta, factor)
    return dt + scaled


def _ensure_period_arguments(feature: str, start: Optional[datetime], end: Optional[datetime], periods: Optional[int]) -> Tuple[Optional[datetime], Optional[datetime], Optional[int]]:
    if periods is not None and periods <= 0:
        raise ValueError(f"{feature} requires periods > 0.")
    if start is None and end is None:
        raise ValueError(f"{feature} requires either 'start' or 'end'.")
    return start, end, periods


def _generate_sequence(
    feature: str,
    start: Optional[datetime],
    end: Optional[datetime],
    periods: Optional[int],
    delta: Union[timedelta, relativedelta],
    *,
    business: bool = False,
) -> List[datetime]:
    start, end, periods = _ensure_period_arguments(feature, start, end, periods)

    values: List[datetime] = []

    if start is not None and periods is not None:
        current = start
        count = 0
        while count < periods:
            if not business or current.weekday() < 5:
                values.append(current)
                count += 1
            current = _apply_delta(current, delta)

    elif start is not None and end is not None:
        current = start
        while current <= end:
            if not business or current.weekday() < 5:
                values.append(current)
            current = _apply_delta(current, delta)

    elif end is not None and periods is not None:
        current = end
        count = 0
        while count < periods:
            if not business or current.weekday() < 5:
                values.append(current)
                count += 1
            current = _apply_delta(current, delta, -1)
        values.reverse()

    else:  # pragma: no cover - guarded by validation
        raise ValueError(f"{feature} received unsupported parameter combination.")

    if business:
        values = [dt if dt.weekday() < 5 else _next_business_day(dt) for dt in values]
    return values


def _next_business_day(dt: datetime) -> datetime:
    current = dt
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current


def _business_range(
    feature: str,
    start: Optional[datetime],
    end: Optional[datetime],
    periods: Optional[int],
    delta: timedelta,
) -> List[date]:
    start, end, periods = _ensure_period_arguments(feature, start, end, periods)

    results: List[datetime] = []

    if start is not None:
        current = _next_business_day(start)
        while True:
            if end is not None and current > end:
                break
            if current.weekday() < 5:
                results.append(current)
                if periods is not None and len(results) >= periods:
                    break
            current += delta

    elif end is not None and periods is not None:
        current = end
        while len(results) < periods:
            if current.weekday() < 5:
                results.append(current)
            current -= delta
        results = sorted(_next_business_day(dt) if dt.weekday() >= 5 else dt for dt in results)

    return [dt.date() for dt in results if dt.weekday() < 5]


def _parse_timedelta_string(value: str) -> timedelta:
    text = value.strip()
    if not text:
        raise ValueError("Empty timedelta string.")

    iso_match = re.fullmatch(
        r"(?:(?P<days>[+-]?\d+)\s*(?:days?|d))?"
        r"(?:\s*(?P<hours>[+-]?\d+)\s*(?:hours?|h))?"
        r"(?:\s*(?P<minutes>[+-]?\d+)\s*(?:minutes?|mins?|m))?"
        r"(?:\s*(?P<seconds>[+-]?\d+)\s*(?:seconds?|secs?|s))?"
        r"(?:\s*(?P<microseconds>[+-]?\d+)\s*(?:microseconds?|µs|us))?",
        text,
        flags=re.IGNORECASE,
    )
    if iso_match:
        days = int(iso_match.group("days") or 0)
        hours = int(iso_match.group("hours") or 0)
        minutes = int(iso_match.group("minutes") or 0)
        seconds = int(iso_match.group("seconds") or 0)
        microseconds = int(iso_match.group("microseconds") or 0)
        return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds)

    colon_match = re.fullmatch(r"^([+-]?\d+):(\d+):(\d+)$", text)
    if colon_match:
        hours, minutes, seconds = map(int, colon_match.groups())
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)

    raise ValueError(f"Unsupported timedelta string '{value}'.")


def _coerce_timedelta(value: Any, default_unit: str = "ns") -> Optional[timedelta]:
    if value is None:
        return None
    if isinstance(value, timedelta):
        return value
    if isinstance(value, (int, float)):
        return _numeric_to_timedelta(value, default_unit)
    if isinstance(value, str):
        return _parse_timedelta_string(value)
    raise ValueError(f"Unsupported timedelta value: {value!r}")


def _unit_to_timedelta(unit: str) -> timedelta:
    key = unit.lower()
    if key in {"d", "day", "days"}:
        return timedelta(days=1)
    if key in {"h", "hour", "hours"}:
        return timedelta(hours=1)
    if key in {"m", "min", "minute", "minutes"}:
        return timedelta(minutes=1)
    if key in {"s", "sec", "second", "seconds"}:
        return timedelta(seconds=1)
    if key in {"ms", "millisecond", "milliseconds"}:
        return timedelta(milliseconds=1)
    if key in {"us", "µs", "microsecond", "microseconds"}:
        return timedelta(microseconds=1)
    if key in {"ns", "nanosecond", "nanoseconds"}:
        raise ValueError("Nanosecond duration precision is unsupported without pandas.")
    raise ValueError(f"Unsupported timedelta unit '{unit}'.")


def _numeric_to_timedelta(value: Union[int, float], unit: str) -> timedelta:
    base = _unit_to_timedelta(unit)
    multiplier = value
    if isinstance(base, timedelta):
        return base * multiplier
    raise ValueError(f"Unsupported timedelta conversion for unit '{unit}'.")


def _series_from_list(values: Sequence[Any], name: Optional[str], default: str) -> Series:
    series_name = name or default
    return Series(pl.Series(series_name, list(values)))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def date_range(
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    periods: Optional[int] = None,
    freq: str = "D",
    tz: Optional[str] = None,
    **kwargs: Any,
) -> Series:
    """
    Generate a sequence of datetimes at a fixed frequency.

    Supported frequencies: daily (`D`), hourly (`H`), minutely (`min`/`T`), secondly (`S`),
    weekly (`W`), monthly (`M`), quarterly (`Q`), yearly (`Y`), and business daily (`B`).
    """

    name = kwargs.pop("name", None)
    if kwargs:
        unsupported = ", ".join(sorted(kwargs.keys()))
        raise NotImplementedError(f"Unsupported keyword arguments for date_range: {unsupported}")

    _validate_range_params("date_range()", start, end, periods)

    interval, freq_code = _split_freq(freq)
    start_dt = _coerce_datetime(start, tz)
    end_dt = _coerce_datetime(end, tz)

    delta = _freq_to_delta(freq_code, interval)
    if freq_code == "b":
        dates = _business_range("date_range()", start_dt, end_dt, periods, timedelta(days=interval))
        return _series_from_list(dates, name, "bdate")

    sequence = _generate_sequence("date_range()", start_dt, end_dt, periods, delta)
    return _series_from_list(sequence, name, "date")


def to_datetime(arg: Any, **kwargs: Any) -> DataFrame:
    """
    Convert basic iterables into a PolarPandas DataFrame containing datetimes.
    """

    if isinstance(arg, list):
        series = pl.Series("datetime", arg).str.to_datetime(**kwargs)
        return DataFrame(pl.DataFrame({"datetime": series}))

    if isinstance(arg, DataFrame):
        return arg.copy()

    raise ValueError(f"Unsupported type for to_datetime: {type(arg)}")


def bdate_range(
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    periods: Optional[int] = None,
    freq: str = "B",
    **kwargs: Any,
) -> Series:
    """
    Generate business day ranges (weekdays only).
    """

    name = kwargs.pop("name", None)
    if kwargs:
        unsupported = ", ".join(sorted(kwargs.keys()))
        raise NotImplementedError(f"Unsupported keyword arguments for bdate_range: {unsupported}")

    _validate_range_params("bdate_range()", start, end, periods)
    interval, freq_code = _split_freq(freq or "B")
    if freq_code != "b":
        raise ValueError("bdate_range only supports business-day frequency ('B').")

    start_dt = _coerce_datetime(start)
    end_dt = _coerce_datetime(end)
    values = _business_range("bdate_range()", start_dt, end_dt, periods, timedelta(days=interval))
    return _series_from_list(values, name, "bdate")


def timedelta_range(
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    periods: Optional[int] = None,
    freq: str = "D",
    **kwargs: Any,
) -> Series:
    """
    Generate timedelta values at a fixed frequency.
    """

    name = kwargs.pop("name", None)
    if kwargs:
        unsupported = ", ".join(sorted(kwargs.keys()))
        raise NotImplementedError(f"Unsupported keyword arguments for timedelta_range: {unsupported}")

    _validate_range_params("timedelta_range()", start, end, periods)
    interval, freq_code = _split_freq(freq)

    delta = _freq_to_delta(freq_code, interval)
    if isinstance(delta, relativedelta):
        raise ValueError("timedelta_range does not support month/quarter/year frequencies.")

    start_td = _coerce_timedelta(start, "ns")
    end_td = _coerce_timedelta(end, "ns")

    start_dt = datetime(1970, 1, 1) + start_td if start_td is not None else None
    end_dt = datetime(1970, 1, 1) + end_td if end_td is not None else None

    base = _generate_sequence("timedelta_range()", start_dt, end_dt, periods, delta)
    timedeltas = [dt - datetime(1970, 1, 1) for dt in base]
    return _series_from_list(timedeltas, name, "timedelta")


def period_range(
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    periods: Optional[int] = None,
    freq: str = "D",
    **kwargs: Any,
) -> Series:
    """
    Construct labelled period strings using a datetime range as the backbone.
    """

    name = kwargs.pop("name", None)
    if kwargs:
        unsupported = ", ".join(sorted(kwargs.keys()))
        raise NotImplementedError(f"Unsupported keyword arguments for period_range: {unsupported}")

    _validate_range_params("period_range()", start, end, periods)
    interval, freq_code = _split_freq(freq)

    delta = _freq_to_delta(freq_code, interval)
    start_dt = _coerce_datetime(start)
    end_dt = _coerce_datetime(end)
    sequence = _generate_sequence("period_range()", start_dt, end_dt, periods, delta)

    format_map = {
        "d": "%Y-%m-%d",
        "h": "%Y-%m-%d %H",
        "min": "%Y-%m-%d %H:%M",
        "s": "%Y-%m-%d %H:%M:%S",
        "w": "%Y-%W",
        "m": "%Y-%m",
        "q": lambda dt: f"{dt.year}-Q{((dt.month - 1) // 3) + 1}",
        "y": "%Y",
    }

    formatter = format_map.get(freq_code, "%Y-%m-%d")
    if callable(formatter):
        labels = [formatter(dt) for dt in sequence]
    else:
        labels = [dt.strftime(formatter) for dt in sequence]

    return _series_from_list(labels, name, "period")


def interval_range(
    start: Optional[Union[int, float]] = None,
    end: Optional[Union[int, float]] = None,
    periods: Optional[int] = None,
    freq: Optional[str] = None,
    **kwargs: Any,
) -> Series:
    """
    Construct simple half-open interval labels.
    """

    name = kwargs.pop("name", None)
    if kwargs:
        unsupported = ", ".join(sorted(kwargs.keys()))
        raise NotImplementedError(f"Unsupported keyword arguments for interval_range: {unsupported}")

    _validate_range_params("interval_range()", start, end, periods)

    if freq is not None:
        raise NotImplementedError("interval_range does not yet support explicit frequency strings.")

    if start is not None and end is not None and periods is not None:
        step = (end - start) / periods
        edges = [start + step * i for i in range(periods + 1)]
    elif start is not None and periods is not None:
        step = 1.0
        edges = [start + step * i for i in range(periods + 1)]
    elif end is not None and periods is not None:
        step = 1.0
        edges = [end - step * (periods - i) for i in range(periods + 1)]
    else:  # pragma: no cover - guarded earlier
        raise ValueError("interval_range requires start/end with periods.")

    def _format_edge(val: float) -> str:
        if isinstance(val, (int, float)) and float(val).is_integer():
            return str(int(val))
        return str(val)

    labels = [
        f"[{_format_edge(edges[i])}, {_format_edge(edges[i + 1])})"
        for i in range(len(edges) - 1)
    ]
    return _series_from_list(labels, name, "interval")


def to_timedelta(arg: Any, unit: str = "ns", **kwargs: Any) -> Any:
    """
    Convert scalars or iterables to timedeltas without relying on pandas.
    """

    if kwargs:
        unsupported = ", ".join(sorted(kwargs.keys()))
        raise NotImplementedError(f"Unsupported keyword arguments for to_timedelta: {unsupported}")

    if isinstance(arg, Series):
        values = [_coerce_timedelta(value, unit) if value is not None else None for value in arg.to_list()]
        return _series_from_list(values, arg.name, arg.name or "timedelta")

    if isinstance(arg, (list, tuple)):
        values = [_coerce_timedelta(value, unit) if value is not None else None for value in arg]
        series = pl.Series("timedelta", values)
        return Series(series)

    if isinstance(arg, (int, float)):
        return _numeric_to_timedelta(arg, unit)

    if isinstance(arg, str):
        return _parse_timedelta_string(arg)

    raise ValueError(f"Unsupported type for to_timedelta: {type(arg)}")
