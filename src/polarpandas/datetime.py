"""
Datetime utilities for PolarPandas.

This module provides pandas-compatible functions for working with datetime
data, including date range generation and datetime conversion.

Functions
---------
date_range : Generate sequences of dates
to_datetime : Convert argument to datetime

Examples
--------
>>> import polarpandas as ppd
>>> # Generate date range
>>> dates = ppd.date_range("2021-01-01", periods=5)
>>> # Convert to datetime
>>> dt = ppd.to_datetime(["2021-01-01", "2021-01-02"])

Notes
-----
- Datetime operations may have limitations compared to pandas
- Some datetime formats may not be fully supported
"""

from typing import Any, Optional

from .frame import DataFrame
from .series import Series


def date_range(
    start: Optional[str] = None,
    end: Optional[str] = None,
    periods: Optional[int] = None,
    freq: str = "D",
    **kwargs: Any,
) -> Series:
    """
    Create a date range.

    Parameters
    ----------
    start : str, optional
        Start date
    end : str, optional
        End date
    periods : int, optional
        Number of periods
    freq : str, default "D"
        Frequency string
    **kwargs
        Additional arguments

    Returns
    -------
    Series
        Series with date range

    Examples
    --------
    >>> import polarpandas as ppd
    >>> dates = ppd.date_range("2021-01-01", periods=5)
    """

    if start and periods:
        # Create range from start with specified periods
        # This is a simplified implementation
        dates = [start] * periods
        return Series(dates)
    elif start and end:
        # Create range between start and end
        dates = [start, end]
        return Series(dates)
    else:
        raise ValueError("Must specify either (start and end) or (start and periods)")


def to_datetime(arg: Any, **kwargs: Any) -> DataFrame:
    """
    Convert argument to datetime.

    Parameters
    ----------
    arg : Any
        Input to convert to datetime
    **kwargs
        Additional arguments

    Returns
    -------
    DataFrame
        DataFrame with datetime values

    Examples
    --------
    >>> import polarpandas as ppd
    >>> dates = ppd.to_datetime(["2021-01-01", "2021-01-02"])
    """
    if isinstance(arg, list):
        import polars as pl

        # Convert list to DataFrame with datetime column
        datetime_series = pl.Series("datetime", arg).str.to_datetime()
        return DataFrame(pl.DataFrame({"datetime": datetime_series}))
    elif isinstance(arg, DataFrame):
        # Convert DataFrame columns to datetime
        return arg.copy()  # Simplified implementation
    else:
        raise ValueError(f"Unsupported type for to_datetime: {type(arg)}")


def bdate_range(
    start: Optional[str] = None,
    end: Optional[str] = None,
    periods: Optional[int] = None,
    freq: str = "B",
    **kwargs: Any,
) -> Series:
    """
    Return a fixed frequency DatetimeIndex with business day as the default.

    Parameters
    ----------
    start : str, optional
        Start date
    end : str, optional
        End date
    periods : int, optional
        Number of periods
    freq : str, default "B"
        Frequency string (B = business day)
    **kwargs
        Additional arguments

    Returns
    -------
    Series
        Series with business date range

    Examples
    --------
    >>> import polarpandas as ppd
    >>> dates = ppd.bdate_range("2021-01-01", periods=5)
    """
    # Simplified implementation - business days exclude weekends
    # For a full implementation, would need to handle holidays
    from datetime import datetime, timedelta

    import polars as pl

    from .series import Series

    if start and periods:
        # Generate business days from start
        dates = []
        current = (
            datetime.fromisoformat(str(start)) if isinstance(start, str) else start
        )
        count = 0
        while count < periods:
            # Check if weekday (0=Monday, 6=Sunday)
            if current.weekday() < 5:  # Monday-Friday
                dates.append(current.date())
                count += 1
            current = current + timedelta(days=1)
        return Series(pl.Series(dates))
    elif start and end:
        # Generate business days between start and end
        dates = []
        current = (
            datetime.fromisoformat(str(start)) if isinstance(start, str) else start
        )
        end_date = datetime.fromisoformat(str(end)) if isinstance(end, str) else end
        while current <= end_date:
            if current.weekday() < 5:  # Monday-Friday
                dates.append(current.date())
            current = current + timedelta(days=1)
        return Series(pl.Series(dates))
    else:
        raise ValueError("Must specify either (start and end) or (start and periods)")


def timedelta_range(
    start: Optional[str] = None,
    end: Optional[str] = None,
    periods: Optional[int] = None,
    freq: str = "D",
    **kwargs: Any,
) -> Series:
    """
    Return a fixed frequency TimedeltaIndex.

    Parameters
    ----------
    start : str, optional
        Start timedelta
    end : str, optional
        End timedelta
    periods : int, optional
        Number of periods
    freq : str, default "D"
        Frequency string
    **kwargs
        Additional arguments

    Returns
    -------
    Series
        Series with timedelta range

    Examples
    --------
    >>> import polarpandas as ppd
    >>> deltas = ppd.timedelta_range(start="1 day", periods=5)
    """
    import polars as pl

    from .series import Series

    if start and periods:
        # Parse start timedelta and generate range
        # Simplified: assume format like "1 day", "2 days", etc.
        base_delta = pl.duration(days=1)  # Default
        if "day" in start.lower():
            try:
                days = int(start.split()[0])
                base_delta = pl.duration(days=days)
            except (ValueError, IndexError):
                pass

        deltas = [base_delta * i for i in range(periods)]
        return Series(deltas)
    elif start and end:
        # Generate range between start and end
        deltas = [pl.duration(days=1), pl.duration(days=2)]  # Simplified
        return Series(deltas)
    else:
        raise ValueError("Must specify either (start and end) or (start and periods)")


def period_range(
    start: Optional[str] = None,
    end: Optional[str] = None,
    periods: Optional[int] = None,
    freq: str = "D",
    **kwargs: Any,
) -> Series:
    """
    Return a fixed frequency PeriodIndex.

    Parameters
    ----------
    start : str, optional
        Start period
    end : str, optional
        End period
    periods : int, optional
        Number of periods
    freq : str, default "D"
        Frequency string
    **kwargs
        Additional arguments

    Returns
    -------
    Series
        Series with period range

    Examples
    --------
    >>> import polarpandas as ppd
    >>> periods = ppd.period_range("2021-01", periods=5, freq="M")
    """
    # Simplified implementation - Polars doesn't have native Period type
    # This would need more sophisticated handling
    from .series import Series

    if start and periods:
        dates = [start] * periods
        return Series(dates)
    elif start and end:
        dates = [start, end]
        return Series(dates)
    else:
        raise ValueError("Must specify either (start and end) or (start and periods)")


def interval_range(
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    periods: Optional[int] = None,
    freq: Optional[str] = None,
    **kwargs: Any,
) -> Series:
    """
    Return a fixed frequency IntervalIndex.

    Parameters
    ----------
    start : Any, optional
        Start value
    end : Any, optional
        End value
    periods : int, optional
        Number of periods
    freq : str, optional
        Frequency string
    **kwargs
        Additional arguments

    Returns
    -------
    Series
        Series with interval range

    Examples
    --------
    >>> import polarpandas as ppd
    >>> intervals = ppd.interval_range(start=0, end=5, periods=3)
    """
    # Simplified implementation - Polars doesn't have native Interval type
    from .series import Series

    if start is not None and periods:
        # Generate intervals
        step = (end - start) / periods if end is not None else 1
        intervals = [
            f"[{start + i * step}, {start + (i + 1) * step})" for i in range(periods)
        ]
        return Series(intervals)
    elif start is not None and end is not None:
        intervals = [f"[{start}, {end})"]
        return Series(intervals)
    else:
        raise ValueError("Must specify either (start and end) or (start and periods)")


def to_timedelta(arg: Any, unit: str = "ns", **kwargs: Any) -> Any:
    """
    Convert argument to timedelta.

    Parameters
    ----------
    arg : str, int, float, Series, or array-like
        Input to convert to timedelta
    unit : str, default "ns"
        Unit of the timedelta
    **kwargs
        Additional arguments

    Returns
    -------
    Series or scalar
        Timedelta data

    Examples
    --------
    >>> import polarpandas as ppd
    >>> deltas = ppd.to_timedelta(["1 day", "2 days"])
    """
    import polars as pl

    from .series import Series

    if isinstance(arg, Series):
        # Convert Series to timedelta
        if arg._series.dtype == pl.Utf8:
            # Parse string timedeltas
            result = arg._series.str.replace(" day", "d").str.replace(" days", "d")
            return Series(result.str.to_datetime())
        else:
            # Convert numeric to timedelta
            return Series(arg._series.cast(pl.Duration))
    elif isinstance(arg, (list, tuple)):
        # Convert list to Series and then to timedelta
        pl_series = pl.Series(arg)
        if pl_series.dtype == pl.Utf8:
            result = pl_series.str.replace(" day", "d").str.replace(" days", "d")
            return Series(result.str.to_datetime())
        else:
            return Series(pl_series.cast(pl.Duration))
    elif isinstance(arg, (int, float)):
        # Convert scalar to timedelta based on unit
        if unit == "ns":
            delta = pl.duration(nanoseconds=int(arg))
        elif unit == "us":
            delta = pl.duration(microseconds=int(arg))
        elif unit == "ms":
            delta = pl.duration(milliseconds=int(arg))
        elif unit == "s":
            delta = pl.duration(seconds=int(arg))
        elif unit == "m":
            delta = pl.duration(minutes=int(arg))
        elif unit == "h":
            delta = pl.duration(hours=int(arg))
        elif unit == "d":
            delta = pl.duration(days=int(arg))
        else:
            delta = pl.duration(days=int(arg))
        return Series([delta])
    else:
        raise ValueError(f"Unsupported type for to_timedelta: {type(arg)}")
