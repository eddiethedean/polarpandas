"""
PolarPandas - A pandas-compatible API layer on top of Polars.

Provides pandas-like functionality with Polars performance.
"""

from polarpandas.frame import DataFrame
from polarpandas.series import Series
from polarpandas.index import Index
import polars as pl
from typing import Any, List, Optional


# Module-level read functions (pandas-compatible)
def read_csv(path, **kwargs):
    """
    Read a CSV file into DataFrame.

    Parameters
    ----------
    path : str
        Path to CSV file
    **kwargs
        Additional arguments passed to Polars read_csv()

    Returns
    -------
    DataFrame
        DataFrame loaded from CSV

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_csv("data.csv")
    """
    return DataFrame.read_csv(path, **kwargs)


def read_parquet(path, **kwargs):
    """
    Read a Parquet file into DataFrame.

    Parameters
    ----------
    path : str
        Path to Parquet file
    **kwargs
        Additional arguments passed to Polars read_parquet()

    Returns
    -------
    DataFrame
        DataFrame loaded from Parquet

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_parquet("data.parquet")
    """
    return DataFrame.read_parquet(path, **kwargs)


def read_json(path, **kwargs):
    """
    Read a JSON file into DataFrame.

    Parameters
    ----------
    path : str
        Path to JSON file
    **kwargs
        Additional arguments passed to Polars read_json()

    Returns
    -------
    DataFrame
        DataFrame loaded from JSON

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_json("data.json")
    """
    return DataFrame.read_json(path, **kwargs)


def read_excel(path, **kwargs):
    """
    Read an Excel file into DataFrame.

    Parameters
    ----------
    path : str
        Path to Excel file
    **kwargs
        Additional arguments passed to Polars read_excel()

    Returns
    -------
    DataFrame
        DataFrame loaded from Excel

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_excel("data.xlsx")
    """
    return DataFrame(pl.read_excel(path, **kwargs))


# Module-level data manipulation functions
def concat(objs, axis=0, **kwargs):
    """
    Concatenate DataFrames.

    Parameters
    ----------
    objs : list of DataFrame
        DataFrames to concatenate
    axis : {0, 1}, default 0
        0 for vertical, 1 for horizontal
    **kwargs
        Additional arguments

    Returns
    -------
    DataFrame
        Concatenated DataFrame

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df1 = ppd.DataFrame({"a": [1, 2]})
    >>> df2 = ppd.DataFrame({"a": [3, 4]})
    >>> result = ppd.concat([df1, df2])
    """
    # Extract underlying Polars DataFrames
    pl_dfs = [df._df if isinstance(df, DataFrame) else df for df in objs]

    if axis == 0:
        # Vertical concatenation
        result = pl.concat(pl_dfs, how="vertical", **kwargs)
    else:
        # Horizontal concatenation
        result = pl.concat(pl_dfs, how="horizontal", **kwargs)

    return DataFrame(result)


def merge(left, right, **kwargs):
    """
    Merge two DataFrames.

    Parameters
    ----------
    left : DataFrame
        Left DataFrame
    right : DataFrame
        Right DataFrame
    **kwargs
        Additional arguments passed to join()

    Returns
    -------
    DataFrame
        Merged DataFrame

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df1 = ppd.DataFrame({"key": ["A", "B"], "val1": [1, 2]})
    >>> df2 = ppd.DataFrame({"key": ["A", "B"], "val2": [3, 4]})
    >>> result = ppd.merge(df1, df2, on="key")
    """
    return left.merge(right, **kwargs)


def get_dummies(data, **kwargs):
    """
    Convert categorical variables into dummy/indicator variables.

    Parameters
    ----------
    data : DataFrame or Series
        Data to encode
    **kwargs
        Additional arguments

    Returns
    -------
    DataFrame
        One-hot encoded DataFrame

    Examples
    --------
    >>> import polarpandas as ppd
    >>> s = ppd.Series(["a", "b", "a", "c"])
    >>> result = ppd.get_dummies(s)
    """
    if isinstance(data, Series):
        # Use Polars to_dummies
        result = pl.DataFrame({"col": data._series}).to_dummies(columns=["col"])
        return DataFrame(result)
    elif isinstance(data, DataFrame):
        # Convert all string/categorical columns
        return DataFrame(data._df.to_dummies())
    else:
        # Try to handle as list
        s = Series(data)
        return get_dummies(s)


def cut(x, bins, labels=None, **kwargs):
    """
    Bin values into discrete intervals.

    Parameters
    ----------
    x : Series or array-like
        Input data
    bins : int or sequence
        Number of bins or bin edges
    labels : array-like, optional
        Labels for bins
    **kwargs
        Additional arguments

    Returns
    -------
    Series
        Binned data

    Examples
    --------
    >>> import polarpandas as ppd
    >>> s = ppd.Series([1, 7, 5, 4, 6, 3])
    >>> result = ppd.cut(s, bins=3)
    """
    if isinstance(x, Series):
        series = x._series
    else:
        series = pl.Series(x)

    # Use Polars cut function
    try:
        if isinstance(bins, int):
            # Equal-width bins
            result = series.cut(breaks=[i for i in range(bins + 1)])
        else:
            # Custom bin edges
            result = series.cut(breaks=list(bins))

        return Series(result)
    except (TypeError, AttributeError):
        # Polars cut may not be available or has different API
        # Return a simple categorical based on bin ranges
        if isinstance(bins, int):
            result_list = pd_cut_simple(series.to_list(), bins)
        else:
            result_list = pd_cut_simple(series.to_list(), bins)
        return Series(pl.Series(result_list))


def pd_cut_simple(data: List[Any], bins: Any) -> List[Optional[str]]:
    """Simple binning function."""
    import numpy as np

    if isinstance(bins, int):
        min_val, max_val = min(data), max(data)
        edges = np.linspace(min_val, max_val, bins + 1).tolist()  # type: ignore
    else:
        edges = sorted(bins)

    result: List[Optional[str]] = []
    for val in data:
        found = False
        for i in range(len(edges) - 1):
            if edges[i] <= val <= edges[i + 1]:
                result.append(f"({edges[i]:.1f}, {edges[i + 1]:.1f}]")
                found = True
                break
        if not found:
            result.append(None)
    return result


def pivot_table(data, values=None, index=None, columns=None, aggfunc="mean", **kwargs):
    """
    Create a spreadsheet-style pivot table.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame
    values : str, optional
        Column to aggregate
    index : str or list, optional
        Column(s) for index
    columns : str, optional
        Column for columns
    aggfunc : str, default 'mean'
        Aggregation function
    **kwargs
        Additional arguments

    Returns
    -------
    DataFrame
        Pivoted DataFrame

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.DataFrame({
    ...     "A": ["foo", "foo", "bar", "bar"],
    ...     "B": ["one", "two", "one", "two"],
    ...     "C": [1, 2, 3, 4]
    ... })
    >>> result = ppd.pivot_table(df, values="C", index="A", columns="B")
    """
    return data.pivot(index=index, columns=columns, values=values)


def date_range(start=None, end=None, periods=None, freq="D", **kwargs):
    """
    Generate a range of dates.

    Parameters
    ----------
    start : str or datetime, optional
        Start date
    end : str or datetime, optional
        End date
    periods : int, optional
        Number of periods
    freq : str, default 'D'
        Frequency (D=day, H=hour, etc.)
    **kwargs
        Additional arguments

    Returns
    -------
    Series
        Date range as Series

    Examples
    --------
    >>> import polarpandas as ppd
    >>> dates = ppd.date_range("2021-01-01", periods=5)
    """
    # Use Polars date_range with correct parameters
    try:
        if periods is not None and start is not None:
            # Calculate end date based on periods
            import datetime

            if isinstance(start, str):
                start_dt = datetime.datetime.fromisoformat(start)
            else:
                start_dt = start

            # Create date range
            result = pl.date_range(
                start=start_dt,
                periods=periods,
                interval=freq,
                eager=True,  # type: ignore
            )
        else:
            result = pl.date_range(start=start, end=end, interval=freq, eager=True)

        return Series(result)
    except (TypeError, AttributeError):
        # Fallback: create simple date range
        import datetime

        dates = []
        if isinstance(start, str):
            current = datetime.datetime.fromisoformat(start)
        else:
            current = start

        if periods:
            for _ in range(periods):
                dates.append(current)
                current += datetime.timedelta(days=1)  # Simple day increment

        return Series(pl.Series(dates))


def to_datetime(arg, **kwargs):
    """
    Convert argument to datetime.

    Parameters
    ----------
    arg : str, Series, or array-like
        Data to convert
    **kwargs
        Additional arguments

    Returns
    -------
    Series or scalar
        Converted datetime

    Examples
    --------
    >>> import polarpandas as ppd
    >>> dates = ppd.to_datetime(["2021-01-01", "2021-01-02"])
    """
    if isinstance(arg, Series):
        return Series(arg._series.str.to_datetime(**kwargs))
    elif isinstance(arg, list):
        s = pl.Series(arg)
        return Series(s.str.to_datetime(**kwargs))
    else:
        # Single value
        s = pl.Series([arg])
        return s.str.to_datetime(**kwargs)[0]


# Utility functions
def isna(obj):
    """
    Detect missing values.

    Parameters
    ----------
    obj : DataFrame, Series, or scalar
        Object to check

    Returns
    -------
    DataFrame, Series, or bool
        Boolean mask indicating missing values

    Examples
    --------
    >>> import polarpandas as ppd
    >>> s = ppd.Series([1, None, 3])
    >>> ppd.isna(s)
    """
    if isinstance(obj, (DataFrame, Series)):
        return obj.isna()
    else:
        return obj is None


def notna(obj):
    """
    Detect non-missing values.

    Parameters
    ----------
    obj : DataFrame, Series, or scalar
        Object to check

    Returns
    -------
    DataFrame, Series, or bool
        Boolean mask indicating non-missing values

    Examples
    --------
    >>> import polarpandas as ppd
    >>> s = ppd.Series([1, None, 3])
    >>> ppd.notna(s)
    """
    if isinstance(obj, (DataFrame, Series)):
        return obj.notna()
    else:
        return obj is not None


# Export all public functions
__all__ = [
    # Core classes
    "DataFrame",
    "Series",
    "Index",
    # Read functions
    "read_csv",
    "read_parquet",
    "read_json",
    "read_excel",
    # Data manipulation
    "concat",
    "merge",
    "get_dummies",
    "cut",
    "pivot_table",
    # Datetime utilities
    "date_range",
    "to_datetime",
    # Utility functions
    "isna",
    "notna",
]


# Version info
__version__ = "0.1.0"
