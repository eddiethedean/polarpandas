"""
I/O operations for PolarPandas.

This module provides pandas-compatible functions for reading and writing data
in various formats. It supports both eager loading (immediate) and lazy loading
(deferred) operations for different file sizes and use cases.

Functions
---------
Eager I/O (immediate loading):
    read_csv, read_parquet, read_json, read_excel, read_sql, read_feather
    Load data immediately into DataFrame

Lazy I/O (deferred loading):
    scan_csv, scan_parquet, scan_json
    Load data lazily into LazyFrame for optimization

Examples
--------
>>> import polarpandas as ppd
>>> # Eager loading
>>> df = ppd.read_csv("data.csv")
>>> # Lazy loading for large files
>>> lf = ppd.scan_csv("large_file.csv")
>>> df = lf.collect()  # Materialize when ready

Notes
-----
- Use eager I/O for small to medium files (< 1M rows)
- Use lazy I/O for large files or when building complex query chains
- Lazy I/O allows Polars to optimize the query plan before execution
"""

from typing import Any

from .frame import DataFrame
from .lazyframe import LazyFrame


def read_csv(path: str, **kwargs: Any) -> DataFrame:
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


def read_parquet(path: str, **kwargs: Any) -> DataFrame:
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


def read_json(path: str, **kwargs: Any) -> DataFrame:
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


def read_excel(path: str, **kwargs: Any) -> DataFrame:
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
    # Excel reading not yet implemented in polarpandas
    raise NotImplementedError("read_excel not yet implemented")


def read_sql(sql: str, con: Any, **kwargs: Any) -> DataFrame:
    """
    Read SQL query into DataFrame.

    Parameters
    ----------
    sql : str
        SQL query string
    con : Any
        Database connection
    **kwargs
        Additional arguments passed to Polars read_sql()

    Returns
    -------
    DataFrame
        DataFrame loaded from SQL query

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_sql("SELECT * FROM table", connection)
    """
    return DataFrame.read_sql(sql, con, **kwargs)


def read_feather(path: str, **kwargs: Any) -> DataFrame:
    """
    Read a Feather file into DataFrame.

    Parameters
    ----------
    path : str
        Path to Feather file
    **kwargs
        Additional arguments passed to Polars read_ipc()

    Returns
    -------
    DataFrame
        DataFrame loaded from Feather

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_feather("data.feather")
    """
    return DataFrame.read_feather(path, **kwargs)


def scan_csv(path: str, **kwargs: Any) -> LazyFrame:
    """
    Scan a CSV file into LazyFrame for lazy execution.

    Parameters
    ----------
    path : str
        Path to CSV file
    **kwargs
        Additional arguments passed to Polars scan_csv()

    Returns
    -------
    LazyFrame
        LazyFrame loaded from CSV

    Examples
    --------
    >>> import polarpandas as ppd
    >>> lf = ppd.scan_csv("data.csv")
    >>> df = lf.collect()  # Materialize when ready
    """
    import polars as pl

    return LazyFrame(pl.scan_csv(path, **kwargs))


def scan_parquet(path: str, **kwargs: Any) -> LazyFrame:
    """
    Scan a Parquet file into LazyFrame for lazy execution.

    Parameters
    ----------
    path : str
        Path to Parquet file
    **kwargs
        Additional arguments passed to Polars scan_parquet()

    Returns
    -------
    LazyFrame
        LazyFrame loaded from Parquet

    Examples
    --------
    >>> import polarpandas as ppd
    >>> lf = ppd.scan_parquet("data.parquet")
    >>> df = lf.collect()  # Materialize when ready
    """
    import polars as pl

    return LazyFrame(pl.scan_parquet(path, **kwargs))


def scan_json(path: str, **kwargs: Any) -> LazyFrame:
    """
    Scan a JSON file into LazyFrame for lazy execution.

    Parameters
    ----------
    path : str
        Path to JSON file
    **kwargs
        Additional arguments passed to Polars scan_ndjson()

    Returns
    -------
    LazyFrame
        LazyFrame loaded from JSON

    Examples
    --------
    >>> import polarpandas as ppd
    >>> lf = ppd.scan_json("data.json")
    >>> df = lf.collect()  # Materialize when ready
    """
    import polars as pl

    return LazyFrame(pl.scan_ndjson(path, **kwargs))
