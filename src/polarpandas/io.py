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
from .utils import convert_schema_to_polars


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
    dtype : dict, pl.Schema, or None, optional
        Schema specification for columns. Can be pandas-style dict or Polars schema.
        See DataFrame constructor for details.
    schema : dict, pl.Schema, or None, optional
        Direct Polars schema specification (alternative to dtype).
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

    # Handle dtype/schema parameters
    dtype = kwargs.pop("dtype", None)
    schema = kwargs.pop("schema", None)

    # If both are provided, schema takes precedence
    schema_to_use = schema if schema is not None else dtype

    if schema_to_use is not None:
        polars_schema = convert_schema_to_polars(schema_to_use)
        if polars_schema is not None:
            kwargs["schema"] = polars_schema

    return LazyFrame(pl.scan_csv(path, **kwargs))


def scan_parquet(path: str, **kwargs: Any) -> LazyFrame:
    """
    Scan a Parquet file into LazyFrame for lazy execution.

    Parameters
    ----------
    path : str
        Path to Parquet file
    dtype : dict, pl.Schema, or None, optional
        Schema specification for columns. Can be pandas-style dict or Polars schema.
        See DataFrame constructor for details.
    schema : dict, pl.Schema, or None, optional
        Direct Polars schema specification (alternative to dtype).
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

    # Handle dtype/schema parameters
    dtype = kwargs.pop("dtype", None)
    schema = kwargs.pop("schema", None)

    # If both are provided, schema takes precedence
    schema_to_use = schema if schema is not None else dtype

    # Note: Parquet files don't support schema parameter in scan_parquet,
    # so we need to cast after scanning using lazy expressions
    lf = pl.scan_parquet(path, **kwargs)

    # Apply schema conversion if provided (cast columns using lazy expressions)
    if schema_to_use is not None:
        polars_schema = convert_schema_to_polars(schema_to_use)
        if polars_schema:
            # Build with_columns expression for casting
            cast_expressions = [
                pl.col(col).cast(dtype_val) for col, dtype_val in polars_schema.items()
            ]
            if cast_expressions:
                # Apply casts using lazy expressions
                # This will be evaluated during collect
                lf = lf.with_columns(cast_expressions)

    return LazyFrame(lf)


def scan_json(path: str, **kwargs: Any) -> LazyFrame:
    """
    Scan a JSON file into LazyFrame for lazy execution.

    Parameters
    ----------
    path : str
        Path to JSON file
    dtype : dict, pl.Schema, or None, optional
        Schema specification for columns. Can be pandas-style dict or Polars schema.
        See DataFrame constructor for details.
    schema : dict, pl.Schema, or None, optional
        Direct Polars schema specification (alternative to dtype).
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

    # Handle dtype/schema parameters
    dtype = kwargs.pop("dtype", None)
    schema = kwargs.pop("schema", None)

    # If both are provided, schema takes precedence
    schema_to_use = schema if schema is not None else dtype

    if schema_to_use is not None:
        polars_schema = convert_schema_to_polars(schema_to_use)
        if polars_schema is not None:
            kwargs["schema"] = polars_schema

    return LazyFrame(pl.scan_ndjson(path, **kwargs))
