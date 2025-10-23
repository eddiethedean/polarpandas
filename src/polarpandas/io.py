"""
I/O operations for PolarPandas.

Provides pandas-compatible I/O functions for reading and writing data.
"""

from typing import Any

from .frame import DataFrame


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
