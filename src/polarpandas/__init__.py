"""
PolarPandas - A pandas-compatible API layer on top of Polars.

PolarPandas provides a pandas-compatible API built on top of Polars, offering
the familiar pandas interface you know while harnessing the blazing-fast performance
of Polars under the hood.

This module exports the main classes and functions for data manipulation:

Classes
-------
DataFrame : Main DataFrame class for eager data manipulation
LazyFrame : LazyFrame class for deferred execution and optimization
Series : Series class for single-column data operations
Index : Index class for DataFrame index management

I/O Functions
-------------
read_csv, read_parquet, read_json, read_excel, read_sql, read_feather
    Eager I/O operations that load data immediately
scan_csv, scan_parquet, scan_json
    Lazy I/O operations for large files (deferred loading)

Data Manipulation
-----------------
concat, merge, get_dummies, pivot_table
    Functions for combining and transforming DataFrames

Datetime Utilities
------------------
date_range, to_datetime
    Functions for working with datetime data

Utility Functions
-----------------
isna, notna, cut
    Helper functions for data analysis

Examples
--------
>>> import polarpandas as ppd
>>> df = ppd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
>>> df["C"] = df["A"] + df["B"]
>>> print(df.head())

Notes
-----
- PolarPandas is designed to be a drop-in replacement for pandas where possible
- For maximum performance with large datasets, use LazyFrame for lazy execution
- Some pandas features may have limitations due to Polars architecture differences

See Also
--------
pandas : The original pandas library
polars : The underlying Polars library
"""

# Core classes
# Datetime utilities
from .datetime import (
    date_range,
    to_datetime,
)
from .frame import DataFrame
from .index import Index

# I/O operations
from .io import (
    read_csv,
    read_excel,
    read_feather,
    read_json,
    read_parquet,
    read_sql,
    scan_csv,
    scan_json,
    scan_parquet,
)
from .lazyframe import LazyFrame

# Data manipulation operations
from .operations import (
    concat,
    get_dummies,
    merge,
    pivot_table,
)
from .series import Series

# Utility functions
from .utils import (
    cut,
    isna,
    notna,
)

# Version
__version__ = "0.5.0"

# Main exports
__all__ = [
    # Core classes
    "DataFrame",
    "LazyFrame",
    "Series",
    "Index",
    # I/O operations
    "read_csv",
    "read_parquet",
    "read_json",
    "read_excel",
    "read_sql",
    "read_feather",
    "scan_csv",
    "scan_parquet",
    "scan_json",
    # Data manipulation
    "concat",
    "merge",
    "get_dummies",
    "pivot_table",
    # Datetime utilities
    "date_range",
    "to_datetime",
    # Utility functions
    "isna",
    "notna",
    "cut",
]
