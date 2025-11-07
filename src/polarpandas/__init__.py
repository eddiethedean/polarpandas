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
    bdate_range,
    date_range,
    interval_range,
    period_range,
    timedelta_range,
    to_datetime,
    to_timedelta,
)
from .frame import DataFrame
from .index import Index, MultiIndex

# I/O operations
from .io import (
    read_clipboard,
    read_csv,
    read_excel,
    read_feather,
    read_fwf,
    read_hdf,
    read_html,
    read_iceberg,
    read_json,
    read_orc,
    read_parquet,
    read_pickle,
    read_sas,
    read_spss,
    read_sql,
    read_sql_query,
    read_sql_table,
    read_stata,
    read_table,
    read_xml,
    scan_csv,
    scan_json,
    scan_parquet,
    to_pickle,
)
from .lazyframe import LazyFrame

# Data manipulation operations
from .operations import (
    array,
    col,
    concat,
    crosstab,
    describe_option,
    eval,
    factorize,
    from_dummies,
    get_dummies,
    get_option,
    infer_freq,
    json_normalize,
    lreshape,
    melt,
    merge,
    merge_asof,
    merge_ordered,
    option_context,
    pivot,
    pivot_table,
    qcut,
    reset_option,
    set_eng_float_format,
    set_option,
    show_versions,
    test,
    wide_to_long,
)
from .series import Series

# Utility functions
from .utils import (
    cut,
    isna,
    isnull,
    notna,
    notnull,
    to_numeric,
    unique,
)

# Version
__version__ = "0.9.0"

# Main exports
__all__ = [
    # Core classes
    "DataFrame",
    "LazyFrame",
    "Series",
    "Index",
    "MultiIndex",
    # I/O operations
    "read_csv",
    "read_parquet",
    "read_json",
    "read_excel",
    "read_sql",
    "read_feather",
    "read_table",
    "read_clipboard",
    "read_fwf",
    "read_hdf",
    "read_html",
    "read_iceberg",
    "read_orc",
    "read_pickle",
    "read_sas",
    "read_spss",
    "read_sql_query",
    "read_sql_table",
    "read_stata",
    "read_xml",
    "to_pickle",
    "scan_csv",
    "scan_parquet",
    "scan_json",
    # Data manipulation
    "concat",
    "merge",
    "get_dummies",
    "pivot_table",
    "melt",
    "pivot",
    "factorize",
    "qcut",
    "wide_to_long",
    "array",
    "col",
    "crosstab",
    "from_dummies",
    "lreshape",
    "merge_asof",
    "merge_ordered",
    "eval",
    "json_normalize",
    # Configuration
    "describe_option",
    "get_option",
    "option_context",
    "reset_option",
    "set_eng_float_format",
    "set_option",
    "show_versions",
    "test",
    "infer_freq",
    # Datetime utilities
    "date_range",
    "to_datetime",
    "bdate_range",
    "timedelta_range",
    "period_range",
    "interval_range",
    "to_timedelta",
    # Utility functions
    "isna",
    "isnull",
    "notna",
    "notnull",
    "cut",
    "to_numeric",
    "unique",
]
