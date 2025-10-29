"""
PolarPandas - A pandas-compatible API layer on top of Polars.

Provides pandas-like functionality with Polars performance.
"""

# Core classes
# Datetime utilities
from .datetime import (
    date_range,
    to_datetime,
)
from .frame import DataFrame
from .index import Index
from .lazyframe import LazyFrame

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
__version__ = "0.2.0"

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
