"""
Shared Polars dtype groupings used across PolarPandas modules.
"""

from __future__ import annotations

import polars as pl

INTEGER_DTYPES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
)

FLOAT_DTYPES = (
    pl.Float32,
    pl.Float64,
)

NUMERIC_DTYPES = INTEGER_DTYPES + FLOAT_DTYPES

BOOLEAN_DTYPES = (pl.Boolean,)

NUMERIC_OR_BOOLEAN_DTYPES = NUMERIC_DTYPES + BOOLEAN_DTYPES

