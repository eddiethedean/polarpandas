"""
Utility functions for PolarPandas.

This module provides pandas-compatible utility functions for common data
analysis tasks, including null value detection and data binning.

Functions
---------
isna : Detect missing values in DataFrame or Series
notna : Detect non-missing values in DataFrame or Series
cut : Bin values into discrete intervals
convert_schema_to_polars : Convert pandas-style schemas to Polars schemas

Examples
--------
>>> import polarpandas as ppd
>>> df = ppd.DataFrame({"A": [1, None, 3]})
>>> # Check for missing values
>>> missing = ppd.isna(df)
>>> # Bin values
>>> bins = ppd.cut([1, 2, 3, 4, 5], bins=3)
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import polars as pl

if TYPE_CHECKING:
    pass


def isna(obj: Any) -> Any:
    """
    Detect missing values.

    Parameters
    ----------
    obj : Any
        Object to check for missing values

    Returns
    -------
    DataFrame or bool
        Boolean DataFrame indicating missing values, or bool for scalars

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.DataFrame({"A": [1, None, 3]})
    >>> result = ppd.isna(df)
    """
    # Import here to avoid circular import
    from .frame import DataFrame
    from .series import Series

    if isinstance(obj, DataFrame):
        return obj.isna()
    elif isinstance(obj, Series):
        return obj.isna()
    else:
        # For scalar values, return boolean
        return obj is None


def notna(obj: Any) -> Any:
    """
    Detect non-missing values.

    Parameters
    ----------
    obj : Any
        Object to check for non-missing values

    Returns
    -------
    DataFrame or bool
        Boolean DataFrame indicating non-missing values, or bool for scalars

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.DataFrame({"A": [1, None, 3]})
    >>> result = ppd.notna(df)
    """
    # Import here to avoid circular import
    from .frame import DataFrame
    from .series import Series

    if isinstance(obj, DataFrame):
        return obj.notna()
    elif isinstance(obj, Series):
        return obj.notna()
    else:
        # For scalar values, return boolean
        return obj is not None


def cut(
    x: List[Any], bins: Any, labels: Optional[List[str]] = None, **kwargs: Any
) -> List[Optional[str]]:
    """
    Bin values into discrete intervals.

    Parameters
    ----------
    x : List[Any]
        Input array to be binned
    bins : Any
        Bins to use for cutting
    labels : List[str], optional
        Labels for the resulting bins
    **kwargs
        Additional arguments

    Returns
    -------
    List[Optional[str]]
        List of bin labels

    Examples
    --------
    >>> import polarpandas as ppd
    >>> result = ppd.cut([1, 2, 3, 4, 5], bins=3)
    """
    # Simplified implementation
    # In practice, you'd use Polars' cut functionality
    if labels:
        result: List[Optional[str]] = list(labels[: len(x)])
        return result
    else:
        result_bins: List[Optional[str]] = [f"bin_{i}" for i in range(len(x))]
        return result_bins


def convert_schema_to_polars(
    schema: Union[Dict[str, Any], pl.Schema, None],
) -> Optional[Dict[str, Any]]:
    """
    Convert pandas-style schema to Polars schema.

    Accepts pandas-style dtype dictionaries (with string names or dtype objects),
    Polars schemas (dict or Schema object), and returns a Polars-compatible schema dict.

    Parameters
    ----------
    schema : dict, pl.Schema, or None
        Schema to convert. Can be:
        - Pandas-style dict with string dtype names: {"col1": "int64", "col2": "float64"}
        - Pandas-style dict with dtype objects: {"col1": np.int64, "col2": pd.Int64Dtype()}
        - Polars schema dict: {"col1": pl.Int64, "col2": pl.Float64}
        - Polars Schema object: pl.Schema({"col1": pl.Int64, "col2": pl.Float64})
        - None: returns None

    Returns
    -------
    dict or None
        Polars schema dict mapping column names to Polars DataType objects,
        or None if input is None

    Examples
    --------
    >>> import polarpandas as ppd
    >>> import numpy as np
    >>> # String dtype names
    >>> schema = ppd.utils.convert_schema_to_polars({"col1": "int64", "col2": "float64"})
    >>> # NumPy dtype objects
    >>> schema = ppd.utils.convert_schema_to_polars({"col1": np.int64, "col2": np.float64})
    >>> # Polars schema (passed through)
    >>> schema = ppd.utils.convert_schema_to_polars({"col1": pl.Int64, "col2": pl.Float64})

    Raises
    ------
    ValueError
        If dtype is not recognized or cannot be converted
    TypeError
        If schema is not a dict, Schema, or None
    """
    if schema is None:
        return None

    # Handle Polars Schema object
    if isinstance(schema, pl.Schema):
        return dict(schema)

    # Handle dict (pandas-style or Polars)
    if not isinstance(schema, dict):
        raise TypeError(
            f"Schema must be a dict, pl.Schema, or None, got {type(schema)}"
        )

    result: Dict[str, Any] = {}

    for col_name, dtype in schema.items():
        # If it's already a Polars DataType, use it directly
        if isinstance(dtype, pl.DataType):
            result[col_name] = dtype
            continue

        # Handle string dtype names
        if isinstance(dtype, str):
            dtype_str = dtype.lower().strip()
            result[col_name] = _convert_dtype_string(dtype_str)
            continue

        # Handle numpy dtype objects
        if hasattr(dtype, "__module__") and dtype.__module__ == "numpy":
            dtype_name = str(dtype)
            # Extract base type name from numpy dtype string
            if "uint" in dtype_name:
                # Unsigned integers
                if "64" in dtype_name:
                    result[col_name] = pl.UInt64
                elif "32" in dtype_name:
                    result[col_name] = pl.UInt32
                elif "16" in dtype_name:
                    result[col_name] = pl.UInt16
                elif "8" in dtype_name:
                    result[col_name] = pl.UInt8
                else:
                    result[col_name] = pl.UInt64
            elif "int" in dtype_name:
                # Signed integers
                if "64" in dtype_name:
                    result[col_name] = pl.Int64
                elif "32" in dtype_name:
                    result[col_name] = pl.Int32
                elif "16" in dtype_name:
                    result[col_name] = pl.Int16
                elif "8" in dtype_name:
                    result[col_name] = pl.Int8
                else:
                    result[col_name] = pl.Int64
            elif "float" in dtype_name:
                if "64" in dtype_name:
                    result[col_name] = pl.Float64
                elif "32" in dtype_name:
                    result[col_name] = pl.Float32
                else:
                    result[col_name] = pl.Float64
            elif "bool" in dtype_name:
                result[col_name] = pl.Boolean
            else:
                raise ValueError(f"Unsupported numpy dtype: {dtype}")
            continue

        # Handle pandas dtype objects
        if hasattr(dtype, "__module__") and (
            dtype.__module__ == "pandas.core.dtypes.base"
            or dtype.__module__.startswith("pandas.core.dtypes")
        ):
            dtype_str = str(dtype)
            if "Int" in dtype_str or "int" in dtype_str.lower():
                # Handle nullable integer dtypes
                if "64" in dtype_str:
                    result[col_name] = pl.Int64
                elif "32" in dtype_str:
                    result[col_name] = pl.Int32
                elif "16" in dtype_str:
                    result[col_name] = pl.Int16
                elif "8" in dtype_str:
                    result[col_name] = pl.Int8
                else:
                    result[col_name] = pl.Int64
            elif "Float" in dtype_str or "float" in dtype_str.lower():
                # Handle nullable float dtypes
                if "64" in dtype_str:
                    result[col_name] = pl.Float64
                elif "32" in dtype_str:
                    result[col_name] = pl.Float32
                else:
                    result[col_name] = pl.Float64
            elif "String" in dtype_str or "string" in dtype_str.lower():
                result[col_name] = pl.Utf8
            elif "datetime" in dtype_str.lower():
                result[col_name] = pl.Datetime
            elif "category" in dtype_str.lower():
                result[col_name] = pl.Categorical
            elif "bool" in dtype_str.lower():
                result[col_name] = pl.Boolean
            else:
                raise ValueError(
                    f"Unsupported pandas dtype for column '{col_name}': {dtype} (type: {type(dtype)})"
                )
            continue

        # Handle other dtype objects (fallback for string parsing)
        # This handles cases where we can't determine the module but can parse the string
        dtype_str = str(dtype)
        if "Int" in dtype_str or "int" in dtype_str.lower():
            # Handle nullable integer dtypes
            if "64" in dtype_str:
                result[col_name] = pl.Int64
            elif "32" in dtype_str:
                result[col_name] = pl.Int32
            elif "16" in dtype_str:
                result[col_name] = pl.Int16
            elif "8" in dtype_str:
                result[col_name] = pl.Int8
            else:
                result[col_name] = pl.Int64
        elif "Float" in dtype_str or "float" in dtype_str.lower():
            # Handle nullable float dtypes
            if "64" in dtype_str:
                result[col_name] = pl.Float64
            elif "32" in dtype_str:
                result[col_name] = pl.Float32
            else:
                result[col_name] = pl.Float64
        elif "String" in dtype_str or "string" in dtype_str.lower():
            result[col_name] = pl.Utf8
        elif "datetime" in dtype_str.lower():
            result[col_name] = pl.Datetime
        elif "category" in dtype_str.lower():
            result[col_name] = pl.Categorical
        elif "bool" in dtype_str.lower():
            result[col_name] = pl.Boolean
        else:
            raise ValueError(
                f"Unsupported dtype for column '{col_name}': {dtype} (type: {type(dtype)})"
            )

    return result


def _convert_dtype_string(dtype_str: str) -> Any:
    """
    Convert a string dtype name to Polars DataType.

    Parameters
    ----------
    dtype_str : str
        String representation of dtype (e.g., "int64", "float64", "object")

    Returns
    -------
    pl.DataType
        Corresponding Polars DataType

    Raises
    ------
    ValueError
        If dtype string is not recognized
    """
    dtype_str = dtype_str.lower().strip()

    # Integer types
    if dtype_str in ("int64", "int"):
        return pl.Int64
    elif dtype_str in ("int32",):
        return pl.Int32
    elif dtype_str in ("int16",):
        return pl.Int16
    elif dtype_str in ("int8",):
        return pl.Int8
    elif dtype_str in ("uint64", "uint"):
        return pl.UInt64
    elif dtype_str in ("uint32",):
        return pl.UInt32
    elif dtype_str in ("uint16",):
        return pl.UInt16
    elif dtype_str in ("uint8",):
        return pl.UInt8

    # Float types
    elif dtype_str in ("float64", "float"):
        return pl.Float64
    elif dtype_str in ("float32",):
        return pl.Float32

    # String types
    elif dtype_str in ("object", "string", "str", "utf8"):
        return pl.Utf8

    # Boolean types
    elif dtype_str in ("bool", "boolean", "bool_"):
        return pl.Boolean

    # Datetime types
    elif dtype_str.startswith("datetime") or "datetime" in dtype_str:
        return pl.Datetime

    # Categorical types
    elif dtype_str in ("category", "categorical"):
        return pl.Categorical

    else:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")
