"""
Utility functions for PolarPandas.

This module provides pandas-compatible utility functions for common data
analysis tasks, including null value detection and data binning.

Functions
---------
isna : Detect missing values in DataFrame or Series
notna : Detect non-missing values in DataFrame or Series
cut : Bin values into discrete intervals

Examples
--------
>>> import polarpandas as ppd
>>> df = ppd.DataFrame({"A": [1, None, 3]})
>>> # Check for missing values
>>> missing = ppd.isna(df)
>>> # Bin values
>>> bins = ppd.cut([1, 2, 3, 4, 5], bins=3)
"""

from typing import Any, List, Optional

from .frame import DataFrame
from .series import Series


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
