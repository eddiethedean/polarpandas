"""
Index implementation wrapping Polars Series with pandas-like API.

This module provides the Index class that wraps a Polars Series to represent
DataFrame indices. It provides a pandas-compatible interface for index operations
while using Polars for the underlying data storage.

Classes
-------
Index : Index object for DataFrame index management

Examples
--------
>>> import polarpandas as ppd
>>> idx = ppd.Index([1, 2, 3, 4, 5])
>>> # Use as DataFrame index
>>> df = ppd.DataFrame({"A": [10, 20, 30]}, index=idx)

Notes
-----
- Index is stored separately from DataFrame data in Polars
- Index operations may be slower than column operations
"""

from typing import Any, Iterator, List, Optional, Tuple, Union

import polars as pl


class Index:
    """
    Immutable sequence used for indexing and alignment.

    Index is the basic object for storing axis labels (row labels) for
    DataFrames in PolarPandas. It wraps a Polars Series and provides a
    pandas-compatible interface for index operations.

    Parameters
    ----------
    data : array-like, pl.Series, or None, optional
        Input data. Can be:
        - List or array-like of values
        - Existing Polars Series
        - None for empty Index

    Attributes
    ----------
    _series : pl.Series
        The underlying Polars Series storing the index values.

    Examples
    --------
    >>> import polarpandas as ppd
    >>> # Create Index from list
    >>> idx = ppd.Index([1, 2, 3, 4, 5])
    >>> # Use with DataFrame
    >>> df = ppd.DataFrame({"A": [10, 20, 30]}, index=idx)

    See Also
    --------
    DataFrame : Two-dimensional data structure with Index support
    Series : One-dimensional data structure

    Notes
    -----
    - Index values are stored in a Polars Series
    - Index is immutable (cannot be modified after creation)
    - Index operations delegate to underlying Polars Series
    """

    def __init__(
        self,
        data: Optional[Union[List[Any], pl.Series]] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize an Index from various data sources.

        Parameters
        ----------
        data : list, pl.Series, or None
            Data to initialize the Index with
        """
        if data is None:
            self._series = pl.Series(name="index", values=[])
        elif isinstance(data, pl.Series):
            self._series = data
        else:
            # Handle list or other array-like data
            self._series = pl.Series("index", data)

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the underlying Polars Series.

        This allows transparent access to Polars methods and properties.
        """
        if name.startswith("_"):
            # Avoid infinite recursion for private attributes
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        try:
            attr = getattr(self._series, name)
            return attr
        except AttributeError as e:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from e

    def __repr__(self) -> str:
        """Return string representation of the Index."""
        return repr(self._series)

    def __str__(self) -> str:
        """Return string representation of the Index."""
        return str(self._series)

    def __len__(self) -> int:
        """Return the length of the Index."""
        return len(self._series)

    def __iter__(self) -> Iterator[Any]:
        """Return an iterator over the Index values."""
        return iter(self._series.to_list())

    def tolist(self) -> List[Any]:
        """Return the Index values as a list."""
        return self._series.to_list()

    @property
    def shape(self) -> Tuple[int]:
        """Return the shape of the Index."""
        return (len(self._series),)

    @property
    def size(self) -> int:
        """Return the size of the Index."""
        return len(self._series)
