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

import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import polars as pl

if TYPE_CHECKING:
    from polarpandas.frame import DataFrame
    from polarpandas.series import Series


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

    # Boolean/Logical Operations
    def all(
        self,
        axis: Optional[int] = None,
        bool_only: Optional[bool] = None,
        skipna: bool = True,
        **kwargs: Any,
    ) -> bool:
        """Return whether all elements are True."""
        if self._series.dtype != pl.Boolean:
            bool_series = self._series.cast(pl.Boolean)
        else:
            bool_series = self._series

        if skipna:
            return bool_series.all()
        else:
            if bool_series.null_count() > 0:
                return False
            return bool_series.all()

    def any(
        self,
        axis: Optional[int] = None,
        bool_only: Optional[bool] = None,
        skipna: bool = True,
        **kwargs: Any,
    ) -> bool:
        """Return whether any element is True."""
        if self._series.dtype != pl.Boolean:
            bool_series = self._series.cast(pl.Boolean)
        else:
            bool_series = self._series

        if skipna:
            return bool_series.any()
        else:
            if bool_series.null_count() > 0:
                return False
            return bool_series.any()

    def is_(self, other: Any) -> "Index":
        """Check object identity."""
        result = self._series == other
        return Index(result)

    def isin(self, values: Union[List[Any], Any]) -> "Index":
        """Whether elements in Index are contained in values."""
        if not isinstance(values, (list, tuple, set)):
            values_list = [values]
        else:
            values_list = list(values)

        result_series = self._series.is_in(values_list)
        return Index(result_series)

    def isna(self) -> "Index":
        """Detect missing values."""
        return Index(self._series.is_null())

    def isnull(self) -> "Index":
        """Alias for isna."""
        return self.isna()

    def notna(self) -> "Index":
        """Detect non-missing values."""
        return Index(self._series.is_not_null())

    def notnull(self) -> "Index":
        """Alias for notna."""
        return self.notna()

    def where(self, cond: Any, other: Any = None, **kwargs: Any) -> "Index":
        """Replace values where condition is False."""
        if isinstance(cond, Index):
            cond = cond._series
        elif not isinstance(cond, pl.Series):
            # Convert scalar or list to Series
            cond = (
                pl.Series("cond", [cond] * len(self._series))
                if len(self._series) > 0
                else pl.Series("cond", [])
            )

        # Use DataFrame to evaluate the expression
        temp_df = pl.DataFrame({"values": self._series, "cond": cond})
        result_df = temp_df.select(
            pl.when(pl.col("cond"))
            .then(pl.col("values"))
            .otherwise(pl.lit(other) if other is not None else None)
            .alias("result")
        )
        result_series = result_df["result"]
        return Index(result_series)

    # Statistical Operations
    def argmax(
        self, axis: Optional[int] = None, skipna: bool = True, **kwargs: Any
    ) -> int:
        """Return int position of the largest value."""
        if skipna:
            max_val = self._series.max()
            if max_val is None:
                return -1
            indices = self._series.arg_max()
            return indices if indices is not None else -1
        else:
            indices = self._series.arg_max()
            return indices if indices is not None else -1

    def argmin(
        self, axis: Optional[int] = None, skipna: bool = True, **kwargs: Any
    ) -> int:
        """Return int position of the smallest value."""
        if skipna:
            min_val = self._series.min()
            if min_val is None:
                return -1
            indices = self._series.arg_min()
            return indices if indices is not None else -1
        else:
            indices = self._series.arg_min()
            return indices if indices is not None else -1

    def argsort(
        self, ascending: bool = True, kind: Optional[str] = None, **kwargs: Any
    ) -> "Index":
        """Return the indices that would sort the Index."""
        # Use arg_sort() method (Polars uses arg_sort, not argsort)
        if ascending:
            result = self._series.arg_sort()
        else:
            result = self._series.arg_sort(descending=True)
        return Index(result)

    def max(
        self, axis: Optional[int] = None, skipna: bool = True, **kwargs: Any
    ) -> Any:
        """Return the maximum of the values."""
        return self._series.max() if skipna else self._series.max()

    def min(
        self, axis: Optional[int] = None, skipna: bool = True, **kwargs: Any
    ) -> Any:
        """Return the minimum of the values."""
        return self._series.min() if skipna else self._series.min()

    def nunique(self, dropna: bool = True, **kwargs: Any) -> int:
        """Return number of unique elements."""
        if dropna:
            return self._series.n_unique()
        else:
            return len(self._series.unique())

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        **kwargs: Any,
    ) -> "Index":
        """Return a Series containing counts of unique values."""
        from polarpandas.series import Series

        result = self._series.value_counts(sort=sort)
        if sort:
            if ascending:
                result = result.sort("count")
            else:
                result = result.sort("count", descending=True)
        if normalize:
            total = len(self._series)
            result = result.with_columns([pl.col("count") / total])
        return Series(result)  # type: ignore[arg-type]

    # Data Manipulation
    def append(self, other: Union["Index", List[Any], pl.Series]) -> "Index":
        """Append another Index."""
        if isinstance(other, Index):
            other_series = other._series
        elif isinstance(other, pl.Series):
            other_series = other
        else:
            other_series = pl.Series("index", other)
        result = pl.concat([self._series, other_series])
        return Index(result)

    def astype(self, dtype: Any, errors: str = "raise", **kwargs: Any) -> "Index":
        """Cast Index to dtype."""
        try:
            result = self._series.cast(dtype)
            return Index(result)
        except Exception:
            if errors == "raise":
                raise
            return self

    def copy(self, deep: bool = True, **kwargs: Any) -> "Index":
        """Make a copy of the Index."""
        return Index(self._series.clone())

    def delete(self, loc: Union[int, List[int]]) -> "Index":
        """Delete locations."""
        if isinstance(loc, int):
            loc = [loc]
        # Convert to list and remove indices
        values = self._series.to_list()
        # Sort indices in descending order to avoid index shifting
        for idx in sorted(loc, reverse=True):
            if 0 <= idx < len(values):
                del values[idx]
        return Index(values)

    def diff(self, periods: int = 1, **kwargs: Any) -> "Index":
        """Calculate the first discrete difference."""
        result = self._series.diff(periods)
        return Index(result)

    def drop(
        self, labels: Union[Any, List[Any]], errors: str = "raise", **kwargs: Any
    ) -> "Index":
        """Drop labels."""
        if not isinstance(labels, (list, tuple)):
            labels = [labels]

        # Filter out labels
        mask = ~self._series.is_in(labels)
        result = self._series.filter(mask)

        if errors == "raise" and len(result) == len(self._series):
            # Check if any labels were actually dropped
            found = any(label in self._series.to_list() for label in labels)
            if not found:
                raise KeyError(f"labels {labels} not found in index")

        return Index(result)

    def drop_duplicates(
        self, keep: Union[str, bool] = "first", **kwargs: Any
    ) -> "Index":
        """Drop duplicate values."""
        if keep == "first":
            result = self._series.unique(maintain_order=True)
        elif keep == "last":
            # Reverse, get unique, reverse back
            reversed_series = self._series.reverse()
            result = reversed_series.unique(maintain_order=True).reverse()
        else:  # keep == False
            # Keep only values that appear once
            value_counts = self._series.value_counts()
            single_occurrence = value_counts.filter(pl.col("count") == 1)["index"]
            result = self._series.filter(self._series.is_in(single_occurrence))
        return Index(result)

    def dropna(self, **kwargs: Any) -> "Index":
        """Drop NA values."""
        result = self._series.filter(self._series.is_not_null())
        return Index(result)

    def fillna(
        self, value: Optional[Any] = None, method: Optional[str] = None, **kwargs: Any
    ) -> "Index":
        """Fill NA values."""
        if method is not None:
            raise NotImplementedError(
                "method parameter not supported, use value instead"
            )
        if value is None:
            return self
        result = self._series.fill_null(value)
        return Index(result)

    def insert(self, loc: int, item: Any) -> "Index":
        """Insert item at location."""
        values = self._series.to_list()
        values.insert(loc, item)
        return Index(values)

    def map(self, arg: Union[Dict[Any, Any], Callable[..., Any], "Index"]) -> "Index":
        """Map values using a dictionary or function."""
        if isinstance(arg, dict):

            def map_func(x: Any) -> Any:
                return arg.get(x, x)

            result = self._series.map_elements(map_func, return_dtype=pl.Object)
        elif callable(arg):
            result = self._series.map_elements(arg, return_dtype=pl.Object)
        elif isinstance(arg, Index):
            # Map using another Index (positional mapping)
            arg_list = arg._series.to_list()

            def map_func(x: Any, idx: int) -> Any:
                if idx < len(arg_list):
                    return arg_list[idx]
                return x

            result = self._series.map_elements(
                lambda x, idx: map_func(x, idx), return_dtype=pl.Object
            )
        else:
            raise TypeError(
                f"map() arg must be dict, callable, or Index, got {type(arg)}"
            )
        return Index(result)

    def putmask(
        self, mask: Union["Index", pl.Series, List[bool]], value: Any
    ) -> "Index":
        """Set values where mask is True."""
        if isinstance(mask, Index):
            mask_series = mask._series
        elif isinstance(mask, list):
            mask_series = pl.Series(mask)
        else:
            mask_series = mask

        # Use DataFrame to evaluate the expression
        temp_df = pl.DataFrame({"values": self._series, "mask": mask_series})
        result_df = temp_df.select(
            pl.when(pl.col("mask"))
            .then(pl.lit(value))
            .otherwise(pl.col("values"))
            .alias("result")
        )
        result_series = result_df["result"]
        return Index(result_series)

    def ravel(self, order: Optional[str] = None, **kwargs: Any) -> "Index":
        """Return flattened view (no-op for 1D Index)."""
        return self

    def repeat(self, repeats: Union[int, List[int]], **kwargs: Any) -> "Index":
        """Repeat elements."""
        if isinstance(repeats, int):
            # Use explode with list comprehension
            values = self._series.to_list()
            result = pl.Series([v for v in values for _ in range(repeats)])
        else:
            # Repeat each element by corresponding value in repeats
            values = self._series.to_list()
            result = pl.Series(
                [v for i, v in enumerate(values) for _ in range(repeats[i])]
            )
        return Index(result)

    def round(self, decimals: int = 0, **kwargs: Any) -> "Index":
        """Round each value to the given number of decimals."""
        result = self._series.round(decimals)
        return Index(result)

    def shift(
        self, periods: int = 1, fill_value: Optional[Any] = None, **kwargs: Any
    ) -> "Index":
        """Shift values by periods."""
        result = self._series.shift(periods)
        if fill_value is not None:
            result = result.fill_null(fill_value)
        return Index(result)

    def take(self, indices: Union[List[int], "Index"], **kwargs: Any) -> "Index":
        """Take elements by position."""
        if isinstance(indices, Index):
            indices_list = indices._series.to_list()
        else:
            indices_list = indices
        result = self._series[indices_list]
        return Index(result)

    def transpose(self, *args: Any, **kwargs: Any) -> "Index":
        """Transpose (no-op for 1D Index)."""
        return self

    # Set Operations
    def difference(
        self, other: Union["Index", List[Any], pl.Series], sort: Optional[bool] = None
    ) -> "Index":
        """Return elements in self but not in other."""
        if isinstance(other, Index):
            other_series = other._series
        elif isinstance(other, pl.Series):
            other_series = other
        else:
            other_series = pl.Series("index", other)

        mask = ~self._series.is_in(other_series)
        result = self._series.filter(mask)

        if sort is True:
            result = result.sort()
        elif sort is False:
            # Maintain order
            pass

        return Index(result)

    def intersection(
        self, other: Union["Index", List[Any], pl.Series], sort: Optional[bool] = None
    ) -> "Index":
        """Return elements in both self and other."""
        if isinstance(other, Index):
            other_series = other._series
        elif isinstance(other, pl.Series):
            other_series = other
        else:
            other_series = pl.Series("index", other)

        mask = self._series.is_in(other_series)
        result = self._series.filter(mask)

        if sort is True:
            result = result.sort()
        elif sort is False:
            # Maintain order
            pass

        return Index(result)

    def symmetric_difference(
        self,
        other: Union["Index", List[Any], pl.Series],
        result_name: Optional[str] = None,
        sort: Optional[bool] = None,
    ) -> "Index":
        """Return elements in either self or other, but not both."""
        if isinstance(other, Index):
            other_series = other._series
        elif isinstance(other, pl.Series):
            other_series = other
        else:
            other_series = pl.Series("index", other)

        # Elements in self but not other
        self_only = self._series.filter(~self._series.is_in(other_series))
        # Elements in other but not self
        other_only = other_series.filter(~other_series.is_in(self._series))

        # Combine
        result = pl.concat([self_only, other_only])

        if sort is True:
            result = result.sort()
        elif sort is False:
            # Maintain order
            pass

        return Index(result)

    def union(
        self, other: Union["Index", List[Any], pl.Series], sort: Optional[bool] = None
    ) -> "Index":
        """Return elements in either self or other."""
        if isinstance(other, Index):
            other_series = other._series
        elif isinstance(other, pl.Series):
            other_series = other
        else:
            other_series = pl.Series("index", other)

        result = pl.concat([self._series, other_series]).unique(maintain_order=True)

        if sort is True:
            result = result.sort()
        elif sort is False:
            # Maintain order
            pass

        return Index(result)

    def unique(self, **kwargs: Any) -> "Index":
        """Return unique values."""
        result = self._series.unique(maintain_order=True)
        return Index(result)

    # Indexing/Location Operations
    def asof(self, label: Any) -> Any:
        """Return last valid index up to label."""
        # Find the last value <= label
        mask = self._series <= label
        filtered = self._series.filter(mask)
        if len(filtered) == 0:
            return None
        return filtered[-1]

    def asof_locs(
        self,
        where: Union["Index", List[Any], pl.Series],
        mask: Optional["Index"] = None,
    ) -> "Index":
        """Return locations for asof."""
        if isinstance(where, Index):
            where_series = where._series
        elif isinstance(where, pl.Series):
            where_series = where
        else:
            where_series = pl.Series("index", where)

        def find_asof_loc(w: Any) -> int:
            mask = self._series <= w
            filtered = self._series.filter(mask)
            if len(filtered) == 0:
                return -1
            # Find index of last matching value
            last_val = filtered[-1]
            indices = [i for i, v in enumerate(self._series.to_list()) if v == last_val]
            return indices[-1] if indices else -1

        result = where_series.map_elements(find_asof_loc, return_dtype=pl.Int64)
        return Index(result)

    def get_indexer(
        self,
        target: Union["Index", List[Any], pl.Series],
        method: Optional[str] = None,
        limit: Optional[int] = None,
        tolerance: Optional[Any] = None,
    ) -> List[int]:
        """Compute indexer for target."""
        if isinstance(target, Index):
            target_series = target._series
        elif isinstance(target, pl.Series):
            target_series = target
        else:
            target_series = pl.Series("index", target)

        # Create mapping from value to index
        value_to_index = {val: i for i, val in enumerate(self._series.to_list())}

        def get_idx(t: Any) -> int:
            return value_to_index.get(t, -1)

        result = target_series.map_elements(get_idx, return_dtype=pl.Int64)
        return result.to_list()

    def get_indexer_for(
        self, target: Union["Index", List[Any], pl.Series], **kwargs: Any
    ) -> List[int]:
        """Compute indexer for target (deprecated, use get_indexer)."""
        return self.get_indexer(target, **kwargs)

    def get_indexer_non_unique(
        self, target: Union["Index", List[Any], pl.Series]
    ) -> Tuple[List[int], List[int]]:
        """Compute indexer for target (non-unique)."""
        if isinstance(target, Index):
            target_series = target._series
        elif isinstance(target, pl.Series):
            target_series = target
        else:
            target_series = pl.Series("index", target)

        # Create mapping from value to list of indices
        value_to_indices: Dict[Any, List[int]] = {}
        for i, val in enumerate(self._series.to_list()):
            if val not in value_to_indices:
                value_to_indices[val] = []
            value_to_indices[val].append(i)

        indexer = []
        missing = []
        for i, t in enumerate(target_series.to_list()):
            if t in value_to_indices:
                # Use first occurrence
                indexer.append(value_to_indices[t][0])
            else:
                indexer.append(-1)
                missing.append(i)

        return indexer, missing

    def get_level_values(self, level: Union[int, str]) -> "Index":
        """Get level values (for MultiIndex)."""
        # For regular Index, just return self
        # MultiIndex would override this
        return self

    def get_loc(
        self, key: Any, method: Optional[str] = None, tolerance: Optional[Any] = None
    ) -> Union[int, slice, List[int]]:
        """Get location for label."""
        values = self._series.to_list()

        if method is None:
            # Exact match
            try:
                idx = values.index(key)
                return idx
            except ValueError:
                raise KeyError(f"{key} not in index") from None
        elif method == "pad" or method == "ffill":
            # Forward fill
            mask = self._series <= key
            filtered = self._series.filter(mask)
            if len(filtered) == 0:
                raise KeyError(f"{key} not in index")
            last_val = filtered[-1]
            indices = [i for i, v in enumerate(values) if v == last_val]
            return indices[-1] if indices else -1
        elif method == "backfill" or method == "bfill":
            # Backward fill
            mask = self._series >= key
            filtered = self._series.filter(mask)
            if len(filtered) == 0:
                raise KeyError(f"{key} not in index")
            first_val = filtered[0]
            indices = [i for i, v in enumerate(values) if v == first_val]
            return indices[0] if indices else -1
        elif method == "nearest":
            # Nearest
            distances = [
                (
                    abs(v - key)
                    if isinstance(v, (int, float)) and isinstance(key, (int, float))
                    else float("inf"),
                    i,
                )
                for i, v in enumerate(values)
            ]
            distances.sort()
            return distances[0][1]
        else:
            raise ValueError(
                f"method must be None, 'pad', 'backfill', or 'nearest', got {method}"
            )

    def get_slice_bound(self, label: Any, side: str, kind: Optional[str] = None) -> int:
        """Get slice bound for label."""
        values = self._series.to_list()

        if side == "left":
            # Find first index >= label
            for i, v in enumerate(values):
                if v >= label:
                    return i
            return len(values)
        elif side == "right":
            # Find first index > label
            for i, v in enumerate(values):
                if v > label:
                    return i
            return len(values)
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side}")

    def searchsorted(
        self, value: Any, side: str = "left", sorter: Optional[List[int]] = None
    ) -> Union[int, List[int]]:
        """Find insertion points to maintain sorted order."""
        if sorter is not None:
            # Sort by sorter
            sorted_values = [self._series.to_list()[i] for i in sorter]
        else:
            sorted_values = sorted(self._series.to_list())

        if not isinstance(value, (list, tuple)):
            value = [value]

        result = []
        for v in value:
            if side == "left":
                # Find first index where sorted_values[i] >= v
                idx = 0
                while idx < len(sorted_values) and sorted_values[idx] < v:
                    idx += 1
                result.append(idx)
            else:  # side == "right"
                # Find first index where sorted_values[i] > v
                idx = 0
                while idx < len(sorted_values) and sorted_values[idx] <= v:
                    idx += 1
                result.append(idx)

        return result[0] if len(result) == 1 else result

    def slice_indexer(
        self,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        step: Optional[int] = None,
        kind: Optional[str] = None,
    ) -> slice:
        """Compute slice indexer."""
        start_idx = self.get_slice_bound(start, "left") if start is not None else 0
        end_idx = (
            self.get_slice_bound(end, "right") if end is not None else len(self._series)
        )
        # For slice, end should be exclusive, so if we found the first index > end, that's correct
        # But get_slice_bound with "right" already returns the first index > end, which is what we want
        return slice(start_idx, end_idx, step)

    def slice_locs(
        self,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        step: Optional[int] = None,
        kind: Optional[str] = None,
    ) -> Tuple[int, int]:
        """Compute slice locations."""
        start_idx = self.get_slice_bound(start, "left") if start is not None else 0
        end_idx = (
            self.get_slice_bound(end, "right") if end is not None else len(self._series)
        )
        return (start_idx, end_idx)

    # MultiIndex Operations
    def droplevel(
        self, level: Union[int, str, List[Union[int, str]]], **kwargs: Any
    ) -> "Index":
        """Drop level from MultiIndex."""
        # For regular Index, just return self
        # MultiIndex would override this
        return self

    def set_names(
        self,
        names: Union[str, List[str]],
        level: Optional[Union[int, str, List[Union[int, str]]]] = None,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Optional["Index"]:
        """Set names."""
        if inplace:
            # Index is immutable, so we can't modify in place
            # But we can update the series name
            if isinstance(names, str):
                self._series = self._series.rename(names)
            return None
        else:
            result = self._series.clone()
            if isinstance(names, str):
                result = result.rename(names)
            return Index(result)

    def sortlevel(
        self,
        level: Optional[Union[int, str, List[Union[int, str]]]] = None,
        ascending: bool = True,
        sort_remaining: bool = True,
        **kwargs: Any,
    ) -> Tuple["Index", List[int]]:
        """Sort by level."""
        # For regular Index, just sort
        sorted_series = (
            self._series.sort() if ascending else self._series.sort(descending=True)
        )
        sorted_indices = sorted(
            range(len(self._series)),
            key=lambda i: self._series[i],
            reverse=not ascending,
        )
        return Index(sorted_series), sorted_indices

    def to_flat_index(self) -> "Index":
        """Convert to flat Index."""
        # For regular Index, just return self
        return self

    # Type Conversion/Export
    def to_frame(self, name: Optional[str] = None) -> "DataFrame":  # type: ignore[name-defined]
        """Convert to DataFrame."""
        from polarpandas.frame import DataFrame

        col_name = name if name is not None else (self._series.name or "index")
        return DataFrame({col_name: self._series.to_list()})

    def to_list(self) -> List[Any]:
        """Convert to list (alias for tolist)."""
        return self.tolist()

    def to_numpy(
        self,
        dtype: Optional[Any] = None,
        copy: bool = False,
        na_value: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Convert to numpy array."""
        import numpy as np

        arr = self._series.to_numpy()
        if dtype is not None:
            arr = arr.astype(dtype)
        if na_value is not None:
            arr = np.where(arr is None, na_value, arr)
        return arr

    def to_series(
        self, index: Optional[Any] = None, name: Optional[str] = None
    ) -> "Series":  # type: ignore[name-defined]
        """Convert to Series."""
        from polarpandas.series import Series

        return Series(self._series.clone(), name=name or self._series.name, index=index)

    def view(self, dtype: Optional[Any] = None) -> "Index":
        """Return view."""
        # For Index, view is same as copy
        return self.copy()

    # Comparison/Equality
    def duplicated(self, keep: Union[str, bool] = "first") -> "Index":
        """Check for duplicates."""
        if keep == "first":
            # Mark all duplicates except first
            seen = set()
            result = []
            for val in self._series.to_list():
                if val in seen:
                    result.append(True)
                else:
                    seen.add(val)
                    result.append(False)
        elif keep == "last":
            # Mark all duplicates except last
            seen = set()
            result = []
            for val in reversed(self._series.to_list()):
                if val in seen:
                    result.append(True)
                else:
                    seen.add(val)
                    result.append(False)
            result.reverse()
        else:  # keep == False
            # Mark all duplicates
            from collections import Counter

            counts = Counter(self._series.to_list())
            result = [counts[val] > 1 for val in self._series.to_list()]

        return Index(result)

    def equals(self, other: Any) -> bool:
        """Check equality."""
        if not isinstance(other, Index):
            return False
        return self._series.equals(other._series)

    def identical(self, other: Any) -> bool:
        """Check identity."""
        if not isinstance(other, Index):
            return False
        # Check if same object or same underlying series
        return self is other or (self._series is other._series)

    # Other Operations
    def factorize(
        self,
        sort: bool = False,
        na_sentinel: Optional[int] = None,
        use_na_sentinel: Optional[bool] = None,
        **kwargs: Any,
    ) -> Tuple[List[int], "Index"]:
        """Encode as enumerated type."""
        if use_na_sentinel is False:
            na_sentinel = None

        unique_vals = self._series.unique(maintain_order=True)
        if sort:
            unique_vals = unique_vals.sort()

        # Create mapping
        val_to_code = {val: i for i, val in enumerate(unique_vals.to_list())}

        codes = []
        for val in self._series.to_list():
            if val is None:
                codes.append(na_sentinel if na_sentinel is not None else -1)
            else:
                codes.append(val_to_code[val])

        uniques = Index(unique_vals)
        return codes, uniques

    def groupby(self, **kwargs: Any) -> Any:
        """Group by values."""
        # Index groupby is not fully implemented yet
        # For now, raise NotImplementedError with helpful message
        raise NotImplementedError(
            "groupby() for Index is not yet implemented. Use DataFrame.groupby() instead."
        )

    def infer_objects(self, copy: bool = True, **kwargs: Any) -> "Index":
        """Infer better dtypes."""
        # Try to infer better dtype
        # This is a simplified version
        return self.copy() if copy else self

    def item(self) -> Any:
        """Return single element."""
        if len(self._series) != 1:
            raise ValueError("can only convert an array of size 1 to a Python scalar")
        return self._series[0]

    def join(
        self,
        other: Union["Index", List[Any], pl.Series],
        how: str = "left",
        level: Optional[Any] = None,
        return_indexers: bool = False,
        sort: bool = False,
        **kwargs: Any,
    ) -> Union["Index", Tuple["Index", Optional[List[int]], Optional[List[int]]]]:
        """Join with another Index."""
        if isinstance(other, Index):
            other_series = other._series
        elif isinstance(other, pl.Series):
            other_series = other
        else:
            other_series = pl.Series("index", other)

        if how == "left":
            result = self._series
        elif how == "right":
            result = other_series
        elif how == "inner":
            mask = self._series.is_in(other_series)
            result = self._series.filter(mask)
        elif how == "outer":
            result = pl.concat([self._series, other_series]).unique(maintain_order=True)
        else:
            raise ValueError(
                f"how must be 'left', 'right', 'inner', or 'outer', got {how}"
            )

        if sort:
            result = result.sort()

        if return_indexers:
            left_indexer = self.get_indexer(result)
            right_indexer = Index(result).get_indexer(other_series)
            return Index(result), left_indexer, right_indexer

        return Index(result)

    def memory_usage(self, deep: bool = False, **kwargs: Any) -> int:
        """Memory usage in bytes."""
        if deep:
            return sys.getsizeof(self._series.to_list())
        else:
            return sys.getsizeof(self._series)

    def reindex(
        self,
        target: Union["Index", List[Any], pl.Series],
        method: Optional[str] = None,
        level: Optional[Any] = None,
        limit: Optional[int] = None,
        tolerance: Optional[Any] = None,
        fill_value: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple["Index", Optional[List[int]]]:
        """Reindex to new labels."""
        if isinstance(target, Index):
            target_series = target._series
        elif isinstance(target, pl.Series):
            target_series = target
        else:
            target_series = pl.Series("index", target)

        # Get indexer
        indexer = self.get_indexer(
            target_series, method=method, limit=limit, tolerance=tolerance
        )

        # Create result with fill_value for missing
        result_values = []
        for _i, idx in enumerate(indexer):
            if idx == -1:
                result_values.append(fill_value)
            else:
                result_values.append(self._series[idx])

        result = Index(result_values)
        return result, indexer

    def rename(
        self, name: Optional[Union[str, List[str]]] = None, **kwargs: Any
    ) -> "Index":
        """Rename Index."""
        if name is None:
            return self.copy()

        result = self._series.clone()
        if isinstance(name, str):
            result = result.rename(name)
        return Index(result)

    def sort_values(
        self,
        return_indexer: bool = False,
        ascending: bool = True,
        na_position: str = "last",
        key: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Union["Index", Tuple["Index", "Index"]]:
        """Sort by values."""
        if key is not None:
            # Apply key function
            mapped = self._series.map_elements(key, return_dtype=pl.Object)
            sorted_indices = sorted(
                range(len(mapped)), key=lambda i: mapped[i], reverse=not ascending
            )
        else:
            sorted_indices = sorted(
                range(len(self._series)),
                key=lambda i: self._series[i],
                reverse=not ascending,
            )

        sorted_values = [self._series[i] for i in sorted_indices]
        result = Index(sorted_values)

        if return_indexer:
            indexer = Index(sorted_indices)
            return result, indexer

        return result

    @property
    def str(self) -> Any:
        """String accessor property."""
        from polarpandas.series import Series

        return Series(self._series).str
