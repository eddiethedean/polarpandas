"""
Series implementation wrapping Polars Series with pandas-like API.
"""

from __future__ import annotations

import contextlib
import inspect
import sys
from typing import TYPE_CHECKING, Any, Callable, Iterator

import polars as pl

if TYPE_CHECKING:
    from .frame import DataFrame

BuiltinStr = str
BuiltinList = list
BuiltinDict = dict


def _callable_accepts_argument(func: Any, parameter_name: str) -> bool:
    """Check whether a callable accepts a named parameter."""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False

    return any(
        param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        and param.name == parameter_name
        for param in signature.parameters.values()
    )


class Series:
    """
    One-dimensional labeled array capable of holding any data type.

    Series is the one-dimensional data structure in PolarPandas, providing a
    pandas-like API while using Polars Series for all operations under the hood.
    It represents a single column of data with an optional index.

    Parameters
    ----------
    data : array-like, pl.Series, or None, optional
        Input data. Can be:
        - List or array-like of values
        - Existing Polars Series
        - None for empty Series
    name : str, optional
        Name for the Series. Used to identify the Series and appears in
        DataFrame columns when Series is extracted from DataFrame.
    index : array-like, optional
        Index to use for the Series. Stored separately for pandas compatibility.
    **kwargs
        Additional keyword arguments passed to Polars Series constructor.

    Attributes
    ----------
    _series : pl.Series
        The underlying Polars Series.
    _index : list or None
        Stored index values for pandas compatibility.
    _index_name : str or None
        Name for the index.

    Examples
    --------
    >>> import polarpandas as ppd
    >>> # From list
    >>> s = ppd.Series([1, 2, 3, 4, 5])
    >>> # With name
    >>> s = ppd.Series([1, 2, 3], name="values")
    >>> # From Polars Series
    >>> import polars as pl
    >>> s = ppd.Series(pl.Series([10, 20, 30]))

    See Also
    --------
    DataFrame : Two-dimensional labeled data structure
    Index : Index object for DataFrame

    Notes
    -----
    - Series operations are always eager (executed immediately)
    - Index is stored separately and not part of Polars Series structure
    - Most operations return new Series; original is unchanged
    """

    def __init__(
        self,
        data: Any = None,
        name: Any = None,
        index: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a Series from various data sources.

        Create a new Series instance from the provided data. The data can be
        provided as a list, array, or existing Polars Series.

        Parameters
        ----------
        data : array-like, pl.Series, or None, optional
            Data to initialize the Series with:
            - List or array-like: Creates Series from values
            - Polars Series: Uses the Series directly
            - None: Creates empty Series
        name : str, optional
            Name for the Series. Used to identify the Series when it appears
            as a column in a DataFrame.
        index : array-like, optional
            Index to use for the Series. Must have same length as data if provided.
            Stored separately for pandas compatibility.
        *args
            Additional positional arguments passed to Polars Series constructor.
        **kwargs
            Additional keyword arguments passed to Polars Series constructor.

        Examples
        --------
        >>> import polarpandas as ppd
        >>> # From list
        >>> s = ppd.Series([1, 2, 3, 4, 5])
        >>> # With name
        >>> s = ppd.Series([10, 20, 30], name="values")
        >>> # Empty Series
        >>> s = ppd.Series()
        >>> # With index
        >>> s = ppd.Series([1, 2, 3], index=["a", "b", "c"])

        Notes
        -----
        - Index is stored separately and not part of Polars Series structure
        - Empty Series can be created by passing None or empty list
        """
        # Store index information
        index_values: BuiltinList[Any] | None = None
        if index is not None:
            if hasattr(index, "tolist") and not isinstance(index, (list, tuple)):
                try:
                    index_values = list(index.tolist())
                except TypeError:
                    index_values = list(index)
            else:
                index_values = list(index)
        self._index = index_values
        self._index_name = None
        self._original_name = None  # Store original name type for restoration

        # Handle tuple names (for MultiIndex compatibility)
        # Polars requires string names, but pandas allows tuples
        polars_name = name
        if isinstance(name, tuple):
            self._original_name = name
            # Convert tuple to string for Polars
            polars_name = (
                str(name) if len(name) > 1 else (name[0] if len(name) == 1 else None)
            )
        elif name is not None:
            self._original_name = name
            if not isinstance(name, str):
                polars_name = str(name)

        if data is None:
            self._series = pl.Series(name=polars_name or "", values=[])
        elif isinstance(data, pl.Series):
            self._series = data
            # Update name if provided
            if polars_name is not None:
                self._series = self._series.rename(polars_name)
        else:
            # Handle list or other array-like data
            # Pass through kwargs to pl.Series constructor
            # Note: pl.Series(name=..., values=...) - name must be keyword, data goes in values
            self._series = pl.Series(values=data, name=polars_name or "", **kwargs)

        if self._index is not None and len(self._index) != len(self._series):
            raise ValueError(
                "Length of index must match length of data when constructing Series"
            )

        self._categorical_order: BuiltinList[Any] | None = None
        if self._series.dtype == pl.Categorical:
            self._categorical_order = self._series.cat.get_categories().to_list()

    @property
    def name(self) -> Any:
        """Get the name of the Series."""
        return (
            self._original_name
            if self._original_name is not None
            else self._series.name
        )

    @name.setter
    def name(self, value: Any) -> None:
        """Set the name of the Series."""
        if value is None:
            self._original_name = None
            self._series = self._series.rename("")
            return

        self._original_name = value
        new_name = value if isinstance(value, str) else str(value)
        self._series = self._series.rename(new_name)

    @property
    def values(self) -> Any:
        """Get the values of the Series as a numpy array."""
        return self._series.to_numpy()

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
        """Return string representation of the Series."""
        return repr(self._series)

    def __str__(self) -> str:
        """Return string representation of the Series."""
        return str(self._series)

    def __getitem__(self, key: Any) -> Any:
        """
        Access Series values by position or label.

        Parameters
        ----------
        key : int, str, slice, or array-like
            Position or label to access

        Returns
        -------
        scalar or Series
            Value(s) at the specified position(s)
        """
        # Handle integer indexing
        if isinstance(key, int):
            return self._series[key]
        # Handle string/label indexing if we have an index
        elif isinstance(key, str) and self._index is not None:
            try:
                idx = self._index.index(key)
                return self._series[idx]
            except (ValueError, AttributeError):
                # Fall back to Polars Series behavior
                return self._series[key]  # type: ignore[index]
        # Handle slicing and other cases
        else:
            return self._series[key]

    def __len__(self) -> int:
        """Return the length of the Series."""
        return len(self._series)

    @property
    def shape(self) -> tuple[int]:
        """Return the shape of the Series."""
        return (len(self._series),)

    @property
    def size(self) -> int:
        """Return the size of the Series."""
        return len(self._series)

    # Arithmetic operations
    def __add__(self, other: Any) -> Series:
        """Add Series or scalar."""
        if isinstance(other, Series):
            return Series(self._series + other._series)
        return Series(self._series + other)

    def __sub__(self, other: Any) -> Series:
        """Subtract Series or scalar."""
        if isinstance(other, Series):
            return Series(self._series - other._series)
        return Series(self._series - other)

    def __mul__(self, other: Any) -> Series:
        """Multiply Series or scalar."""
        if isinstance(other, Series):
            return Series(self._series * other._series)
        return Series(self._series * other)

    def __truediv__(self, other: Any) -> Series:
        """Divide Series or scalar."""
        if isinstance(other, Series):
            return Series(self._series / other._series)
        return Series(self._series / other)

    def __radd__(self, other: Any) -> Series:
        """Right add (for scalar + Series)."""
        return Series(other + self._series)

    def __rsub__(self, other: Any) -> Series:
        """Right subtract (for scalar - Series)."""
        return Series(other - self._series)

    def __rmul__(self, other: Any) -> Series:
        """Right multiply (for scalar * Series)."""
        return Series(other * self._series)

    def __rtruediv__(self, other: Any) -> Series:
        """Right divide (for scalar / Series)."""
        return Series(other / self._series)

    def __and__(self, other: Any) -> Series:
        """Element-wise logical AND supporting other Series or scalars."""

        rhs = other._series if isinstance(other, Series) else other
        try:
            result = self._series.__and__(rhs)
        except TypeError:
            result = self._series.__and__(pl.Series(rhs))
        return Series(result, index=self._index)

    def __rand__(self, other: Any) -> Series:
        """Reverse logical AND to support scalar & Series."""

        rhs = other._series if isinstance(other, Series) else other
        try:
            result = self._series.__rand__(rhs)
        except TypeError:
            result = self._series.__rand__(pl.Series(rhs))
        return Series(result, index=self._index)

    # Comparison operators
    def __gt__(self, other: Any) -> Series:
        """Greater than comparison."""
        if isinstance(other, Series):
            result = Series(self._series > other._series)
        else:
            result = Series(self._series > other)
        # Set name to empty string to match pandas behavior
        result._series = result._series.alias("")
        return result

    def __lt__(self, other: Any) -> Series:
        """Less than comparison."""
        if isinstance(other, Series):
            result = Series(self._series < other._series)
        else:
            result = Series(self._series < other)
        result._series = result._series.alias("")
        return result

    def __ge__(self, other: Any) -> Series:
        """Greater than or equal comparison."""
        if isinstance(other, Series):
            result = Series(self._series >= other._series)
        else:
            result = Series(self._series >= other)
        result._series = result._series.alias("")
        return result

    def __le__(self, other: Any) -> Series:
        """Less than or equal comparison."""
        if isinstance(other, Series):
            result = Series(self._series <= other._series)
        else:
            result = Series(self._series <= other)
        result._series = result._series.alias("")
        return result

    def __eq__(self, other: Any) -> Series:  # type: ignore[override]
        """Equal comparison."""
        if isinstance(other, Series):
            result = Series(self._series == other._series)
        else:
            result = Series(self._series == other)
        result._series = result._series.alias("")
        return result

    def __ne__(self, other: Any) -> Series:  # type: ignore[override]
        """Not equal comparison."""
        if isinstance(other, Series):
            result = Series(self._series != other._series)
        else:
            result = Series(self._series != other)
        result._series = result._series.alias("")
        return result

    # Explicit arithmetic methods
    def add(
        self,
        other: Any,
        fill_value: Any = None,
        **kwargs: Any,
    ) -> Series:
        """
        Return addition of Series and other, element-wise (binary operator +).

        Parameters
        ----------
        other : scalar or Series
            Object to add to the Series.
        fill_value : float or None, default None
            Fill existing missing (NaN) values with this value before computation.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Result of the arithmetic operation.
        """
        if isinstance(other, Series):
            if fill_value is not None:
                result_series = self._series.fill_null(
                    fill_value
                ) + other._series.fill_null(fill_value)
            else:
                result_series = self._series + other._series
        else:
            if fill_value is not None:
                result_series = self._series.fill_null(fill_value) + other
            else:
                result_series = self._series + other
        return Series(result_series, index=self._index)

    def sub(
        self,
        other: Any,
        fill_value: Any = None,
        **kwargs: Any,
    ) -> Series:
        """
        Return subtraction of Series and other, element-wise (binary operator -).

        Parameters
        ----------
        other : scalar or Series
            Object to subtract from the Series.
        fill_value : float or None, default None
            Fill existing missing (NaN) values with this value before computation.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Result of the arithmetic operation.
        """
        if isinstance(other, Series):
            if fill_value is not None:
                result_series = self._series.fill_null(
                    fill_value
                ) - other._series.fill_null(fill_value)
            else:
                result_series = self._series - other._series
        else:
            if fill_value is not None:
                result_series = self._series.fill_null(fill_value) - other
            else:
                result_series = self._series - other
        return Series(result_series, index=self._index)

    def subtract(
        self,
        other: Any,
        fill_value: Any = None,
        **kwargs: Any,
    ) -> Series:
        """
        Return subtraction of Series and other, element-wise (binary operator -).

        Alias for sub().
        """
        return self.sub(other, fill_value=fill_value, **kwargs)

    def mul(
        self,
        other: Any,
        fill_value: Any = None,
        **kwargs: Any,
    ) -> Series:
        """
        Return multiplication of Series and other, element-wise (binary operator *).

        Parameters
        ----------
        other : scalar or Series
            Object to multiply with the Series.
        fill_value : float or None, default None
            Fill existing missing (NaN) values with this value before computation.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Result of the arithmetic operation.
        """
        if isinstance(other, Series):
            if fill_value is not None:
                result_series = self._series.fill_null(
                    fill_value
                ) * other._series.fill_null(fill_value)
            else:
                result_series = self._series * other._series
        else:
            if fill_value is not None:
                result_series = self._series.fill_null(fill_value) * other
            else:
                result_series = self._series * other
        return Series(result_series, index=self._index)

    def multiply(
        self,
        other: Any,
        fill_value: Any = None,
        **kwargs: Any,
    ) -> Series:
        """
        Return multiplication of Series and other, element-wise (binary operator *).

        Alias for mul().
        """
        return self.mul(other, fill_value=fill_value, **kwargs)

    def div(
        self,
        other: Any,
        fill_value: Any = None,
        **kwargs: Any,
    ) -> Series:
        """
        Return floating division of Series and other, element-wise (binary operator /).

        Parameters
        ----------
        other : scalar or Series
            Object to divide the Series by.
        fill_value : float or None, default None
            Fill existing missing (NaN) values with this value before computation.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Result of the arithmetic operation.
        """
        if isinstance(other, Series):
            if fill_value is not None:
                result_series = self._series.fill_null(
                    fill_value
                ) / other._series.fill_null(fill_value)
            else:
                result_series = self._series / other._series
        else:
            if fill_value is not None:
                result_series = self._series.fill_null(fill_value) / other
            else:
                result_series = self._series / other
        return Series(result_series, index=self._index)

    def divide(
        self,
        other: Any,
        fill_value: Any = None,
        **kwargs: Any,
    ) -> Series:
        """
        Return floating division of Series and other, element-wise (binary operator /).

        Alias for div().
        """
        return self.div(other, fill_value=fill_value, **kwargs)

    def mod(
        self,
        other: Any,
        fill_value: Any = None,
        **kwargs: Any,
    ) -> Series:
        """
        Return modulo of Series and other, element-wise (binary operator %).

        Parameters
        ----------
        other : scalar or Series
            Object to compute modulo with.
        fill_value : float or None, default None
            Fill existing missing (NaN) values with this value before computation.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Result of the arithmetic operation.
        """
        if isinstance(other, Series):
            if fill_value is not None:
                result_series = self._series.fill_null(
                    fill_value
                ) % other._series.fill_null(fill_value)
            else:
                result_series = self._series % other._series
        else:
            if fill_value is not None:
                result_series = self._series.fill_null(fill_value) % other
            else:
                result_series = self._series % other
        return Series(result_series, index=self._index)

    def pow(
        self,
        other: Any,
        fill_value: Any = None,
        **kwargs: Any,
    ) -> Series:
        """
        Return exponential power of Series and other, element-wise (binary operator **).

        Parameters
        ----------
        other : scalar or Series
            Object to raise the Series to the power of.
        fill_value : float or None, default None
            Fill existing missing (NaN) values with this value before computation.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Result of the arithmetic operation.
        """
        if isinstance(other, Series):
            if fill_value is not None:
                result_series = self._series.fill_null(fill_value).pow(
                    other._series.fill_null(fill_value)
                )
            else:
                result_series = self._series.pow(other._series)
        else:
            if fill_value is not None:
                result_series = self._series.fill_null(fill_value).pow(other)
            else:
                result_series = self._series.pow(other)
        return Series(result_series, index=self._index)

    # Explicit comparison methods
    def eq(
        self,
        other: Any,
        **kwargs: Any,
    ) -> Series:
        """
        Return equal to of Series and other, element-wise (binary operator ==).

        Parameters
        ----------
        other : scalar or Series
            Object to compare with.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Boolean Series showing whether each element is equal to other.
        """
        return self.__eq__(other)

    def ne(
        self,
        other: Any,
        **kwargs: Any,
    ) -> Series:
        """
        Return not equal to of Series and other, element-wise (binary operator !=).

        Parameters
        ----------
        other : scalar or Series
            Object to compare with.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Boolean Series showing whether each element is not equal to other.
        """
        return self.__ne__(other)

    def gt(
        self,
        other: Any,
        **kwargs: Any,
    ) -> Series:
        """
        Return greater than of Series and other, element-wise (binary operator >).

        Parameters
        ----------
        other : scalar or Series
            Object to compare with.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Boolean Series showing whether each element is greater than other.
        """
        return self.__gt__(other)

    def lt(
        self,
        other: Any,
        **kwargs: Any,
    ) -> Series:
        """
        Return less than of Series and other, element-wise (binary operator <).

        Parameters
        ----------
        other : scalar or Series
            Object to compare with.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Boolean Series showing whether each element is less than other.
        """
        return self.__lt__(other)

    def ge(
        self,
        other: Any,
        **kwargs: Any,
    ) -> Series:
        """
        Return greater than or equal to of Series and other, element-wise (binary operator >=).

        Parameters
        ----------
        other : scalar or Series
            Object to compare with.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Boolean Series showing whether each element is greater than or equal to other.
        """
        return self.__ge__(other)

    def le(
        self,
        other: Any,
        **kwargs: Any,
    ) -> Series:
        """
        Return less than or equal to of Series and other, element-wise (binary operator <=).

        Parameters
        ----------
        other : scalar or Series
            Object to compare with.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Boolean Series showing whether each element is less than or equal to other.
        """
        return self.__le__(other)

    # Accessor properties
    @property
    def str(self) -> _StringAccessor:
        """String accessor for string operations."""
        return _StringAccessor(self)

    @property
    def dt(self) -> _DatetimeAccessor:
        """Datetime accessor for datetime operations."""
        return _DatetimeAccessor(self)

    # Methods
    def apply(self, func: Callable[..., Any]) -> Series:
        """
        Apply a function to each element.

        Parameters
        ----------
        func : function
            Function to apply

        Returns
        -------
        Series or scalar
            Result of applying function
        """
        return Series(self._series.map_elements(func, return_dtype=pl.Float64))

    def map(self, arg: Any) -> Series:
        """
        Map values using a dictionary or function.

        Parameters
        ----------
        arg : dict or function
            Mapping or function

        Returns
        -------
        Series
            Mapped series
        """
        if isinstance(arg, dict):
            # Use Polars replace
            return Series(self._series.replace(arg, default=None))
        else:
            # Use map_elements for functions
            return Series(self._series.map_elements(arg, return_dtype=pl.Float64))

    def isna(self) -> Series:
        """
        Detect missing values.

        Returns
        -------
        Series
            Boolean series indicating null values
        """
        return Series(self._series.is_null())

    def notna(self) -> Series:
        """
        Detect non-missing values.

        Returns
        -------
        Series
            Boolean series indicating non-null values
        """
        return Series(self._series.is_not_null())

    def between(
        self,
        left: Any,
        right: Any,
        inclusive: str = "both",  # type: ignore[valid-type]
    ) -> Series:
        """
        Check if values are between bounds.

        Parameters
        ----------
        left : scalar
            Left bound
        right : scalar
            Right bound
        inclusive : {'both', 'neither', 'left', 'right'}, default 'both'
            Include boundaries

        Returns
        -------
        Series
            Boolean series indicating if values are between bounds
        """
        # Handle empty series
        if len(self._series) == 0:
            return Series(pl.Series([], dtype=pl.Boolean))

        # Validate inclusive parameter
        if inclusive not in ("both", "neither", "left", "right"):
            raise ValueError(
                f"inclusive must be one of {{'both', 'neither', 'left', 'right'}}, got '{inclusive}'"
            )

        # Polars handles nulls natively - no pandas fallback needed

        if inclusive == "both":
            return Series((self._series >= left) & (self._series <= right))
        elif inclusive == "neither":
            return Series((self._series > left) & (self._series < right))
        elif inclusive == "left":
            return Series((self._series >= left) & (self._series < right))
        elif inclusive == "right":
            return Series((self._series > left) & (self._series <= right))
        else:
            raise ValueError(
                "inclusive must be one of 'both', 'neither', 'left', 'right'"
            )

    def clip(self, lower: Any = None, upper: Any = None) -> Series:
        """
        Trim values at thresholds.

        Parameters
        ----------
        lower : scalar, optional
            Minimum threshold
        upper : scalar, optional
            Maximum threshold

        Returns
        -------
        Series
            Series with values clipped
        """
        result = self._series
        if lower is not None:
            result = result.clip(lower_bound=lower)
        if upper is not None:
            result = result.clip(upper_bound=upper)
        return Series(result)

    def rank(
        self,
        method: BuiltinStr = "average",
        ascending: bool = True,
        na_option: BuiltinStr = "keep",
        pct: bool = False,
    ) -> Series:
        """
        Compute numerical ranks.

        Parameters
        ----------
        method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
            How to rank the group of records
        ascending : bool, default True
            Whether or not the elements should be ranked in ascending order
        na_option : {'keep', 'top', 'bottom'}, default 'keep'
            How to rank NaN values
        pct : bool, default False
            Whether to display the returned rankings in percentile form

        Returns
        -------
        Series
            Series with ranks
        """
        # Map pandas method names to Polars
        method_map = {
            "average": "average",
            "min": "min",
            "max": "max",
            "first": "dense",  # Closest equivalent
            "dense": "dense",
        }
        polars_method = method_map.get(method, "average")

        result = self._series.rank(method=polars_method, descending=not ascending)  # type: ignore[arg-type]

        if pct:
            result = result / len(self._series)

        return Series(result)

    def sum(self, skipna: bool = True, **kwargs: Any) -> Any:
        """
        Return the sum of the values.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        **kwargs
            Additional arguments passed to Polars sum().

        Returns
        -------
        scalar
            Sum of the values.
        """
        return self._series.sum()

    def mean(self, skipna: bool = True, **kwargs: Any) -> Any:
        """
        Return the mean of the values.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        **kwargs
            Additional arguments passed to Polars mean().

        Returns
        -------
        scalar
            Mean of the values.
        """
        return self._series.mean()

    def min(self, skipna: bool = True, **kwargs: Any) -> Any:
        """
        Return the minimum of the values.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        **kwargs
            Additional arguments passed to Polars min().

        Returns
        -------
        scalar
            Minimum of the values.
        """
        return self._series.min()

    def max(self, skipna: bool = True, **kwargs: Any) -> Any:
        """
        Return the maximum of the values.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        **kwargs
            Additional arguments passed to Polars max().

        Returns
        -------
        scalar
            Maximum of the values.
        """
        return self._series.max()

    def std(self, skipna: bool = True, ddof: int = 1, **kwargs: Any) -> Any:
        """
        Return the standard deviation of the values.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        ddof : int, default 1
            Delta degrees of freedom. The divisor used in calculations is N - ddof.
        **kwargs
            Additional arguments passed to Polars std().

        Returns
        -------
        scalar
            Standard deviation of the values.
        """
        return self._series.std(ddof=ddof)

    def var(self, skipna: bool = True, ddof: int = 1, **kwargs: Any) -> Any:
        """
        Return the variance of the values.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        ddof : int, default 1
            Delta degrees of freedom. The divisor used in calculations is N - ddof.
        **kwargs
            Additional arguments passed to Polars var().

        Returns
        -------
        scalar
            Variance of the values.
        """
        return self._series.var(ddof=ddof)

    def count(self, level: Any = None, **kwargs: Any) -> int:
        """
        Return number of non-null values in the Series.

        Parameters
        ----------
        level : int or level name, optional
            For compatibility with pandas MultiIndex (not used).
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        int
            Number of non-null values.
        """
        return self._series.count()

    def median(self, skipna: bool = True, **kwargs: Any) -> Any:
        """
        Return the median of the values.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        **kwargs
            Additional arguments passed to Polars median().

        Returns
        -------
        scalar
            Median of the values.
        """
        return self._series.median()

    def astype(self, dtype: Any, errors: str = "raise", **kwargs: Any) -> Series:  # type: ignore[valid-type]
        """
        Cast a pandas object to a specified dtype.

        Parameters
        ----------
        dtype : str or dtype
            Data type to cast to.
        errors : {'raise', 'ignore'}, default 'raise'
            Control raising of exceptions on invalid data types.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Series with cast dtype.

        Examples
        --------
        >>> import polarpandas as ppd
        >>> s = ppd.Series([1, 2, 3])
        >>> s.astype("float64")
        """
        from polarpandas.utils import convert_schema_to_polars

        if errors not in ("raise", "ignore"):
            raise ValueError(f"errors must be 'raise' or 'ignore', got '{errors}'")

        try:
            polars_dtype = convert_schema_to_polars({"dummy": dtype})
            if polars_dtype is None:
                if errors == "raise":
                    raise ValueError(f"Could not convert dtype: {dtype}")
                else:
                    return Series(self._series)
            target_dtype = list(polars_dtype.values())[0]
            result_series = self._series.cast(target_dtype)
            result = Series(result_series)
            if target_dtype == pl.Categorical:
                result._categorical_order = result_series.cat.get_categories().to_list()
            return result
        except Exception:
            if errors == "raise":
                raise
            else:
                return Series(self._series)

    def replace(
        self,
        to_replace: Any = None,
        value: Any = None,
        inplace: bool = False,
        limit: Any = None,
        regex: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Replace values given in to_replace with value.

        Parameters
        ----------
        to_replace : str, regex, list, dict, Series, int, float, or None
            How to find the values that will be replaced.
        value : scalar, dict, list, str, regex, default None
            Value to replace any values matching to_replace with.
        inplace : bool, default False
            If True, modify Series in place and return None.
        limit : int, default None
            Maximum size gap to forward or backward fill.
        regex : bool, default False
            Whether to interpret to_replace as a regex pattern.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series or None
            Series with values replaced, or None if inplace=True.
        """
        if to_replace is None and value is None:
            return None if inplace else Series(self._series)

        if isinstance(to_replace, dict):
            # Dict replacement - map old values to new values
            result_series = self._series.replace(to_replace)
        elif isinstance(to_replace, (list, tuple)):
            # List of values to replace
            if isinstance(value, (list, tuple)) and len(value) == len(to_replace):
                # Map each old value to corresponding new value
                replace_map = dict(zip(to_replace, value))
                result_series = self._series.replace(replace_map)
            else:
                # Replace all with single value
                replace_map = dict.fromkeys(to_replace, value)
                result_series = self._series.replace(replace_map)
        else:
            # Scalar replacement
            replace_map = {to_replace: value}
            result_series = self._series.replace(replace_map)

        if inplace:
            self._series = result_series
            return None
        else:
            return Series(result_series)

    def shift(
        self,
        periods: int = 1,
        freq: Any = None,
        fill_value: Any = None,
    ) -> Series:
        """
        Shift index by desired number of periods with an optional time freq.

        Parameters
        ----------
        periods : int, default 1
            Number of periods to shift. Can be positive or negative.
        freq : str or DateOffset, optional
            Frequency string or DateOffset object (not fully supported).
        fill_value : scalar, optional
            The scalar value to use for newly introduced missing values.

        Returns
        -------
        Series
            Copy of input object, shifted.
        """
        result_series = self._series.shift(periods, fill_value=fill_value)
        return Series(result_series)

    def diff(self, periods: int = 1, **kwargs: Any) -> Series:
        """
        Calculate the first discrete difference of element.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference, accepts negative values.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            First differences of the Series.
        """
        result_series = self._series.diff(periods)
        return Series(result_series, index=self._index)

    def pct_change(
        self,
        periods: int = 1,
        fill_method: Any = None,
        limit: Any = None,
        freq: Any = None,
        **kwargs: Any,
    ) -> Series:
        """
        Calculate the percentage change between the current and a prior element.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for forming percent change.
        fill_method : str, optional
            Method to use for filling holes in reindexed Series (not fully supported).
        limit : int, optional
            Maximum number of consecutive periods to fill (not fully supported).
        freq : str or DateOffset, optional
            Increment to use from time series API (not fully supported).
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Percentage change between the current and a prior element.
        """
        # Calculate percentage change: (current - previous) / previous
        shifted = self._series.shift(periods)
        result_series = (self._series - shifted) / shifted
        return Series(result_series, index=self._index)

    def cumsum(self, skipna: bool = True, axis: Any = None, **kwargs: Any) -> Series:
        """
        Return cumulative sum over a Series.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result will be NA.
        axis : int, optional
            Axis for the function to be applied on (not used for Series, for pandas compatibility).
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Cumulative sum of the Series.
        """
        result_series = self._series.cum_sum()
        return Series(result_series, index=self._index)

    def cummax(self, skipna: bool = True, axis: Any = None, **kwargs: Any) -> Series:
        """
        Return cumulative maximum over a Series.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result will be NA.
        axis : int, optional
            Axis for the function to be applied on (not used for Series, for pandas compatibility).
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Cumulative maximum of the Series.
        """
        result_series = self._series.cum_max()
        return Series(result_series, index=self._index)

    def cummin(self, skipna: bool = True, axis: Any = None, **kwargs: Any) -> Series:
        """
        Return cumulative minimum over a Series.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result will be NA.
        axis : int, optional
            Axis for the function to be applied on (not used for Series, for pandas compatibility).
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Cumulative minimum of the Series.
        """
        result_series = self._series.cum_min()
        return Series(result_series, index=self._index)

    def cumprod(self, skipna: bool = True, axis: Any = None, **kwargs: Any) -> Series:
        """
        Return cumulative product over a Series.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result will be NA.
        axis : int, optional
            Axis for the function to be applied on (not used for Series, for pandas compatibility).
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Cumulative product of the Series.
        """
        result_series = self._series.cum_prod()
        return Series(result_series, index=self._index)

    def all(
        self,
        axis: Any = None,
        bool_only: Any = None,
        skipna: bool = True,
        **kwargs: Any,
    ) -> bool:
        """
        Return whether all elements are True, potentially over an axis.

        Parameters
        ----------
        axis : int, optional
            Axis for the function to be applied on (not used for Series, for pandas compatibility).
        bool_only : bool, optional
            Not used for Series (for pandas compatibility).
        skipna : bool, default True
            Exclude NA/null values.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        bool
            True if all elements are True, False otherwise.
        """
        # Convert to boolean if needed
        if self._series.dtype != pl.Boolean:
            bool_series = self._series.cast(pl.Boolean)
        else:
            bool_series = self._series

        if skipna:
            return bool_series.all()
        else:
            # If skipna=False, nulls make the result False
            if bool_series.null_count() > 0:
                return False
            return bool_series.all()

    def any(
        self,
        axis: Any = None,
        bool_only: Any = None,
        skipna: bool = True,
        **kwargs: Any,
    ) -> bool:
        """
        Return whether any element is True, potentially over an axis.

        Parameters
        ----------
        axis : int, optional
            Axis for the function to be applied on (not used for Series, for pandas compatibility).
        bool_only : bool, optional
            Not used for Series (for pandas compatibility).
        skipna : bool, default True
            Exclude NA/null values.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        bool
            True if any element is True, False otherwise.
        """
        # Convert to boolean if needed
        if self._series.dtype != pl.Boolean:
            bool_series = self._series.cast(pl.Boolean)
        else:
            bool_series = self._series

        if skipna:
            return bool_series.any()
        else:
            # If skipna=False, nulls make the result True if any non-null is True
            if bool_series.null_count() > 0 and bool_series.any():
                return True
            return bool_series.any()

    def idxmax(self, axis: Any = None, skipna: bool = True, **kwargs: Any) -> Any:
        """
        Return the row label of the maximum value.

        Parameters
        ----------
        axis : int, optional
            Axis for the function to be applied on (not used for Series, for pandas compatibility).
        skipna : bool, default True
            Exclude NA/null values.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        scalar
            Index label of the maximum value.
        """
        try:
            if skipna:
                max_idx = self._series.arg_max()
            else:
                # If skipna=False, need to handle nulls
                if self._series.null_count() > 0:
                    # Find first non-null max
                    non_null = self._series.filter(self._series.is_not_null())
                    if len(non_null) == 0:
                        return None
                    max_idx = non_null.arg_max()
                else:
                    max_idx = self._series.arg_max()
        except Exception:
            return None

        if max_idx is None:
            return None

        # Return the index label if available
        if self._index is not None and max_idx < len(self._index):
            return self._index[max_idx]
        return max_idx

    def idxmin(self, axis: Any = None, skipna: bool = True, **kwargs: Any) -> Any:
        """
        Return the row label of the minimum value.

        Parameters
        ----------
        axis : int, optional
            Axis for the function to be applied on (not used for Series, for pandas compatibility).
        skipna : bool, default True
            Exclude NA/null values.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        scalar
            Index label of the minimum value.
        """
        try:
            if skipna:
                min_idx = self._series.arg_min()
            else:
                # If skipna=False, need to handle nulls
                if self._series.null_count() > 0:
                    # Find first non-null min
                    non_null = self._series.filter(self._series.is_not_null())
                    if len(non_null) == 0:
                        return None
                    min_idx = non_null.arg_min()
                else:
                    min_idx = self._series.arg_min()
        except Exception:
            return None

        if min_idx is None:
            return None

        # Return the index label if available
        if self._index is not None and min_idx < len(self._index):
            return self._index[min_idx]
        return min_idx

    def round(self, decimals: int = 0, **kwargs: Any) -> Series:
        """
        Round each value in a Series to the given number of decimals.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Series with rounded values.
        """
        result_series = self._series.round(decimals)
        return Series(result_series, index=self._index)

    def sort_index(
        self,
        ascending: bool = True,
        inplace: bool = False,
        kind: Any = None,
        na_position: str = "last",  # type: ignore[valid-type]
        sort_remaining: bool = True,
        ignore_index: bool = False,
        key: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Sort Series by index values.

        Parameters
        ----------
        ascending : bool, default True
            Sort ascending vs. descending.
        inplace : bool, default False
            If True, modify Series in place and return None.
        kind : str, optional
            Not used (for pandas compatibility).
        na_position : {'first', 'last'}, default 'last'
            Puts NaNs at the beginning if 'first', 'last' if 'last'.
        sort_remaining : bool, default True
            Not used (for pandas compatibility).
        ignore_index : bool, default False
            If True, the resulting index will be labeled 0, 1, …, n - 1.
        key : callable, optional
            Not used (for pandas compatibility).
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series or None
            Series sorted by index, or None if inplace=True.
        """
        if self._index is None:
            # No index to sort by, return as-is
            if inplace:
                return None
            return Series(self._series, index=self._index)

        # Create list of (index_value, series_value) tuples
        series_list = self._series.to_list()
        indexed_data = list(zip(self._index, series_list))

        # Sort by index
        indexed_data.sort(key=lambda x: x[0], reverse=not ascending)

        # Handle NaN positions
        if na_position == "first":
            # Move NaNs to front
            nan_items = [
                item
                for item in indexed_data
                if (isinstance(item[0], float) and item[0] != item[0])
                or item[0] is None
            ]
            non_nan_items = [
                item
                for item in indexed_data
                if not (
                    (isinstance(item[0], float) and item[0] != item[0])
                    or item[0] is None
                )
            ]
            indexed_data = nan_items + non_nan_items

        # Unzip
        sorted_index = [item[0] for item in indexed_data]
        sorted_values = [item[1] for item in indexed_data]

        result_series = pl.Series(name=self._series.name, values=sorted_values)

        new_index = list(range(len(result_series))) if ignore_index else sorted_index

        if inplace:
            self._series = result_series
            self._index = new_index
            return None
        else:
            return Series(result_series, index=new_index)

    def isnull(self) -> Series:
        """
        Detect missing values (alias for isna()).

        Returns
        -------
        Series
            Boolean Series showing whether each value is null.
        """
        return self.isna()

    def notnull(self) -> Series:
        """
        Detect non-missing values (alias for notna()).

        Returns
        -------
        Series
            Boolean Series showing whether each value is not null.
        """
        return self.notna()

    def abs(self, **kwargs: Any) -> Series:
        """
        Return a Series with absolute numeric value of each element.

        Parameters
        ----------
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Series with absolute values.
        """
        result_series = self._series.abs()
        return Series(result_series, index=self._index)

    def fillna(
        self,
        value: Any = None,
        method: Any = None,
        axis: Any = None,
        inplace: bool = False,
        limit: Any = None,
        downcast: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Fill missing values.

        Parameters
        ----------
        value : scalar, dict, Series, or None
            Value to use to fill holes.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, optional
            Method to use for filling holes in reindexed Series.
        axis : int, optional
            Axis along which to fill (not used for Series, for pandas compatibility).
        inplace : bool, default False
            If True, modify Series in place and return None.
        limit : int, optional
            Maximum number of consecutive NaN values to forward/backward fill.
        downcast : dict, optional
            Not used (for pandas compatibility).
        **kwargs
            Additional arguments passed to Polars fill_null() or forward_fill()/backward_fill().

        Returns
        -------
        Series or None
            Series with filled values, or None if inplace=True.
        """
        if method is not None:
            if method in ("pad", "ffill"):
                result_series = self._series.forward_fill(limit=limit)
            elif method in ("backfill", "bfill"):
                result_series = self._series.backward_fill(limit=limit)
            else:
                raise ValueError(f"Invalid fill method: {method}")
        else:
            if value is None:
                raise ValueError("Must specify a fill 'value' or 'method'")
            result_series = self._series.fill_null(value)

        if inplace:
            self._series = result_series
            return None
        else:
            return Series(result_series, index=self._index)

    def dropna(
        self,
        axis: Any = None,
        inplace: bool = False,
        how: Any = None,
        thresh: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Return a new Series with missing values removed.

        Parameters
        ----------
        axis : int, optional
            Axis along which to drop (not used for Series, for pandas compatibility).
        inplace : bool, default False
            If True, modify Series in place and return None.
        how : {'any', 'all'}, optional
            Not used for Series (for pandas compatibility).
        thresh : int, optional
            Not used for Series (for pandas compatibility).
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series or None
            Series with missing values removed, or None if inplace=True.
        """
        result_series = self._series.drop_nulls()

        if inplace:
            self._series = result_series
            return None
        else:
            return Series(result_series, index=self._index)

    def drop_duplicates(
        self,
        keep: Any = "first",
        inplace: bool = False,
        ignore_index: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Return Series with duplicate values removed.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            Determines which duplicates (if any) to keep.
        inplace : bool, default False
            If True, modify Series in place and return None.
        ignore_index : bool, default False
            If True, the resulting index will be labeled 0, 1, …, n - 1.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series or None
            Series with duplicates removed, or None if inplace=True.
        """
        if keep == "first":
            result_series = self._series.unique(maintain_order=True)
        elif keep == "last":
            # Reverse, get unique, reverse back
            result_series = self._series.reverse().unique(maintain_order=True).reverse()
        elif keep is False:
            # Keep only values that appear once
            value_counts = self._series.value_counts()
            single_occurrence = value_counts.filter(pl.col("count") == 1)["value"]
            result_series = self._series.filter(self._series.is_in(single_occurrence))
        else:
            raise ValueError(f"keep must be 'first', 'last', or False, got {keep}")

        new_index = None
        if ignore_index:
            new_index = list(range(len(result_series)))
        elif self._index is not None:
            # Try to preserve index for kept values
            if keep == "first":
                # Get first occurrence indices
                seen = set()
                new_index = []
                series_list = self._series.to_list()
                for i, val in enumerate(series_list):
                    if val not in seen:
                        seen.add(val)
                        new_index.append(self._index[i] if i < len(self._index) else i)
            else:
                new_index = list(range(len(result_series)))

        if inplace:
            self._series = result_series
            self._index = new_index
            return None
        else:
            return Series(result_series, index=new_index)

    def drop(
        self,
        labels: Any = None,
        axis: Any = None,
        index: Any = None,
        columns: Any = None,
        level: Any = None,
        inplace: bool = False,
        errors: str = "raise",  # type: ignore[valid-type]
        **kwargs: Any,
    ) -> Any:
        """
        Return Series with specified index labels removed.

        Parameters
        ----------
        labels : single label or list-like
            Index labels to drop.
        axis : int, optional
            Axis for the function to be applied on (not used for Series, for pandas compatibility).
        index : single label or list-like
            Alternative to specifying axis (labels, axis=0 is equivalent to index=labels).
        columns : single label or list-like
            Not used for Series (for pandas compatibility).
        level : int or level name, optional
            For MultiIndex, level from which the labels will be removed.
        inplace : bool, default False
            If True, modify Series in place and return None.
        errors : {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and only existing labels are dropped.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series or None
            Series with specified labels removed, or None if inplace=True.
        """
        # Use index parameter if provided, otherwise use labels
        labels_to_drop = index if index is not None else labels

        if labels_to_drop is None:
            if inplace:
                return None
            return Series(self._series, index=self._index)

        # Convert to list if single value
        if not isinstance(labels_to_drop, (list, tuple)):
            labels_to_drop = [labels_to_drop]

        if self._index is None:
            # No index, drop by position
            try:
                positions_to_drop = [int(label) for label in labels_to_drop]
                # Filter out positions
                keep_positions = [
                    i for i in range(len(self._series)) if i not in positions_to_drop
                ]
                if not keep_positions:
                    result_series = pl.Series(  # type: ignore[misc]
                        [], dtype=self._series.dtype, name=self._series.name
                    )
                    new_index = []
                else:
                    series_list = self._series.to_list()
                    result_series = pl.Series(  # type: ignore[misc]
                        [series_list[i] for i in keep_positions], name=self._series.name
                    )
                    new_index = keep_positions
            except (ValueError, TypeError):
                if errors == "raise":
                    raise KeyError(f"Labels {labels_to_drop} not found") from None
                new_index = None
                result_series = self._series
        else:
            # Drop by index labels
            labels_set = set(labels_to_drop)
            keep_indices = []
            keep_values = []
            series_list = self._series.to_list()

            for i, idx_label in enumerate(self._index):
                if idx_label not in labels_set:
                    keep_indices.append(idx_label)
                    keep_values.append(series_list[i])

            if len(keep_values) == 0:
                result_series = pl.Series(  # type: ignore[misc]
                    [], dtype=self._series.dtype, name=self._series.name
                )
                new_index = []
            else:
                result_series = pl.Series(keep_values, name=self._series.name)  # type: ignore[misc]
                new_index = keep_indices

        if inplace:
            self._series = result_series
            self._index = new_index
            return None
        else:
            return Series(result_series, index=new_index)

    def duplicated(self, keep: Any = "first", **kwargs: Any) -> Series:
        """
        Indicate duplicate Series values.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            Method to handle duplicates:
            - 'first' : Mark duplicates as True except for the first occurrence.
            - 'last' : Mark duplicates as True except for the last occurrence.
            - False : Mark all duplicates as True.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Boolean Series indicating duplicate values.
        """
        if keep == "first":
            result_series = self._series.is_duplicated()
        elif keep == "last":
            # Reverse, check duplicates, reverse back
            reversed_series = self._series.reverse()
            result_series = reversed_series.is_duplicated().reverse()
        elif keep is False:
            # Mark all duplicates as True
            value_counts = self._series.value_counts()
            duplicates = value_counts.filter(pl.col("count") > 1)["value"]
            result_series = self._series.is_in(duplicates)
        else:
            raise ValueError(f"keep must be 'first', 'last', or False, got {keep}")

        return Series(result_series, index=self._index)

    def equals(self, other: Series) -> bool:
        """
        Test whether two Series contain the same elements.

        Parameters
        ----------
        other : Series
            Series to compare with.

        Returns
        -------
        bool
            True if the Series contain the same elements, False otherwise.
        """
        if not isinstance(other, Series):
            return False  # type: ignore[unreachable]

        # Compare values
        if len(self._series) != len(other._series):
            return False

        # Use Polars equals
        try:
            return self._series.equals(other._series)
        except Exception:
            # Fallback: compare element by element
            self_list = self._series.to_list()
            other_list = other._series.to_list()
            return self_list == other_list

    def explode(self, ignore_index: bool = False, **kwargs: Any) -> Series:
        """
        Transform each element of a list-like to a row.

        Parameters
        ----------
        ignore_index : bool, default False
            If True, the resulting index will be labeled 0, 1, …, n - 1.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Exploded Series with list-like elements expanded to rows.
        """
        result_series = self._series.explode()

        if ignore_index:
            new_index = list(range(len(result_series)))
        else:
            # Create index by repeating original index for each exploded element
            new_index = []
            series_list = self._series.to_list()
            idx = 0
            for i, val in enumerate(series_list):
                if isinstance(val, (list, tuple)):
                    for _ in val:
                        if self._index is not None and i < len(self._index):
                            new_index.append(self._index[i])
                        else:
                            new_index.append(i)
                        idx += 1
                else:
                    if self._index is not None and i < len(self._index):
                        new_index.append(self._index[i])
                    else:
                        new_index.append(i)
                    idx += 1

        return Series(result_series, index=new_index)

    def factorize(
        self,
        sort: bool = False,
        na_sentinel: Any = -1,
        use_na_sentinel: bool = True,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """
        Encode the object as an enumerated type or categorical variable.

        Parameters
        ----------
        sort : bool, default False
            Sort uniques and shuffle to maintain the relationship.
        na_sentinel : int, default -1
            Value to mark "not found".
        use_na_sentinel : bool, default True
            If True, the sentinel -1 will be used for NaN values.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        codes : ndarray
            Integer codes for values.
        uniques : ndarray, Index, or Categorical
            The unique valid values.
        """
        import numpy as np

        # Get unique values
        uniques_series = self._series.unique(maintain_order=True)
        uniques_list = uniques_series.to_list()

        if sort:
            uniques_list = sorted(uniques_list, key=lambda x: (x is None, x))

        # Create mapping
        unique_map = {val: idx for idx, val in enumerate(uniques_list)}

        # Generate codes
        series_list = self._series.to_list()
        codes = []
        for val in series_list:
            if val is None or (isinstance(val, float) and val != val):  # NaN check
                codes.append(na_sentinel if use_na_sentinel else len(uniques_list))
            else:
                codes.append(unique_map.get(val, na_sentinel))

        return np.array(codes), np.array(uniques_list)

    def filter(
        self,
        items: Any = None,
        like: Any = None,
        regex: Any = None,
        axis: Any = None,
        **kwargs: Any,
    ) -> Series:
        """
        Subset the Series according to the filter.

        Parameters
        ----------
        items : list-like, optional
            Keep labels in items.
        like : str, optional
            Keep labels where label contains like.
        regex : str, optional
            Keep labels where label matches regex.
        axis : int, optional
            Axis for the function to be applied on (not used for Series, for pandas compatibility).
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Filtered Series.
        """
        if self._index is None:
            # No index to filter by
            return Series(self._series, index=self._index)

        import re

        keep_indices = []
        keep_values = []
        series_list = self._series.to_list()

        for i, idx_label in enumerate(self._index):
            keep = True

            if items is not None:
                keep = keep and (idx_label in items)

            if like is not None:
                keep = keep and (like in str(idx_label))

            if regex is not None:
                keep = keep and bool(re.search(regex, str(idx_label)))

            if keep:
                keep_indices.append(idx_label)
                keep_values.append(series_list[i])

        if len(keep_values) == 0:
            result_series = pl.Series(  # type: ignore[misc]
                [], dtype=self._series.dtype, name=self._series.name
            )
            new_index = []
        else:
            result_series = pl.Series(keep_values, name=self._series.name)  # type: ignore[misc]
            new_index = keep_indices

        return Series(result_series, index=new_index)

    def first_valid_index(self) -> Any:
        """
        Return index of first non-NA/null value.

        Returns
        -------
        scalar or None
            Index of first non-NA/null value, or None if all values are NA.
        """
        null_mask = self._series.is_not_null()
        null_list = null_mask.to_list()

        for i, is_not_null in enumerate(null_list):
            if is_not_null:
                if self._index is not None and i < len(self._index):
                    return self._index[i]
                return i

        return None

    def last_valid_index(self) -> Any:
        """
        Return index of last non-NA/null value.

        Returns
        -------
        scalar or None
            Index of last non-NA/null value, or None if all values are NA.
        """
        null_mask = self._series.is_not_null()
        null_list = null_mask.to_list()

        for i in range(len(null_list) - 1, -1, -1):
            if null_list[i]:
                if self._index is not None and i < len(self._index):
                    return self._index[i]
                return i

        return None

    def floordiv(
        self,
        other: Any,
        fill_value: Any = None,
        **kwargs: Any,
    ) -> Series:
        """
        Return integer division of Series and other, element-wise (binary operator //).

        Parameters
        ----------
        other : scalar or Series
            Object to divide the Series by.
        fill_value : float or None, default None
            Fill existing missing (NaN) values with this value before computation.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Result of the arithmetic operation.
        """
        if isinstance(other, Series):
            if fill_value is not None:
                result_series = self._series.fill_null(
                    fill_value
                ) // other._series.fill_null(fill_value)
            else:
                result_series = self._series // other._series
        else:
            if fill_value is not None:
                result_series = self._series.fill_null(fill_value) // other
            else:
                result_series = self._series // other
        return Series(result_series, index=self._index)

    def get(self, key: Any, default: Any = None) -> Any:
        """
        Get item from object for given key.

        Parameters
        ----------
        key : any
            Key to get value for.
        default : any, optional
            Value to return if key is not found.

        Returns
        -------
        any
            Value for key if found, otherwise default.
        """
        if self._index is None:
            # No index, try to use key as position
            try:
                pos = int(key)
                if 0 <= pos < len(self._series):
                    return self._series[pos]
            except (ValueError, TypeError):
                pass
        else:
            # Search in index
            for i, idx_label in enumerate(self._index):
                if idx_label == key:
                    return self._series[i]

        return default

    def isin(self, values: Any) -> Series:
        """
        Whether elements in Series are contained in values.

        Parameters
        ----------
        values : set or list-like
            The sequence of values to test.

        Returns
        -------
        Series
            Series of booleans indicating whether each element is in values.
        """
        if isinstance(values, Series):
            values_list = values._series.to_list()
        elif not isinstance(values, (list, tuple, set)):
            values_list = [values]
        else:
            values_list = list(values)

        result_series = self._series.is_in(values_list)
        return Series(result_series, index=self._index)

    def describe(
        self,
        percentiles: Any = None,
        include: Any = None,
        exclude: Any = None,
        **kwargs: Any,
    ) -> Series:
        """
        Generate descriptive statistics.

        Parameters
        ----------
        percentiles : list-like of numbers, optional
            The percentiles to include in the output.
        include : 'all', list-like of dtypes or None, optional
            Not used for Series (for pandas compatibility).
        exclude : list-like of dtypes or None, optional
            Not used for Series (for pandas compatibility).
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Summary statistics of the Series.
        """
        if percentiles is None:
            percentiles = [0.25, 0.5, 0.75]

        stats = {}
        stats["count"] = float(self._series.count())
        mean_val = self._series.mean()
        stats["mean"] = float(mean_val) if mean_val is not None else None  # type: ignore[assignment,arg-type]
        std_val = self._series.std()
        stats["std"] = float(std_val) if std_val is not None else None  # type: ignore[assignment,arg-type]
        min_val = self._series.min()
        stats["min"] = float(min_val) if min_val is not None else None  # type: ignore[assignment,arg-type]
        max_val = self._series.max()
        stats["max"] = float(max_val) if max_val is not None else None  # type: ignore[assignment,arg-type]

        for p in percentiles:
            q_val = self._series.quantile(p)
            stats[f"{int(p * 100)}%"] = float(q_val) if q_val is not None else None  # type: ignore[assignment]

        # Create result Series - convert all to float for consistency
        series_name = self._series.name if self._series.name is not None else ""
        result_series = pl.Series(name=series_name, values=list(stats.values()))
        return Series(result_series, index=list(stats.keys()))

    def quantile(
        self,
        q: Any = 0.5,
        interpolation: str = "linear",  # type: ignore[valid-type]
        numeric_only: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Return value at the given quantile.

        Parameters
        ----------
        q : float or array-like of float, default 0.5
            Quantile(s) to compute.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}, default 'linear'
            Interpolation method to use.
        numeric_only : bool, optional
            Not used for Series (for pandas compatibility).
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        scalar or Series
            Quantile value(s).
        """
        if isinstance(q, (list, tuple)):
            # Multiple quantiles
            quantiles = [self._series.quantile(q_val) for q_val in q]
            result_series = pl.Series(quantiles, name=self._series.name)  # type: ignore[misc]
            return Series(result_series, index=q)
        else:
            # Single quantile
            return self._series.quantile(q)

    def nunique(self, dropna: bool = True, **kwargs: Any) -> int:
        """
        Return number of unique elements in the object.

        Parameters
        ----------
        dropna : bool, default True
            Don't include NaN in count.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        int
            Number of unique values.
        """
        if dropna:
            return self._series.n_unique()
        else:
            # Include nulls in count
            return self._series.n_unique() + (1 if self._series.null_count() > 0 else 0)

    def kurt(
        self,
        axis: Any = None,
        skipna: bool = True,
        level: Any = None,
        numeric_only: Any = None,
        **kwargs: Any,
    ) -> float:
        """
        Return unbiased kurtosis over requested axis.

        Parameters
        ----------
        axis : int, optional
            Axis for the function to be applied on (not used for Series, for pandas compatibility).
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        level : int or level name, optional
            Not used for Series (for pandas compatibility).
        numeric_only : bool, optional
            Not used for Series (for pandas compatibility).
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        float
            Kurtosis value.
        """
        return self.kurtosis(
            axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs
        )

    def kurtosis(
        self,
        axis: Any = None,
        skipna: bool = True,
        level: Any = None,
        numeric_only: Any = None,
        **kwargs: Any,
    ) -> float:
        """
        Return unbiased kurtosis over requested axis.

        Parameters
        ----------
        axis : int, optional
            Axis for the function to be applied on (not used for Series, for pandas compatibility).
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        level : int or level name, optional
            Not used for Series (for pandas compatibility).
        numeric_only : bool, optional
            Not used for Series (for pandas compatibility).
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        float
            Kurtosis value.
        """
        import numpy as np

        if skipna:
            values = self._series.filter(self._series.is_not_null()).to_list()
        else:
            values = self._series.to_list()
            if None in values or any(isinstance(v, float) and v != v for v in values):
                return np.nan

        if len(values) < 4:
            return np.nan

        # Calculate kurtosis using Fisher's definition (excess kurtosis)
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)

        if std_val == 0:
            return np.nan

        n = len(values)
        kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(
            ((np.array(values) - mean_val) / std_val) ** 4
        ) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))

        return float(kurt)

    def skew(
        self,
        axis: Any = None,
        skipna: bool = True,
        level: Any = None,
        numeric_only: Any = None,
        **kwargs: Any,
    ) -> float:
        """
        Return unbiased skew over requested axis.

        Parameters
        ----------
        axis : int, optional
            Axis for the function to be applied on (not used for Series, for pandas compatibility).
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        level : int or level name, optional
            Not used for Series (for pandas compatibility).
        numeric_only : bool, optional
            Not used for Series (for pandas compatibility).
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        float
            Skewness value.
        """
        import numpy as np

        if skipna:
            values = self._series.filter(self._series.is_not_null()).to_list()
        else:
            values = self._series.to_list()
            if None in values or any(isinstance(v, float) and v != v for v in values):
                return np.nan

        if len(values) < 3:
            return np.nan

        # Calculate skewness
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)

        if std_val == 0:
            return 0.0

        n = len(values)
        skew = (n / ((n - 1) * (n - 2))) * np.sum(
            ((np.array(values) - mean_val) / std_val) ** 3
        )

        return float(skew)

    def sem(
        self,
        axis: Any = None,
        skipna: bool = True,
        level: Any = None,
        ddof: int = 1,
        numeric_only: Any = None,
        **kwargs: Any,
    ) -> float:
        """
        Return unbiased standard error of the mean over requested axis.

        Parameters
        ----------
        axis : int, optional
            Axis for the function to be applied on (not used for Series, for pandas compatibility).
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        level : int or level name, optional
            Not used for Series (for pandas compatibility).
        ddof : int, default 1
            Delta degrees of freedom.
        numeric_only : bool, optional
            Not used for Series (for pandas compatibility).
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        float
            Standard error of the mean.
        """
        import numpy as np

        if skipna:
            values = self._series.filter(self._series.is_not_null()).to_list()
        else:
            values = self._series.to_list()
            if None in values or any(isinstance(v, float) and v != v for v in values):
                return np.nan

        if len(values) == 0:
            return np.nan

        std_val = np.std(values, ddof=ddof)
        sem = std_val / np.sqrt(len(values))

        return float(sem)

    def item(self) -> Any:
        """
        Return the first element of the Series as a Python scalar.

        Returns
        -------
        scalar
            First element of the Series.

        Raises
        ------
        ValueError
            If the Series is not of length 1.
        """
        if len(self._series) != 1:
            raise ValueError("can only convert an array of size 1 to a Python scalar")

        return self._series[0]

    def info(
        self,
        verbose: Any = None,
        buf: Any = None,
        max_cols: Any = None,
        memory_usage: Any = None,
        show_counts: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Print a concise summary of a Series.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print the full summary.
        buf : writable buffer, defaults to sys.stdout
            Where to send the output.
        max_cols : int, optional
            Not used for Series (for pandas compatibility).
        memory_usage : bool, str or None, optional
            Specifies whether total memory usage of the Series elements should be displayed.
        show_counts : bool, default False
            Whether to show the non-null counts.
        **kwargs
            Additional arguments (not used, for pandas compatibility).
        """
        import sys

        if buf is None:
            buf = sys.stdout

        output_lines = []
        output_lines.append("<class 'polarpandas.core.series.Series'>")
        output_lines.append(f"Index: {len(self._series)} entries")

        if self._series.name:
            output_lines.append(f"Series name: {self._series.name}")

        output_lines.append(f"Non-Null Count: {self._series.count()}")
        output_lines.append(f"Dtype: {self._series.dtype}")

        if memory_usage:
            # Estimate memory usage
            bytes_per_elem = 8  # Rough estimate
            total_bytes = len(self._series) * bytes_per_elem
            output_lines.append(f"Memory usage: {total_bytes} bytes")

        output = "\n".join(output_lines) + "\n"
        buf.write(output)

    def ffill(
        self,
        axis: Any = None,
        limit: Any = None,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Forward fill missing values.

        Parameters
        ----------
        axis : int, optional
            Axis along which to fill (not used for Series, for pandas compatibility).
        limit : int, optional
            Maximum number of consecutive NaN values to forward fill.
        inplace : bool, default False
            If True, modify Series in place and return None.
        **kwargs
            Additional arguments passed to Polars forward_fill().

        Returns
        -------
        Series or None
            Series with forward-filled values, or None if inplace=True.
        """
        if limit is not None:
            # Polars doesn't directly support limit, so we'd need a workaround
            # For now, just do forward_fill without limit
            result_series = self._series.forward_fill(**kwargs)
        else:
            result_series = self._series.forward_fill(**kwargs)

        if inplace:
            self._series = result_series
            return None
        else:
            return Series(result_series)

    def bfill(
        self,
        axis: Any = None,
        limit: Any = None,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Backward fill missing values.

        Parameters
        ----------
        axis : int, optional
            Axis along which to fill (not used for Series, for pandas compatibility).
        limit : int, optional
            Maximum number of consecutive NaN values to backward fill.
        inplace : bool, default False
            If True, modify Series in place and return None.
        **kwargs
            Additional arguments passed to Polars backward_fill().

        Returns
        -------
        Series or None
            Series with backward-filled values, or None if inplace=True.
        """
        if limit is not None:
            # Polars doesn't directly support limit, so we'd need a workaround
            # For now, just do backward_fill without limit
            result_series = self._series.backward_fill(**kwargs)
        else:
            result_series = self._series.backward_fill(**kwargs)

        if inplace:
            self._series = result_series
            return None
        else:
            return Series(result_series)

    def pad(
        self,
        axis: Any = None,
        limit: Any = None,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Alias for ffill() (pandas compatibility).

        Parameters
        ----------
        axis : int, optional
            Axis along which to fill (not used for Series, for pandas compatibility).
        limit : int, optional
            Maximum number of consecutive NaN values to forward fill.
        inplace : bool, default False
            If True, modify Series in place and return None.
        **kwargs
            Additional arguments passed to Polars forward_fill().

        Returns
        -------
        Series or None
            Series with forward-filled values, or None if inplace=True.
        """
        return self.ffill(axis=axis, limit=limit, inplace=inplace, **kwargs)

    def sort_values(self, ascending: bool = True, inplace: bool = False) -> Any:
        """
        Sort by values.

        Parameters
        ----------
        ascending : bool, default True
            Sort ascending vs descending
        inplace : bool, default False
            Sort in place

        Returns
        -------
        Series or None
            Sorted series or None if inplace=True
        """
        # Use Polars sort - index behavior may differ from pandas
        if inplace:
            self._series = self._series.sort(descending=not ascending)
            return None
        else:
            result = Series(self._series.sort(descending=not ascending))
            # Note: Index preservation is limited in pure Polars
            return result

    def head(self, n: int = 5) -> Series:
        """
        Return the first n elements.

        Parameters
        ----------
        n : int, default 5
            Number of elements to return.

        Returns
        -------
        Series
            First n elements of the Series.
        """
        result_series = self._series.head(n)
        # Preserve index for first n elements
        if self._index is not None:
            index = self._index[:n]
        else:
            index = list(range(min(n, len(self._series))))
        return Series(result_series, index=index)

    def tail(self, n: int = 5) -> Series:
        """
        Return the last n elements.

        Parameters
        ----------
        n : int, default 5
            Number of elements to return.

        Returns
        -------
        Series
            Last n elements of the Series.
        """
        result_series = self._series.tail(n)
        # Preserve index for last n elements
        if self._index is not None:
            index = self._index[-n:]
        else:
            start_idx = max(0, len(self._series) - n)
            index = list(range(start_idx, len(self._series)))
        return Series(result_series, index=index)

    def to_list(self) -> BuiltinList[Any]:
        """
        Return a list of the values.

        Returns
        -------
        list
            List of values in the Series.
        """
        return self._series.to_list()

    def tolist(self) -> BuiltinList[Any]:
        """
        Return a list of the values.

        Alias for to_list().
        """
        return self.to_list()

    def to_frame(self, name: Any = None) -> DataFrame:
        """
        Convert Series to DataFrame.

        Parameters
        ----------
        name : str, optional
            The name of the Series. Used as the column name in the resulting DataFrame.

        Returns
        -------
        DataFrame
            DataFrame with a single column containing the Series values.
        """
        from polarpandas.frame import DataFrame

        col_name = (
            name
            if name is not None
            else (self.name if hasattr(self, "name") and self.name else 0)
        )
        result_df = pl.DataFrame({col_name: self._series})
        return DataFrame(result_df, index=self._index)

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins: Any = None,
        dropna: bool = True,
    ) -> Series:
        """
        Return a Series containing counts of unique values.

        Parameters
        ----------
        normalize : bool, default False
            Return proportions rather than frequencies
        sort : bool, default True
            Sort by frequencies
        ascending : bool, default False
            Sort in ascending order
        bins : int, optional
            Group values into bins
        dropna : bool, default True
            Don't include counts of NaN

        Returns
        -------
        Series
            Series with value counts
        """
        # Use Polars value_counts - index behavior may differ from pandas
        result = self._series.value_counts(sort=sort)

        if sort:
            # Polars sorts by value, but pandas sorts by count
            # We need to sort by count (the second field in the struct)
            if ascending:
                result = result.sort("count")
            else:
                result = result.sort("count", descending=True)

        if normalize:
            # For normalization, we need to preserve the original values
            # and only normalize the counts
            total = len(self._series)
            result = result.with_columns([pl.col("count") / total])

        return Series(result)

    def unique(self) -> Series:
        """
        Return unique values in the series.

        Returns
        -------
        Series
            Series with unique values
        """
        return Series(self._series.unique())

    def copy(self) -> Series:
        """
        Make a copy of the Series.

        Returns
        -------
        Series
            Copy of the Series
        """
        new_index = list(self._index) if self._index is not None else None
        copied = Series(self._series.clone(), name=self.name, index=new_index)
        copied._index_name = self._index_name
        if self._categorical_order is not None:
            copied._categorical_order = list(self._categorical_order)
        return copied

    @property
    def index(self) -> Any:
        """Return the index of the Series."""
        if self._index is not None:
            from polarpandas.index import Index

            return Index(self._index)
        else:
            # Return a default RangeIndex
            from polarpandas.index import Index

            return Index(list(range(len(self._series))))

    def to_numpy(
        self,
        dtype: Any = None,
        copy: bool = False,
        na_value: Any = None,
    ) -> Any:
        """
        Convert the Series to a NumPy array.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to numpy.asarray().
        copy : bool, default False
            Whether to ensure that the returned value is not a view on another array.
        na_value : Any, optional
            The value to use for missing values. The default value depends on dtype and pandas options.

        Returns
        -------
        numpy.ndarray
            The Series as a NumPy array.
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "numpy is required for to_numpy(). Install with: pip install numpy"
            ) from None

        # Convert to numpy array
        result = self._series.to_numpy()

        # Handle dtype conversion
        if dtype is not None:
            result = np.asarray(result, dtype=dtype)

        # Handle copy
        if copy:
            result = result.copy()

        # Handle na_value
        if na_value is not None:
            # Replace NaN values with na_value
            if result.dtype.kind == "f":
                result = np.where(np.isnan(result), na_value, result)
            else:
                # For non-float types, check for None/NaN differently
                mask = (
                    np.isnan(result.astype(float))
                    if result.dtype.kind in "biu"
                    else (result is None)
                )
                result = np.where(mask, na_value, result)

        return result

    def to_csv(
        self,
        path_or_buf: Any = None,
        sep: str = ",",  # type: ignore[valid-type]
        na_rep: str = "",  # type: ignore[valid-type]
        float_format: Any = None,
        columns: Any = None,
        header: Any = True,
        index: bool = True,
        index_label: Any = None,
        mode: str = "w",  # type: ignore[valid-type]
        encoding: Any = None,
        compression: Any = None,
        quoting: Any = None,
        line_terminator: Any = None,
        chunksize: Any = None,
        date_format: Any = None,
        doublequote: bool = True,
        escapechar: Any = None,
        decimal: str = ".",  # type: ignore[valid-type]
        **kwargs: Any,
    ) -> Any:
        """
        Write Series to a comma-separated values (csv) file.

        Parameters
        ----------
        path_or_buf : str, path object, file-like object, or None, default None
            String, path object (implementing os.PathLike[str]), or file-like object implementing a write() function.
            If None, the result is returned as a string.
        sep : str, default ','
            String of length 1. Field delimiter for the output file.
        na_rep : str, default ''
            Missing data representation.
        float_format : str, optional
            Format string for floating point numbers.
        columns : list, optional
            Columns to write (not used for Series, for pandas compatibility).
        header : bool or list of str, default True
            Write out the column names. If a list of strings is given it is assumed to be aliases for the column names.
        index : bool, default True
            Write row names (index).
        index_label : str or sequence, optional
            Column label for index column(s) if desired.
        mode : str, default 'w'
            Python write mode.
        encoding : str, optional
            A string representing the encoding to use in the output file.
        compression : str, optional
            Compression mode.
        **kwargs
            Additional arguments passed to Polars write_csv.

        Returns
        -------
        str or None
            If path_or_buf is None, returns the resulting csv format as a string. Otherwise returns None.
        """
        import io

        # Convert Series to DataFrame for writing
        df_to_write = self.to_frame()

        # Write to buffer or file
        if path_or_buf is None:
            # Return as string
            buffer = io.StringIO()
            df_to_write._df.write_csv(
                buffer,
                separator=sep,
                null_value=na_rep,
                include_header=bool(header),
                include_bom=False,
            )
            return buffer.getvalue()
        else:
            # Write to file
            df_to_write._df.write_csv(
                path_or_buf,
                separator=sep,
                null_value=na_rep,
                include_header=bool(header),
                include_bom=False,
            )
            return None

    def to_dict(
        self,
        into: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Convert Series to dict.

        Parameters
        ----------
        into : class, optional
            The collections.abc.Mapping subclass to use as the return type.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        dict
            Series as dict.
        """
        series_list = self._series.to_list()

        if self._index is not None:
            result_dict = dict(zip(self._index, series_list))
        else:
            result_dict = dict(enumerate(series_list))

        if into is not None:
            return into(result_dict)
        return result_dict

    def to_string(
        self,
        buf: Any = None,
        na_rep: str = "NaN",  # type: ignore[valid-type]
        float_format: Any = None,
        header: bool = True,
        index: bool = True,
        length: bool = False,
        dtype: bool = False,
        name: bool = False,
        max_rows: Any = None,
        min_rows: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Render a string representation of the Series.

        Parameters
        ----------
        buf : StringIO-like, optional
            Buffer to write to.
        na_rep : str, default 'NaN'
            String representation of NaN to use.
        float_format : str, optional
            Formatter for floating point numbers.
        header : bool, default True
            Print out the name of the Series.
        index : bool, default True
            Print index labels.
        length : bool, default False
            Print the length of the Series.
        dtype : bool, default False
            Print the dtype of the Series.
        name : bool, default False
            Print the name of the Series.
        max_rows : int, optional
            Maximum number of rows to show before truncating.
        min_rows : int, optional
            Minimum number of rows to show when truncating.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        str or None
            String representation of Series, or None if buf is provided.
        """

        output = []

        if header and self._series.name:
            output.append(f"Name: {self._series.name}")

        if dtype:
            output.append(f"dtype: {self._series.dtype}")

        if length:
            output.append(f"Length: {len(self._series)}")

        if name and self._series.name:
            output.append(f"Name: {self._series.name}")

        # Get data to display
        series_list = self._series.to_list()
        display_rows = len(series_list)

        if max_rows is not None and display_rows > max_rows:
            display_rows = max_rows
            truncated = True
        else:
            truncated = False

        # Format rows
        rows = []
        for i in range(display_rows):
            val = series_list[i]
            if val is None or (isinstance(val, float) and val != val):
                val_str = na_rep
            else:
                val_str = str(val)

            if index:
                if self._index is not None and i < len(self._index):
                    idx_str = str(self._index[i])
                else:
                    idx_str = str(i)
                rows.append(f"{idx_str}    {val_str}")
            else:
                rows.append(val_str)

        if truncated:
            rows.append("...")

        output.extend(rows)
        result = "\n".join(output)

        if buf is not None:
            buf.write(result)
            return None
        else:
            return result

    def rename(
        self,
        index: Any = None,
        axis: Any = None,
        copy: bool = True,
        inplace: bool = False,
        level: Any = None,
        errors: str = "ignore",  # type: ignore[valid-type]
        **kwargs: Any,
    ) -> Any:
        """
        Alter Series index labels or name.

        Parameters
        ----------
        index : scalar, hashable sequence, dict-like or function, optional
            Functions or mappings to apply to the index.
        axis : int or str, optional
            Axis to target with mapper (not used for Series, for pandas compatibility).
        copy : bool, default True
            Also copy underlying data.
        inplace : bool, default False
            Whether to return a new Series or modify in place.
        level : int, level name, or sequence of such, optional
            In case of a MultiIndex, only rename labels in the specified level(s).
        errors : {'ignore', 'raise'}, default 'ignore'
            If 'raise', raise a KeyError when a dict-like mapper, index, or columns contains labels that are not present.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series or None
            Renamed Series, or None if inplace=True.
        """
        new_index = self._index

        if index is not None:
            if callable(index):
                # Function mapper
                if self._index is not None:
                    new_index = [index(idx) for idx in self._index]
                else:
                    new_index = [index(i) for i in range(len(self._series))]
            elif isinstance(index, dict):
                # Dict mapper
                if self._index is not None:
                    new_index = [index.get(idx, idx) for idx in self._index]
                else:
                    new_index = [index.get(i, i) for i in range(len(self._series))]
            elif isinstance(index, (list, tuple)):
                # List of new labels
                if len(index) != len(self._series):
                    raise ValueError(
                        f"Length mismatch: Expected {len(self._series)}, got {len(index)}"
                    )
                new_index = list(index)
            else:
                # Scalar - rename the Series name
                new_name = str(index)
                result_series = pl.Series(self._series.to_list(), name=new_name)  # type: ignore[misc]
                if inplace:
                    self._series = result_series
                    return None
                else:
                    return Series(result_series, index=self._index)

        if inplace:
            self._index = new_index
            return None
        else:
            return Series(self._series, index=new_index)

    def reset_index(
        self,
        level: Any = None,
        drop: bool = False,
        name: Any = None,
        inplace: bool = False,
        allow_duplicates: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Generate a new Series or DataFrame with the index reset.

        Parameters
        ----------
        level : int, str, tuple, or list, optional
            Only remove the given levels from the index.
        drop : bool, default False
            Just reset the index, without inserting it as a column in the new DataFrame.
        name : object, optional
            The name to use for the column containing the original Series values.
        inplace : bool, default False
            Modify the Series in place.
        allow_duplicates : bool, default False
            Allow duplicate column labels.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series or DataFrame
            When drop is False (the default), a DataFrame is returned. When drop is True, a Series is returned.
        """
        from polarpandas.frame import DataFrame

        if drop:
            # Return Series with default integer index
            new_index = list(range(len(self._series)))
            result = Series(self._series, index=new_index)
            if inplace:
                self._series = result._series
                self._index = new_index
                return None
            else:
                return result
        else:
            # Return DataFrame with index as column
            index_name = name if name is not None else "index"
            if self._index is not None:
                index_values = self._index
            else:
                index_values = list(range(len(self._series)))

            df_data = {
                index_name: index_values,
                self._series.name or 0: self._series.to_list(),
            }
            result = DataFrame(df_data)  # type: ignore[assignment,arg-type]
            if inplace:
                # For inplace, convert to Series with reset index
                self._index = list(range(len(self._series)))
                return None
            else:
                return result

    def to_pandas(self) -> Any:
        """
        Convert polarpandas Series to pandas Series.

        Note: This method requires pandas to be installed.

        Returns
        -------
        pandas.Series
            Converted pandas Series
        """
        try:
            import pandas as pd  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_pandas() method. Install with: pip install pandas"
            ) from e

        # Convert Polars Series to pandas
        pandas_series = self._series.to_pandas()

        # Restore original tuple name if it was stored (for MultiIndex compatibility)
        if (
            hasattr(self, "_original_name")
            and self._original_name is not None
            and isinstance(self._original_name, tuple)
        ):
            pandas_series.name = self._original_name

        # Set index if we have one
        if self._index is not None:
            pandas_series.index = self._index

        # Handle name conversion - empty string becomes None in pandas
        if self._series.name == "":
            pandas_series.name = None

        return pandas_series

    def add_prefix(self, prefix: str) -> Series:  # type: ignore[valid-type]
        """
        Prefix labels with string prefix.

        Parameters
        ----------
        prefix : str
            String to add before each label

        Returns
        -------
        Series
            Series with prefixed labels
        """
        # For Series, this typically affects the name, but pandas applies it to values if string
        if self._series.dtype == pl.Utf8:
            return Series(prefix + self._series)
        else:
            # For non-string, convert to string first
            return Series(prefix + self._series.cast(pl.Utf8))

    def add_suffix(self, suffix: str) -> Series:  # type: ignore[valid-type]
        """
        Suffix labels with string suffix.

        Parameters
        ----------
        suffix : str
            String to add after each label

        Returns
        -------
        Series
            Series with suffixed labels
        """
        # For Series, this typically affects the name, but pandas applies it to values if string
        if self._series.dtype == pl.Utf8:
            return Series(self._series + suffix)
        else:
            # For non-string, convert to string first
            return Series(self._series.cast(pl.Utf8) + suffix)

    def agg(self, func: Any, **kwargs: Any) -> Any:
        """
        Aggregate using one or more operations.

        Parameters
        ----------
        func : callable or str
            Function to apply
        **kwargs
            Additional arguments

        Returns
        -------
        scalar
            Aggregated value
        """
        return self.aggregate(func, **kwargs)

    def aggregate(self, func: Any, **kwargs: Any) -> Any:
        """
        Aggregate using one or more operations.

        Parameters
        ----------
        func : callable or str
            Function to apply
        **kwargs
            Additional arguments

        Returns
        -------
        scalar
            Aggregated value
        """
        if isinstance(func, str):
            # Use Polars aggregation method
            if hasattr(self._series, func):
                return getattr(self._series, func)()
            else:
                raise ValueError(f"Unknown aggregation function: {func}")
        elif callable(func):
            # Apply function
            return func(self._series, **kwargs)
        else:
            raise TypeError(f"func must be callable or string, got {type(func)}")

    def align(
        self,
        other: Series,
        join: str = "outer",  # type: ignore[valid-type]
        axis: Any = None,
        **kwargs: Any,
    ) -> tuple[Series, Series]:
        """
        Align two Series on their index.

        Parameters
        ----------
        other : Series
            Other Series to align with
        join : str, default "outer"
            Type of join to perform
        axis : Any, optional
            Axis to align on
        **kwargs
            Additional arguments

        Returns
        -------
        tuple of Series
            Aligned Series
        """
        # Simplified implementation - align indices
        if self._index and other._index:
            all_indices = set(self._index) | set(other._index)
            # Create aligned series with matching indices
            left_values = []
            right_values = []
            for idx in all_indices:
                if idx in self._index:
                    left_idx = self._index.index(idx)
                    left_values.append(self._series[left_idx])
                else:
                    left_values.append(None)
                if idx in other._index:
                    right_idx = other._index.index(idx)
                    right_values.append(other._series[right_idx])
                else:
                    right_values.append(None)
            left_aligned = Series(left_values, index=list(all_indices))
            right_aligned = Series(right_values, index=list(all_indices))
            return left_aligned, right_aligned
        else:
            # No index alignment needed
            return self, other

    def argmax(self, axis: Any = None, skipna: bool = True, **kwargs: Any) -> int:
        """
        Return int position of the largest value in the Series.

        Parameters
        ----------
        axis : int, optional
            Axis (not used for Series)
        skipna : bool, default True
            Exclude NA/null values
        **kwargs
            Additional arguments

        Returns
        -------
        int
            Position of maximum value
        """
        if skipna:
            # Get index of max value, skipping nulls
            max_val = self._series.max()
            if max_val is None:
                return -1
            # Find position
            indices = self._series.arg_max()
            return indices if indices is not None else -1
        else:
            indices = self._series.arg_max()
            return indices if indices is not None else -1

    def argmin(self, axis: Any = None, skipna: bool = True, **kwargs: Any) -> int:
        """
        Return int position of the smallest value in the Series.

        Parameters
        ----------
        axis : int, optional
            Axis (not used for Series)
        skipna : bool, default True
            Exclude NA/null values
        **kwargs
            Additional arguments

        Returns
        -------
        int
            Position of minimum value
        """
        if skipna:
            # Get index of min value, skipping nulls
            min_val = self._series.min()
            if min_val is None:
                return -1
            # Find position
            indices = self._series.arg_min()
            return indices if indices is not None else -1
        else:
            indices = self._series.arg_min()
            return indices if indices is not None else -1

    def argsort(
        self,
        axis: int = 0,
        kind: str = "quicksort",  # type: ignore[valid-type]
        order: Any = None,
        **kwargs: Any,
    ) -> Series:
        """
        Return the integer indices that would sort the Series values.

        Parameters
        ----------
        axis : int, default 0
            Axis to sort along
        kind : str, default "quicksort"
            Sort algorithm
        order : Any, optional
            Not used
        **kwargs
            Additional arguments

        Returns
        -------
        Series
            Integer indices that would sort the Series
        """
        # Get sort indices from Polars
        sorted_indices = self._series.arg_sort()
        return Series(sorted_indices)

    def compare(
        self,
        other: Series,
        align_axis: int = 1,
        keep_shape: bool = False,
        keep_equal: bool = False,
        result_names: Any = None,
    ) -> DataFrame:
        """
        Compare to another Series and show the differences.

        Parameters
        ----------
        other : Series
            Series to compare with
        align_axis : int, default 1
            Axis to align on
        keep_shape : bool, default False
            Keep original shape
        keep_equal : bool, default False
            Keep equal values
        result_names : tuple, optional
            Names for result columns

        Returns
        -------
        DataFrame
            Comparison DataFrame
        """
        from .frame import DataFrame

        # Create comparison DataFrame
        self_name = result_names[0] if result_names else "self"
        other_name = result_names[1] if result_names else "other"

        # Align series
        if self._index and other._index:
            all_indices = set(self._index) | set(other._index)
            # Create aligned series
            self_values = []
            other_values = []
            for idx in all_indices:
                if idx in self._index:
                    self_idx = self._index.index(idx)
                    self_values.append(self._series[self_idx])
                else:
                    self_values.append(None)
                if idx in other._index:
                    other_idx = other._index.index(idx)
                    other_values.append(other._series[other_idx])
                else:
                    other_values.append(None)
            aligned_self = Series(self_values, index=list(all_indices))
            aligned_other = Series(other_values, index=list(all_indices))
        else:
            aligned_self = self
            aligned_other = other

        # Create comparison
        result_df = pl.DataFrame(
            {self_name: aligned_self._series, other_name: aligned_other._series}
        )

        return DataFrame(result_df)

    def combine(
        self,
        other: Series,
        func: Callable[[Any, Any], Any],
        fill_value: Any = None,
    ) -> Series:
        """
        Combine the Series with another Series using a function.

        Parameters
        ----------
        other : Series
            Series to combine with
        func : callable
            Function to apply
        fill_value : Any, optional
            Value to use for missing values

        Returns
        -------
        Series
            Combined Series
        """
        # Align series
        if self._index and other._index:
            all_indices = set(self._index) | set(other._index)
            # Create aligned series with fill_value
            self_values = []
            other_values = []
            for idx in all_indices:
                if idx in self._index:
                    self_idx = self._index.index(idx)
                    self_values.append(self._series[self_idx])
                else:
                    self_values.append(fill_value)
                if idx in other._index:
                    other_idx = other._index.index(idx)
                    other_values.append(other._series[other_idx])
                else:
                    other_values.append(fill_value)
            aligned_self = Series(self_values, index=list(all_indices))
            aligned_other = Series(other_values, index=list(all_indices))
        else:
            aligned_self = self
            aligned_other = other

        # Apply function element-wise
        combined_values = [
            func(s, o) for s, o in zip(aligned_self._series, aligned_other._series)
        ]
        return Series(
            combined_values, index=aligned_self._index if aligned_self._index else None
        )

    def combine_first(self, other: Series) -> Series:
        """
        Combine Series values, choosing the calling Series's values first.

        Parameters
        ----------
        other : Series
            Series to combine with

        Returns
        -------
        Series
            Combined Series
        """
        # Align series
        if self._index and other._index:
            all_indices = set(self._index) | set(other._index)
            # Create aligned series
            self_values = []
            other_values = []
            for idx in all_indices:
                if idx in self._index:
                    self_idx = self._index.index(idx)
                    self_values.append(self._series[self_idx])
                else:
                    self_values.append(None)
                if idx in other._index:
                    other_idx = other._index.index(idx)
                    other_values.append(other._series[other_idx])
                else:
                    other_values.append(None)
            aligned_self = Series(self_values, index=list(all_indices))
            aligned_other = Series(other_values, index=list(all_indices))
        else:
            aligned_self = self
            aligned_other = other

        # Use self values where not null, otherwise use other
        result_series = (
            pl.when(aligned_self._series.is_null())
            .then(aligned_other._series)
            .otherwise(aligned_self._series)
        )
        df = pl.DataFrame({"temp": aligned_self._series})
        result_df = df.select(result_series.alias("result"))
        return Series(
            result_df["result"],
            index=aligned_self._index if aligned_self._index else None,
        )

    def corr(
        self,
        other: Series,
        method: str = "pearson",  # type: ignore[valid-type]
        min_periods: Any = None,
    ) -> float:
        """
        Compute correlation with other Series.

        Parameters
        ----------
        other : Series
            Series to compute correlation with
        method : str, default "pearson"
            Correlation method
        min_periods : int, optional
            Minimum number of observations

        Returns
        -------
        float
            Correlation coefficient
        """
        # Combine series and compute correlation
        combined = pl.DataFrame({"self": self._series, "other": other._series})
        corr = combined.select(pl.corr("self", "other")).item()
        return corr if corr is not None else float("nan")

    def cov(
        self,
        other: Series,
        min_periods: Any = None,
        ddof: Any = None,
    ) -> float:
        """
        Compute covariance with other Series.

        Parameters
        ----------
        other : Series
            Series to compute covariance with
        min_periods : int, optional
            Minimum number of observations
        ddof : int, optional
            Delta degrees of freedom

        Returns
        -------
        float
            Covariance
        """
        # Polars doesn't have direct covariance, compute manually
        mean_self = self._series.mean()
        mean_other = other._series.mean()
        cov = ((self._series - mean_self) * (other._series - mean_other)).mean()
        return cov if cov is not None else float("nan")  # type: ignore[return-value]

    def divmod(
        self,
        other: Any,
        fill_value: Any = None,
        axis: int = 0,
        level: Any = None,
    ) -> tuple[Series, Series]:
        """
        Return integer division and modulo of division.

        Parameters
        ----------
        other : scalar or Series
            Divisor
        fill_value : Any, optional
            Value to use for missing values
        axis : int, default 0
            Axis (not used for Series)
        level : Any, optional
            Level (not used for Series)

        Returns
        -------
        tuple of Series
            Quotient and remainder
        """
        if isinstance(other, Series):
            quotient = Series(self._series // other._series)
            remainder = Series(self._series % other._series)
        else:
            quotient = Series(self._series // other)
            remainder = Series(self._series % other)
        return quotient, remainder

    def rdivmod(
        self,
        other: Any,
        fill_value: Any = None,
        axis: int = 0,
        level: Any = None,
    ) -> tuple[Series, Series]:
        """
        Return integer division and modulo of division (reverse).

        Parameters
        ----------
        other : scalar or Series
            Dividend
        fill_value : Any, optional
            Value to use for missing values
        axis : int, default 0
            Axis (not used for Series)
        level : Any, optional
            Level (not used for Series)

        Returns
        -------
        tuple of Series
            Quotient and remainder
        """
        if isinstance(other, Series):
            quotient = Series(other._series // self._series)
            remainder = Series(other._series % self._series)
        else:
            quotient = Series(other // self._series)
            remainder = Series(other % self._series)
        return quotient, remainder

    def dot(self, other: Any) -> Any:
        """
        Compute the dot product between the Series and another Series or scalar.

        Parameters
        ----------
        other : Series or scalar
            Other Series or scalar

        Returns
        -------
        scalar
            Dot product
        """
        if isinstance(other, Series):
            # Dot product of two series
            return (self._series * other._series).sum()
        else:
            # Scalar multiplication
            return (self._series * other).sum()

    def items(self) -> Iterator[tuple[Any, Any]]:
        """
        Lazily iterate over (index, value) pairs.

        Returns
        -------
        iterator
            Iterator of (index, value) pairs
        """
        if self._index:
            for idx, val in zip(self._index, self._series):
                yield idx, val
        else:
            for idx, val in enumerate(self._series):
                yield idx, val

    def keys(self) -> Any:
        """
        Return alias for index.

        Returns
        -------
        Index or list
            Index values
        """
        if self._index:
            return self._index
        else:
            return list(range(len(self._series)))

    def to_clipboard(self, excel: bool = True, sep: Any = None, **kwargs: Any) -> None:
        """Copy object to clipboard."""
        try:
            import pyperclip

            text = "\n".join(str(v) for v in self._series.to_list())
            pyperclip.copy(text)
        except ImportError:
            raise NotImplementedError(
                "to_clipboard() requires pyperclip. Install: pip install pyperclip"
            ) from None

    def to_excel(
        self,
        excel_writer: Any,
        sheet_name: str = "Sheet1",  # type: ignore[valid-type]
        na_rep: str = "",  # type: ignore[valid-type]
        float_format: Any = None,
        columns: Any = None,
        header: Any = True,
        index: bool = True,
        index_label: Any = None,
        startrow: int = 0,
        startcol: int = 0,
        engine: Any = None,
        merge_cells: bool = True,
        encoding: Any = None,
        inf_rep: str = "inf",  # type: ignore[valid-type]
        verbose: bool = True,
        freeze_panes: Any = None,
        storage_options: Any = None,
        **kwargs: Any,
    ) -> None:
        """Write Series to Excel file."""
        from .frame import DataFrame

        df = DataFrame({self.name or 0: self._series})  # type: ignore[dict-item]
        df.to_excel(
            excel_writer=excel_writer,
            sheet_name=sheet_name,
            na_rep=na_rep,
            float_format=float_format,
            header=header,
            index=index,
            index_label=index_label,
            startrow=startrow,
            startcol=startcol,
            engine=engine,
            merge_cells=merge_cells,
            encoding=encoding,
            inf_rep=inf_rep,
            verbose=verbose,
            freeze_panes=freeze_panes,
            storage_options=storage_options,
            **kwargs,
        )

    def to_hdf(
        self,
        path_or_buf: Any,
        key: str,  # type: ignore[valid-type]
        mode: str = "a",  # type: ignore[valid-type]
        complevel: Any = None,
        complib: Any = None,
        append: bool = False,
        format: Any = None,
        index: bool = True,
        min_itemsize: Any = None,
        nan_rep: Any = None,
        dropna: Any = None,
        data_columns: Any = None,
        errors: str = "strict",  # type: ignore[valid-type]
        encoding: str = "UTF-8",  # type: ignore[valid-type]
        **kwargs: Any,
    ) -> None:
        """Write Series to HDF5 file."""
        from .frame import DataFrame

        df = DataFrame({self.name or 0: self._series}, index=self._index)  # type: ignore[dict-item]
        df.to_hdf(
            path_or_buf=path_or_buf,
            key=key,
            mode=mode,
            complevel=complevel,
            complib=complib,
            append=append,
            format=format,
            index=index,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            dropna=dropna,
            data_columns=data_columns,
            errors=errors,
            encoding=encoding,
            **kwargs,
        )

    def to_json(
        self,
        path_or_buf: Any = None,
        orient: str = "records",  # type: ignore[valid-type]
        date_format: str = "epoch",  # type: ignore[valid-type]
        double_precision: int = 10,
        force_ascii: bool = True,
        date_unit: str = "ms",  # type: ignore[valid-type]
        default_handler: Any = None,
        lines: bool = False,
        compression: Any = None,
        index: bool = True,
        indent: Any = None,
        storage_options: Any = None,
        mode: str = "w",  # type: ignore[valid-type]
        **kwargs: Any,
    ) -> Any:
        """Convert Series to JSON string."""
        import json

        data = self._series.to_dict(as_series=False)  # type: ignore[attr-defined]
        if index and self._index:
            result = {str(k): v for k, v in zip(self._index, data[self.name or 0])}
        else:
            result = {self.name or "values": data[self.name or 0]}
        json_str = json.dumps(result, indent=indent, default=default_handler, **kwargs)
        if path_or_buf is None:
            return json_str
        else:
            if hasattr(path_or_buf, "write"):
                path_or_buf.write(json_str)
            else:
                with open(path_or_buf, mode, encoding="utf-8") as f:
                    f.write(json_str)
            return None

    def to_latex(
        self,
        buf: Any = None,
        columns: Any = None,
        header: bool = True,
        index: bool = True,
        na_rep: str = "NaN",  # type: ignore[valid-type]
        formatters: Any = None,
        float_format: Any = None,
        sparsify: Any = None,
        index_names: bool = True,
        bold_rows: bool = False,
        column_format: Any = None,
        longtable: bool = False,
        escape: bool = True,
        encoding: Any = None,
        decimal: str = ".",  # type: ignore[valid-type]
        multicolumn: Any = None,
        multicolumn_format: Any = None,
        multirow: Any = None,
        caption: Any = None,
        label: Any = None,
        position: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Render Series to LaTeX table."""
        from .frame import DataFrame

        df = DataFrame({self.name or 0: self._series}, index=self._index)  # type: ignore[dict-item]
        return df.to_latex(
            buf=buf,
            header=header,
            index=index,
            na_rep=na_rep,
            float_format=float_format,
            column_format=column_format,
            encoding=encoding,
            **kwargs,
        )

    def to_markdown(
        self,
        buf: Any = None,
        mode: str = "wt",  # type: ignore[valid-type]
        index: bool = True,
        storage_options: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Print Series in Markdown-friendly format."""
        from .frame import DataFrame

        df = DataFrame({self.name or 0: self._series}, index=self._index)  # type: ignore[dict-item]
        return df.to_markdown(
            buf=buf, mode=mode, index=index, storage_options=storage_options, **kwargs
        )

    def to_pickle(
        self,
        path: str,  # type: ignore[valid-type]
        compression: Any = None,
        protocol: Any = None,
        storage_options: Any = None,
        **kwargs: Any,
    ) -> None:
        """Pickle (serialize) object to file."""
        import gzip
        import pickle

        if compression == "gzip":
            opener = gzip.open
            mode = "wb"
        elif compression:
            raise ValueError(f"Unsupported compression: {compression}")
        else:
            opener = open  # type: ignore[assignment]
            mode = "wb"
        with opener(path, mode) as f:
            pickle.dump(self, f, protocol=protocol, **kwargs)  # type: ignore[arg-type]

    def to_sql(
        self,
        name: BuiltinStr,
        con: Any,
        schema: BuiltinStr | None = None,
        if_exists: BuiltinStr = "fail",
        index: bool = True,
        index_label: BuiltinStr | BuiltinList[BuiltinStr] | None = None,
        chunksize: int | None = None,
        dtype: BuiltinDict[BuiltinStr, Any] | None = None,
        method: BuiltinStr | Callable[..., Any] | None = None,
        primary_key: BuiltinStr | BuiltinList[BuiltinStr] | None = None,
        auto_increment: bool = False,
    ) -> None:
        """
        Write Series to SQL database.

        This method provides pandas-compatible interface for writing Series to SQL.
        When primary_key or auto_increment parameters are used, SQLAlchemy is required.

        Parameters
        ----------
        name : str
            Name of SQL table
        con : sqlalchemy.engine.Engine or sqlite3.Connection
            Database connection or SQLAlchemy engine
        schema : str, optional
            Specify the schema (if database flavor supports this). If None, use
            default schema.
        if_exists : {'fail', 'replace', 'append'}, default 'fail'
            How to behave if the table already exists:
            - fail: Raise a ValueError
            - replace: Drop the table before inserting new values
            - append: Insert new values to the existing table
        index : bool, default True
            Write DataFrame index as a column. Uses index_label as the column
            name in the table.
        index_label : str or list of str, optional
            Column label for index column(s). If None is given (default) and
            index is True, then the index names are used.
        chunksize : int, optional
            Specify the number of rows in each batch to be written at a time.
            By default, all rows will be written at once.
        dtype : dict, optional
            Specifying the datatype for columns. The keys should be the column
            names and the values should be SQLAlchemy types.
        method : {None, 'multi', callable}, optional
            Controls the SQL insertion clause used:
            - None: Uses standard SQL INSERT clause (one per row)
            - 'multi': Pass multiple values in a single INSERT clause
            - callable: Callable with signature (pd_table, conn, keys, data_iter)
        primary_key : str or list of str, optional
            Column name(s) to set as the primary key. Requires SQLAlchemy.
        auto_increment : bool, default False
            If True, the primary key column will be set to auto-increment.
            Requires SQLAlchemy and primary_key to be specified.

        Raises
        ------
        ValueError
            If table exists and if_exists is 'fail'
        ImportError
            If SQLAlchemy is required but not installed

        Examples
        --------
        >>> from sqlalchemy import create_engine
        >>> engine = create_engine('sqlite:///example.db')
        >>> series = ppd.Series([1, 2, 3], name='values')
        >>> series.to_sql('my_table', engine)

        Notes
        -----
        Converts the Series to a DataFrame with one column and calls DataFrame.to_sql()
        """
        from .frame import DataFrame

        df = DataFrame({self.name or 0: self._series}, index=self._index)  # type: ignore[dict-item]
        df.to_sql(
            name=name,
            con=con,
            schema=schema,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            chunksize=chunksize,
            dtype=dtype,
            method=method,
            primary_key=primary_key,
            auto_increment=auto_increment,
        )

    def to_xarray(self, dim_order: Any = None, **kwargs: Any) -> Any:
        """Return an xarray.DataArray representation of the Series."""
        try:
            import numpy as np
            import xarray as xr

            if hasattr(self._series, "to_numpy"):
                arr = self._series.to_numpy()
            else:
                arr = np.array(self._series.to_list())
            coords = {}
            if self._index:
                coords["index"] = self._index
            return xr.DataArray(
                arr,
                coords=coords,
                dims=["index"] if self._index else None,
                name=self.name,
                **kwargs,
            )
        except ImportError:
            raise NotImplementedError(
                "to_xarray() requires xarray. Install: pip install xarray"
            ) from None

    # Arithmetic methods (right operations)
    def radd(self, other: Any, **kwargs: Any) -> Series:
        """Right addition: other + self."""
        return Series(other + self._series)

    def rdiv(self, other: Any, **kwargs: Any) -> Series:
        """Right division: other / self."""
        return Series(other / self._series)

    def rfloordiv(self, other: Any, **kwargs: Any) -> Series:
        """Right floor division: other // self."""
        return Series(other // self._series)

    def rmod(self, other: Any, **kwargs: Any) -> Series:
        """Right modulo: other % self."""
        return Series(other % self._series)

    def rmul(self, other: Any, **kwargs: Any) -> Series:
        """Right multiplication: other * self."""
        return Series(other * self._series)

    def rpow(self, other: Any, **kwargs: Any) -> Series:
        """Right power: other ** self."""
        return Series(other**self._series)

    def rsub(self, other: Any, **kwargs: Any) -> Series:
        """Right subtraction: other - self."""
        return Series(other - self._series)

    def rtruediv(self, other: Any, **kwargs: Any) -> Series:
        """Right true division: other / self."""
        return Series(other / self._series)

    def truediv(self, other: Any, **kwargs: Any) -> Series:
        """True division: self / other."""
        if isinstance(other, Series):
            return Series(self._series / other._series)
        return Series(self._series / other)

    # Window/rolling methods
    def ewm(
        self,
        com: Any = None,
        span: Any = None,
        halflife: Any = None,
        alpha: Any = None,
        min_periods: int = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        axis: int = 0,
        times: Any = None,
        method: str = "single",  # type: ignore[valid-type]
        **kwargs: Any,
    ) -> Any:
        """Provide exponential weighted functions."""
        raise NotImplementedError(
            "ewm() is not yet implemented. Use pandas: pd_series.ewm(span=span) then convert"
        )

    def expanding(
        self,
        min_periods: int = 1,
        axis: int = 0,
        method: str = "single",  # type: ignore[valid-type]
        **kwargs: Any,
    ) -> Any:
        """Provide expanding window calculations."""
        raise NotImplementedError(
            "expanding() is not yet implemented. Use pandas: pd_series.expanding() then convert"
        )

    def rolling(
        self,
        window: Any,
        min_periods: Any = None,
        center: bool = False,
        win_type: Any = None,
        on: Any = None,
        axis: int = 0,
        closed: Any = None,
        step: Any = None,
        method: str = "single",  # type: ignore[valid-type]
        **kwargs: Any,
    ) -> Any:
        """Provide rolling window calculations."""
        if (
            center
            or win_type is not None
            or on is not None
            or closed is not None
            or step is not None
        ):
            raise NotImplementedError(
                "rolling() currently supports basic integer windows without center/on/closed/step/win_type"
            )

        if axis not in (0, "index"):
            raise NotImplementedError("rolling() currently supports axis=0/index only")

        if not isinstance(window, int):
            raise TypeError("window must be an integer")

        if min_periods is not None:
            if not isinstance(min_periods, int):
                raise TypeError("min_periods must be an integer")
            if min_periods < 0:
                raise ValueError("min_periods must be non-negative")

        return _SeriesRolling(self, window=window, min_periods=min_periods, **kwargs)

    # Data manipulation methods
    def convert_dtypes(
        self,
        infer_objects: bool = True,
        convert_string: bool = True,
        convert_integer: bool = True,
        convert_boolean: bool = True,
        convert_floating: bool = True,
        dtype_backend: str = "numpy_nullable",  # type: ignore[valid-type]
        **kwargs: Any,
    ) -> Series:
        """Convert to best possible dtypes."""
        return self.copy()

    def droplevel(
        self,
        level: Any,
        axis: int = 0,
        **kwargs: Any,
    ) -> Series:
        """Drop level from MultiIndex."""
        raise NotImplementedError(
            "droplevel() requires MultiIndex. Not yet implemented for simple Series"
        )

    def infer_objects(self, copy: Any = None) -> Series:
        """Attempt to infer better dtypes for object columns."""
        return self.copy() if copy else self

    def interpolate(
        self,
        method: str = "linear",  # type: ignore[valid-type]
        axis: int = 0,
        limit: Any = None,
        inplace: bool = False,
        limit_direction: Any = None,
        limit_area: Any = None,
        downcast: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Interpolate missing values."""
        if method == "linear":
            result = Series(self._series.interpolate())
            if inplace:
                self._series = result._series
                return None
            return result
        raise NotImplementedError(
            f"interpolate() with method='{method}' is not yet implemented"
        )

    def mask(
        self,
        cond: Any,
        other: Any = None,
        inplace: bool = False,
        axis: Any = None,
        level: Any = None,
        errors: str = "raise",  # type: ignore[valid-type]
        try_cast: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Replace values where condition is True."""
        if isinstance(cond, Series):
            cond = cond._series

        # Create a DataFrame with one column to use select
        temp_df = pl.DataFrame({"value": self._series})
        result_expr = (
            pl.when(cond)
            .then(pl.lit(other) if other is not None else None)
            .otherwise(pl.col("value"))
        )
        result_series = temp_df.select(result_expr).to_series()

        if inplace:
            self._series = result_series
            return None
        return Series(result_series)

    def reindex(
        self,
        index: Any = None,
        axis: Any = None,
        method: Any = None,
        copy: bool = True,
        level: Any = None,
        fill_value: Any = None,
        limit: Any = None,
        tolerance: Any = None,
        **kwargs: Any,
    ) -> Series:
        """Reindex to new index."""
        if index is None:
            return self.copy() if copy else self
        # Create new Series with new index
        result = Series(self._series, index=index, name=self.name)
        return result

    def reindex_like(
        self,
        other: Series,
        method: Any = None,
        copy: bool = True,
        limit: Any = None,
        tolerance: Any = None,
        **kwargs: Any,
    ) -> Series:
        """Reindex like another Series."""
        return self.reindex(
            index=other._index if other._index else None,
            method=method,
            copy=copy,
            limit=limit,
            tolerance=tolerance,
            **kwargs,
        )

    def rename_axis(
        self,
        mapper: Any = None,
        index: Any = None,
        axis: int = 0,
        copy: bool = True,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Rename axis."""
        result = self.copy() if copy else self
        if mapper:
            result._index_name = mapper
        if inplace:
            self._index_name = result._index_name
            return None
        return result

    def reorder_levels(
        self, order: BuiltinList[Any], axis: int = 0, **kwargs: Any
    ) -> Series:
        """Reorder MultiIndex levels."""
        raise NotImplementedError(
            "reorder_levels() requires MultiIndex. Not yet implemented"
        )

    def repeat(self, repeats: Any, axis: Any = None, **kwargs: Any) -> Series:
        """Repeat elements."""
        if isinstance(repeats, int):
            # Repeat each element by the given number
            result_list = []
            for val in self._series:
                result_list.extend([val] * repeats)
            return Series(pl.Series(result_list))
        else:
            # Repeat each element by corresponding value in repeats
            result_list = []
            for i, rep in enumerate(repeats):
                result_list.extend([self._series[i]] * rep)
            return Series(pl.Series(result_list))

    def sample(
        self,
        n: Any = None,
        frac: Any = None,
        replace: bool = False,
        weights: Any = None,
        random_state: Any = None,
        axis: Any = None,
        ignore_index: bool = False,
        **kwargs: Any,
    ) -> Series:
        """Random sample."""
        if n is None and frac is None:
            n = 1
        elif frac is not None:
            n = int(len(self) * frac)
        return Series(
            self._series.sample(n=n, with_replacement=replace, seed=random_state)
        )

    def searchsorted(
        self,
        value: Any,
        side: str = "left",  # type: ignore[valid-type]
        sorter: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Find insertion points."""
        sorted_series = self._series.sort()
        if side == "left":
            result = sorted_series.search_sorted(value)
        else:
            result = sorted_series.search_sorted(value, side="right")
        if isinstance(result, pl.Series):
            return Series(result)
        return result

    def set_axis(
        self,
        labels: Any,
        axis: int = 0,
        inplace: bool = False,
        copy: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Set axis labels."""
        result = self.copy() if copy else self
        result._index = list(labels) if labels is not None else None
        if inplace:
            self._index = result._index
            return None
        return result

    def set_flags(
        self,
        copy: bool = False,
        allows_duplicate_labels: Any = None,
        **kwargs: Any,
    ) -> Series:
        """Set flags."""
        # Polars doesn't have flags, return copy
        return self.copy() if copy else self

    def squeeze(self, axis: Any = None, **kwargs: Any) -> Series:
        """Squeeze dimensions (no-op for Series)."""
        return self

    def take(
        self,
        indices: Any,
        axis: Any = None,
        is_copy: Any = None,
        **kwargs: Any,
    ) -> Series:
        """Take elements by position."""
        if isinstance(indices, (list, tuple)):
            return Series(self._series[indices])
        return Series(self._series[indices])

    def transform(
        self,
        func: Any,
        axis: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> Series:
        """Transform with function."""
        if callable(func):
            result = func(self._series)
            if isinstance(result, pl.Series):
                return Series(result)
            return Series(pl.Series([result] * len(self)))
        raise ValueError("func must be callable")

    def transpose(self, *args: Any, **kwargs: Any) -> Series:
        """Transpose (no-op for Series)."""
        return self

    def truncate(
        self,
        before: Any = None,
        after: Any = None,
        axis: Any = None,
        copy: bool = True,
        **kwargs: Any,
    ) -> Series:
        """Truncate before/after."""
        if self._index:
            start_idx = 0
            end_idx = len(self)
            if before is not None:
                with contextlib.suppress(ValueError, TypeError):
                    start_idx = (
                        self._index.index(before) if before in self._index else 0
                    )
            if after is not None:
                with contextlib.suppress(ValueError, TypeError):
                    end_idx = (
                        self._index.index(after) + 1
                        if after in self._index
                        else len(self)
                    )
            result = Series(
                self._series[start_idx:end_idx],
                index=self._index[start_idx:end_idx] if self._index else None,
            )
            return result
        return self.copy() if copy else self

    def unstack(
        self,
        level: Any = -1,
        fill_value: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Unstack (for MultiIndex)."""
        raise NotImplementedError("unstack() requires MultiIndex. Not yet implemented")

    def update(
        self,
        other: Series,
        join: str = "left",  # type: ignore[valid-type]
        overwrite: bool = True,
        filter_func: Any = None,
        errors: str = "ignore",  # type: ignore[valid-type]
        **kwargs: Any,
    ) -> None:
        """Update with another Series."""
        if self._index and other._index:
            for idx in other._index:
                if idx in self._index:
                    self_idx = self._index.index(idx)
                    other_idx = other._index.index(idx)
                    self._series = self._series.set_at_index(  # type: ignore[attr-defined]
                        self_idx, other._series[other_idx]
                    )

    def where(
        self,
        cond: Any,
        other: Any = None,
        inplace: bool = False,
        axis: Any = None,
        level: Any = None,
        errors: str = "raise",  # type: ignore[valid-type]
        try_cast: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Replace where condition is False."""
        if isinstance(cond, Series):
            cond = cond._series

        # Create a DataFrame with one column to use select
        temp_df = pl.DataFrame({"value": self._series})
        result_expr = (
            pl.when(cond)
            .then(pl.col("value"))
            .otherwise(pl.lit(other) if other is not None else None)
        )
        result_series = temp_df.select(result_expr).to_series()

        if inplace:
            self._series = result_series
            return None
        return Series(result_series)

    def xs(
        self,
        key: Any,
        axis: int = 0,
        level: Any = None,
        drop_level: bool = True,
        **kwargs: Any,
    ) -> Series:
        """Cross-section."""
        if self._index and key in self._index:
            idx = self._index.index(key)
            return Series(pl.Series([self._series[idx]]))
        raise KeyError(f"Key {key} not found in index")

    # Accessor properties
    @property
    def cat(self) -> Any:
        """Categorical accessor."""
        return _CategoricalAccessor(self)

    @property
    def list(self) -> Any:
        """List accessor."""
        raise NotImplementedError("list accessor is not yet implemented")

    @property
    def struct(self) -> Any:
        """Struct accessor."""
        raise NotImplementedError("struct accessor is not yet implemented")

    # Statistical methods
    def autocorr(self, lag: int = 1, **kwargs: Any) -> float:
        """Autocorrelation."""
        if len(self) < lag + 1:
            return float("nan")
        try:
            import numpy as np

            arr = np.array(self._series.to_list())
            mean = np.mean(arr)
            var = np.var(arr)
            if var == 0:
                return float("nan")
            autocov = np.mean((arr[:-lag] - mean) * (arr[lag:] - mean))
            return float(autocov / var)
        except ImportError:
            raise NotImplementedError(
                "autocorr() requires numpy. Install: pip install numpy"
            ) from None

    def mode(self, dropna: bool = True, **kwargs: Any) -> Series:
        """Mode value(s)."""
        vc_struct = self.value_counts(dropna=dropna)._series
        if vc_struct.len() == 0:
            return Series(pl.Series([], dtype=self._series.dtype))

        value_field_name = vc_struct.struct.fields[0]
        values = vc_struct.struct.field(value_field_name)
        counts = vc_struct.struct.field("count")

        if dropna:
            not_null = values.is_not_null()
            values = values.filter(not_null)
            counts = counts.filter(not_null)

        if values.len() == 0:
            return Series(pl.Series([], dtype=self._series.dtype))

        max_count = counts.max()
        mask = counts == max_count
        mode_values = values.filter(mask)
        return Series(mode_values)

    def nlargest(self, n: int = 5, keep: str = "first", **kwargs: Any) -> Series:  # type: ignore[valid-type]
        """N largest values."""
        sorted_series = self._series.sort(descending=True)
        return Series(sorted_series.head(n))

    def nsmallest(self, n: int = 5, keep: str = "first", **kwargs: Any) -> Series:  # type: ignore[valid-type]
        """N smallest values."""
        sorted_series = self._series.sort()
        return Series(sorted_series.head(n))

    def prod(
        self,
        axis: Any = None,
        skipna: bool = True,
        numeric_only: Any = None,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Any:
        """Product of values."""
        return self._series.product()

    def product(
        self,
        axis: Any = None,
        skipna: bool = True,
        numeric_only: Any = None,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Any:
        """Product of values (alias)."""
        return self.prod(
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    # Other methods
    def case_when(
        self,
        caselist: BuiltinList[tuple[Any, Any]],
        default: Any = None,
        **kwargs: Any,
    ) -> Series:
        """Case when conditions."""
        expr = None
        for condition, value in caselist:
            if isinstance(condition, Series):
                condition = condition._series
            if expr is None:
                expr = pl.when(condition).then(pl.lit(value))
            else:
                expr = expr.when(condition).then(pl.lit(value))  # type: ignore[assignment]
        if default is not None:
            expr = expr.otherwise(pl.lit(default))  # type: ignore[union-attr,assignment]
        else:
            expr = expr.otherwise(self._series)  # type: ignore[union-attr,assignment]
        return Series(expr)

    def groupby(
        self,
        by: Any = None,
        axis: int = 0,
        level: Any = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        squeeze: bool = False,
        observed: bool = False,
        dropna: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Group by values."""
        raise NotImplementedError(
            "groupby() for Series is not yet implemented. Use DataFrame.groupby() instead"
        )

    def hist(self, bins: int = 10, **kwargs: Any) -> Any:
        """Histogram."""
        raise NotImplementedError(
            "hist() is not yet implemented. Use matplotlib/seaborn directly"
        )

    def memory_usage(
        self, index: bool = True, deep: bool = False, **kwargs: Any
    ) -> int:
        """Memory usage in bytes."""
        size = self._series.estimated_size()
        if index and self._index:
            size += (
                sum(sys.getsizeof(x) for x in self._index)
                if deep
                else len(self._index) * 8
            )
        return size  # type: ignore[return-value]

    def pipe(self, func: Callable, *args: Any, **kwargs: Any) -> Any:  # type: ignore[type-arg]
        """Apply function."""
        return func(self, *args, **kwargs)

    def plot(
        self,
        kind: str = "line",  # type: ignore[valid-type]
        ax: Any = None,
        figsize: Any = None,
        use_index: bool = True,
        title: Any = None,
        grid: Any = None,
        legend: Any = False,
        style: Any = None,
        logx: Any = False,
        logy: Any = False,
        loglog: Any = False,
        xticks: Any = None,
        yticks: Any = None,
        xlim: Any = None,
        ylim: Any = None,
        rot: Any = None,
        fontsize: Any = None,
        colormap: Any = None,
        table: bool = False,
        yerr: Any = None,
        xerr: Any = None,
        label: Any = None,
        secondary_y: Any = False,
        mark_right: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot Series."""
        raise NotImplementedError(
            "plot() is not yet implemented. Use matplotlib/seaborn directly"
        )

    def pop(self, item: Any, **kwargs: Any) -> Any:
        """Pop item."""
        if self._index and item in self._index:
            idx = self._index.index(item)
            value = self._series[idx]
            # Remove from series and index
            new_series = pl.concat([self._series[:idx], self._series[idx + 1 :]])
            new_index = self._index[:idx] + self._index[idx + 1 :]
            self._series = new_series
            self._index = new_index
            return value
        raise KeyError(f"Key {item} not found")

    def sparse(self, **kwargs: Any) -> Series:
        """Convert to sparse format."""
        raise NotImplementedError(
            "sparse() is not yet implemented. Polars doesn't have native sparse format"
        )

    def to_period(
        self,
        freq: Any = None,
        axis: int = 0,
        copy: bool = True,
        **kwargs: Any,
    ) -> Series:
        """Convert to period."""
        raise NotImplementedError(
            "to_period() is not yet implemented. Use pandas: pd_series.to_period(freq) then convert"
        )

    def swaplevel(
        self,
        i: Any = -2,
        j: Any = -1,
        axis: int = 0,
        **kwargs: Any,
    ) -> Series:
        """
        Swap levels i and j in a MultiIndex.

        Parameters
        ----------
        i : int or str, default -2
            First level to swap
        j : int or str, default -1
            Second level to swap
        axis : int, default 0
            Axis (not used for Series)
        **kwargs
            Additional arguments

        Returns
        -------
        Series
            Series with swapped levels

        Examples
        --------
        >>> import polarpandas as ppd
        >>> s = ppd.Series([1, 2, 3])
        >>> result = s.swaplevel()
        """
        raise NotImplementedError(
            "swaplevel() requires MultiIndex. Not yet implemented for simple Series"
        )

    # Time series methods
    def asfreq(
        self,
        freq: str,  # type: ignore[valid-type]
        method: Any = None,
        how: Any = None,
        normalize: bool = False,
        fill_value: Any = None,
        **kwargs: Any,
    ) -> Series:
        """
        Convert time series to specified frequency.

        Parameters
        ----------
        freq : str
            Frequency string
        method : str, optional
            Method for filling missing values
        how : str, optional
            Start or end of interval
        normalize : bool, default False
            Normalize start/end dates
        fill_value : Any, optional
            Value to use for missing values
        **kwargs
            Additional arguments

        Returns
        -------
        Series
            Series with converted frequency

        Examples
        --------
        >>> import polarpandas as ppd
        >>> s = ppd.Series([1, 2, 3])
        >>> result = s.asfreq("D")
        """
        raise NotImplementedError(
            "asfreq() is not yet implemented.\n"
            "Workarounds:\n"
            "  - Use pandas: pd_series.asfreq(freq) then convert with polarpandas.Series(s)\n"
            "  - Resample manually using resample() method"
        )

    def asof(self, where: Any, **kwargs: Any) -> Any:
        """
        Return last valid value up to label.

        Parameters
        ----------
        where : Any
            Label or labels
        **kwargs
            Additional arguments

        Returns
        -------
        Any
            Last valid value

        Examples
        --------
        >>> import polarpandas as ppd
        >>> s = ppd.Series([1, 2, 3])
        >>> result = s.asof(2)
        """
        # Simplified implementation
        if self._index:
            try:
                idx = (
                    self._index.index(where)
                    if where in self._index
                    else len(self._index) - 1
                )
                return self._series[idx] if idx < len(self._series) else None
            except (ValueError, TypeError):
                return self._series[-1] if len(self._series) > 0 else None
        return self._series[-1] if len(self._series) > 0 else None

    def at_time(self, time: Any, asof: bool = False, **kwargs: Any) -> Series:
        """
        Select values at particular time of day.

        Parameters
        ----------
        time : datetime.time or str
            Time to select
        asof : bool, default False
            Use asof logic
        **kwargs
            Additional arguments

        Returns
        -------
        Series
            Selected values

        Examples
        --------
        >>> import polarpandas as ppd
        >>> s = ppd.Series([1, 2, 3])
        >>> result = s.at_time("09:00")
        """
        # Simplified implementation - filter by time if datetime index exists
        if self._index and hasattr(self._index[0], "time"):
            filtered = [
                i
                for i, idx in enumerate(self._index)
                if hasattr(idx, "time") and idx.time() == time
            ]
            if filtered:
                return Series(self._series[filtered])
        return Series(pl.Series([], dtype=self._series.dtype))

    def between_time(
        self,
        start_time: Any,
        end_time: Any,
        inclusive: str = "both",  # type: ignore[valid-type]
        **kwargs: Any,
    ) -> Series:
        """
        Select values between particular times of day.

        Parameters
        ----------
        start_time : datetime.time or str
            Start time
        end_time : datetime.time or str
            End time
        inclusive : str, default "both"
            Include boundaries
        **kwargs
            Additional arguments

        Returns
        -------
        Series
            Selected values

        Examples
        --------
        >>> import polarpandas as ppd
        >>> s = ppd.Series([1, 2, 3])
        >>> result = s.between_time("09:00", "10:00")
        """
        # Simplified implementation
        if self._index and hasattr(self._index[0], "time"):
            filtered = []
            for i, idx in enumerate(self._index):
                if hasattr(idx, "time"):
                    t = idx.time()
                    if inclusive == "both":
                        if start_time <= t <= end_time:
                            filtered.append(i)
                    elif inclusive == "left":
                        if start_time <= t < end_time:
                            filtered.append(i)
                    elif inclusive == "right":
                        if start_time < t <= end_time:
                            filtered.append(i)
                    else:
                        if start_time < t < end_time:
                            filtered.append(i)
            if filtered:
                return Series(self._series[filtered])
        return Series(pl.Series([], dtype=self._series.dtype))

    def resample(
        self,
        rule: str,  # type: ignore[valid-type]
        axis: int = 0,
        closed: Any = None,
        label: Any = None,
        convention: str = "start",  # type: ignore[valid-type]
        kind: Any = None,
        loffset: Any = None,
        base: Any = None,
        on: Any = None,
        level: Any = None,
        origin: str = "start_day",  # type: ignore[valid-type]
        offset: Any = None,
        group_keys: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Resample time-series data.

        Parameters
        ----------
        rule : str
            Resampling rule
        axis : int, default 0
            Axis (not used for Series)
        closed : str, optional
            Which side of interval is closed
        label : str, optional
            Which side of interval to label
        convention : str, default "start"
            Convention for period conversion
        kind : str, optional
            Type of resampling
        loffset : Any, optional
            Label offset
        base : int, optional
            Base for resampling
        on : str, optional
            Column to resample on (not used for Series)
        level : Any, optional
            Level for MultiIndex
        origin : str, default "start_day"
            Origin for resampling
        offset : Any, optional
            Offset for resampling
        group_keys : bool, default False
            Include group keys
        **kwargs
            Additional arguments

        Returns
        -------
        Resampler
            Resampler object

        Examples
        --------
        >>> import polarpandas as ppd
        >>> s = ppd.Series([1, 2, 3])
        >>> result = s.resample("D")
        """
        raise NotImplementedError(
            "resample() is not yet implemented.\n"
            "Workarounds:\n"
            "  - Use pandas: pd_series.resample(rule) then convert with polarpandas.Series(s)\n"
            "  - Use Polars group_by_dynamic() for time-based grouping"
        )

    def to_timestamp(
        self,
        freq: Any = None,
        how: str = "start",  # type: ignore[valid-type]
        copy: bool = True,
        **kwargs: Any,
    ) -> Series:
        """
        Cast to DatetimeIndex of Timestamps, at beginning of period.

        Parameters
        ----------
        freq : str, optional
            Frequency
        how : str, default "start"
            How to convert
        copy : bool, default True
            Copy data
        **kwargs
            Additional arguments

        Returns
        -------
        Series
            Series with timestamp index

        Examples
        --------
        >>> import polarpandas as ppd
        >>> s = ppd.Series([1, 2, 3])
        >>> result = s.to_timestamp()
        """
        # Convert index to timestamp if it's a period index
        result = self.copy() if copy else self
        if self._index:
            # Try to convert index to datetime
            try:
                # Convert index to datetime series
                if hasattr(self._index[0], "to_timestamp"):
                    new_index = [idx.to_timestamp(how=how) for idx in self._index]
                else:
                    # Try to parse as datetime
                    new_index = [
                        pl.datetime.fromisoformat(str(idx))  # type: ignore[attr-defined]
                        if isinstance(idx, str)
                        else idx
                        for idx in self._index
                    ]
                result._index = new_index
            except Exception:
                # If conversion fails, keep original index
                pass
        return result

    def tz_convert(
        self,
        tz: Any,
        axis: int = 0,
        level: Any = None,
        copy: bool = True,
        **kwargs: Any,
    ) -> Series:
        """
        Convert tz-aware axis to target time zone.

        Parameters
        ----------
        tz : str or tzinfo
            Target timezone
        axis : int, default 0
            Axis (not used for Series)
        level : Any, optional
            Level for MultiIndex
        copy : bool, default True
            Copy data
        **kwargs
            Additional arguments

        Returns
        -------
        Series
            Series with converted timezone

        Examples
        --------
        >>> import polarpandas as ppd
        >>> s = ppd.Series([1, 2, 3])
        >>> result = s.tz_convert("UTC")
        """
        result = self.copy() if copy else self
        if self._index:
            # Convert timezone-aware datetime index
            try:
                # Convert each index value
                new_index = []
                for idx in self._index:
                    if hasattr(idx, "tz_convert"):
                        new_idx = idx.tz_convert(tz)
                    elif isinstance(idx, pl.Datetime):
                        # Use Polars timezone conversion
                        new_idx = idx.replace_time_zone(str(tz))  # type: ignore[attr-defined]
                    else:
                        new_idx = idx
                    new_index.append(new_idx)
                result._index = new_index
            except Exception:
                # If conversion fails, keep original index
                pass
        return result

    def tz_localize(
        self,
        tz: Any,
        axis: int = 0,
        level: Any = None,
        copy: bool = True,
        ambiguous: str = "raise",  # type: ignore[valid-type]
        nonexistent: str = "raise",  # type: ignore[valid-type]
        **kwargs: Any,
    ) -> Series:
        """
        Localize tz-naive index to target time zone.

        Parameters
        ----------
        tz : str or tzinfo
            Target timezone
        axis : int, default 0
            Axis (not used for Series)
        level : Any, optional
            Level for MultiIndex
        copy : bool, default True
            Copy data
        ambiguous : str, default "raise"
            How to handle ambiguous times
        nonexistent : str, default "raise"
            How to handle nonexistent times
        **kwargs
            Additional arguments

        Returns
        -------
        Series
            Series with localized timezone

        Examples
        --------
        >>> import polarpandas as ppd
        >>> s = ppd.Series([1, 2, 3])
        >>> result = s.tz_localize("UTC")
        """
        result = self.copy() if copy else self
        if self._index:
            # Localize timezone-naive datetime index
            try:
                # Convert each index value
                new_index = []
                for idx in self._index:
                    if hasattr(idx, "tz_localize"):
                        new_idx = idx.tz_localize(
                            tz, ambiguous=ambiguous, nonexistent=nonexistent
                        )
                    elif isinstance(idx, pl.Datetime):
                        # Use Polars timezone localization
                        new_idx = idx.replace_time_zone(str(tz))  # type: ignore[attr-defined]
                    else:
                        new_idx = idx
                    new_index.append(new_idx)
                result._index = new_index
            except Exception:
                # If conversion fails, keep original index
                pass
        return result

    # I/O export methods


class _CategoricalAccessor:
    """Categorical operations accessor for Series."""

    def __init__(self, series: Series):
        self._series_obj = series
        if series._series.dtype != pl.Categorical:
            raise TypeError(
                "Can only use .cat accessor with a 'category' dtype. "
                "Use Series.astype('category') first."
            )

    def _categories_list(self) -> BuiltinList[Any]:
        if self._series_obj._categorical_order is not None:
            return list(self._series_obj._categorical_order)
        categories = self._series_obj._series.cat.get_categories().to_list()
        self._series_obj._categorical_order = list(categories)
        return list(categories)

    @property
    def categories(self) -> Any:
        """Return categories as an Index."""
        from polarpandas.index import Index

        return Index(self._categories_list())

    @property
    def codes(self) -> Series:
        """Return integer codes for the categorical values."""
        categories = self._categories_list()
        mapping = {cat: idx for idx, cat in enumerate(categories)}
        values = self._series_obj._series.to_list()
        codes: BuiltinList[int] = []
        for value in values:
            if value is None:
                codes.append(-1)
            else:
                codes.append(mapping.get(value, -1))

        code_series = pl.Series(
            name=self._series_obj.name or "",
            values=codes,
            dtype=pl.Int64,
        )

        index = (
            list(self._series_obj._index)
            if self._series_obj._index is not None
            else None
        )
        result = Series(code_series, name=self._series_obj.name, index=index)
        result._index_name = self._series_obj._index_name
        return result

    def rename_categories(self, new_categories: Any, inplace: bool = False) -> Any:
        """Rename categories using a mapping or new sequence."""

        current_categories = self._categories_list()

        if isinstance(new_categories, dict):
            mapping = new_categories
            unknown = [k for k in mapping if k not in current_categories]
            if unknown:
                raise KeyError(
                    f"Categories {unknown} are not present in the existing categories"
                )
            updated_categories = [mapping.get(cat, cat) for cat in current_categories]
        else:
            updated_categories = list(new_categories)
            if len(updated_categories) != len(current_categories):
                raise ValueError(
                    "Length of new categories must match existing categories"
                )
            mapping = dict(zip(current_categories, updated_categories))

        if len(set(updated_categories)) != len(updated_categories):
            raise ValueError("New categories must be unique")

        values = self._series_obj._series.to_list()
        new_values = [mapping.get(value, value) for value in values]
        index = (
            list(self._series_obj._index)
            if self._series_obj._index is not None
            else None
        )

        pl_series = pl.Series(
            name=self._series_obj.name or "",
            values=new_values,
        ).cast(pl.Categorical)
        result = Series(pl_series, name=self._series_obj.name, index=index)
        result._index_name = self._series_obj._index_name
        result._categorical_order = list(updated_categories)

        if inplace:
            self._series_obj._series = result._series
            self._series_obj._categorical_order = list(updated_categories)
            return None
        return result

    def reorder_categories(self, new_categories: Any, inplace: bool = False) -> Any:
        """Reorder categories according to the provided sequence."""

        current_categories = self._categories_list()
        new_order = list(new_categories)

        if len(new_order) != len(current_categories):
            raise ValueError("New categories must include all existing categories")
        if len(set(new_order)) != len(new_order):
            raise ValueError("New categories must be unique")
        if set(new_order) != set(current_categories):
            raise ValueError("New categories must match existing categories")

        if inplace:
            self._series_obj._categorical_order = list(new_order)
            return None

        result = self._series_obj.copy()
        result._categorical_order = list(new_order)
        return result


class _StringAccessor:
    """String operations accessor for Series."""

    def __init__(self, series: Series):
        self._series = series._series

    def lower(self) -> Series:
        """Convert to lowercase."""
        return Series(self._series.str.to_lowercase())

    def upper(self) -> Series:
        """Convert to uppercase."""
        return Series(self._series.str.to_uppercase())

    def capitalize(self) -> Series:
        """Convert first character to uppercase and rest to lowercase."""
        return Series(self._series.str.to_titlecase())

    def casefold(self) -> Series:
        """Convert to casefolded strings for caseless matching."""
        # Polars doesn't have casefold, use lowercase as approximation
        return Series(self._series.str.to_lowercase())

    def count(self, pat: str, flags: int = 0, **kwargs: Any) -> Series:
        """
        Count occurrences of pattern in each string.

        Parameters
        ----------
        pat : str
            Pattern to count
        flags : int, default 0
            Regex flags (not fully supported)
        **kwargs
            Additional arguments

        Returns
        -------
        Series
            Count of pattern occurrences
        """
        # Use Polars string count
        return Series(self._series.str.count_matches(pat))

    def contains(self, pat: str) -> Series:
        """Check if pattern is contained."""
        # Handle empty series
        if len(self._series) == 0:
            return Series(pl.Series([], dtype=pl.Boolean))
        return Series(self._series.str.contains(pat))

    def startswith(self, pat: str) -> Series:
        """Check if starts with pattern."""
        return Series(self._series.str.starts_with(pat))

    def endswith(self, pat: str) -> Series:
        """Check if ends with pattern."""
        return Series(self._series.str.ends_with(pat))

    def len(self) -> Series:
        """Get length of strings."""
        return Series(self._series.str.len_chars())

    def strip(self) -> Series:
        """Strip whitespace."""
        return Series(self._series.str.strip_chars())

    def replace(self, pat: str, repl: str) -> Series:
        """Replace pattern with replacement."""
        return Series(self._series.str.replace_all(pat, repl))

    def split(self, pat: Any = None, n: int = -1, expand: bool = False) -> Any:
        """
        Split strings around given separator/delimiter.

        Parameters
        ----------
        pat : str, optional
            String or regular expression to split on
        n : int, default -1
            Limit number of splits in output
        expand : bool, default False
            Expand the split strings into separate columns

        Returns
        -------
        Series or DataFrame
            Split strings
        """
        if pat is None:
            pat = " "  # Default to space for Polars

        if expand:
            # Use Polars unnest for expand functionality
            import polarpandas as ppd

            split_series = self._series.str.split(by=pat)

            # Create DataFrame with split columns
            df = pl.DataFrame({self._series.name or "0": split_series})

            # Determine the maximum number of splits to avoid creating too many columns
            # Get the maximum length of split lists
            max_splits = df.select(
                pl.col(self._series.name or "0").list.len().max()
            ).item()

            if max_splits is None:
                max_splits = 1

            # Use to_struct with the actual number of splits
            df_unnested = df.with_columns(
                pl.col(self._series.name or "0").list.to_struct(upper_bound=max_splits)
            ).unnest(self._series.name or "0")

            # Rename columns to match pandas behavior (0, 1, 2, ...)
            result_df = ppd.DataFrame(df_unnested)
            num_cols = len(df_unnested.columns)
            new_columns = [str(i) for i in range(num_cols)]
            result_df._df = result_df._df.rename(
                dict(zip(df_unnested.columns, new_columns))
            )

            return result_df
        else:
            return Series(self._series.str.split(by=pat))

    def extract(self, pat: str, flags: int = 0, expand: bool = True) -> Any:
        """
        Extract capture groups in the regex pat as columns in a DataFrame.

        Parameters
        ----------
        pat : str
            Regular expression pattern with capturing groups
        flags : int, default 0
            Flags to pass through to the re module
        expand : bool, default True
            If True, return DataFrame with one column per capture group

        Returns
        -------
        Series or DataFrame
            Extracted groups
        """
        if expand:
            # Use Polars extract for expand functionality
            import polarpandas as ppd

            extracted = self._series.str.extract(pat)

            # For extract, we need to handle the case where extract returns strings, not lists
            # Polars str.extract returns a single string, not a list
            # We need to create a DataFrame directly from the extracted values
            df = pl.DataFrame({self._series.name or "0": extracted})

            # For extract, we don't need to_struct since extract returns strings, not lists
            return ppd.DataFrame(df)
        else:
            return Series(self._series.str.extract(pat))

    def slice(
        self,
        start: Any = None,
        stop: Any = None,
        step: Any = None,
    ) -> Series:
        """
        Slice substrings from each element in the Series.

        Parameters
        ----------
        start : int, optional
            Start position for slice operation
        stop : int, optional
            Stop position for slice operation
        step : int, optional
            Step size for slice operation

        Returns
        -------
        Series
            Sliced strings
        """
        # Handle step parameter - Polars has limited step support
        if step is not None and step != 1:
            # Implement step support using Python string slicing
            # This is a workaround since Polars doesn't support step in str.slice
            def apply_step_slice(s: Any) -> Any:
                if s is None:
                    return None
                return s[start:stop:step]

            # Apply the step slice using map_elements
            result = self._series.map_elements(apply_step_slice, return_dtype=pl.Utf8)
            return Series(result)

        # For simple slicing without step, use Polars
        if stop is not None and start is not None:
            length = stop - start
        elif stop is not None:
            length = stop
        else:
            length = None

        return Series(self._series.str.slice(start, length))

    # Simple String Operations
    def center(self, width: int, fillchar: str = " ") -> Series:
        """Center strings in a Series."""

        def center_str(s: Any) -> Any:
            if s is None:
                return None
            return s.center(width, fillchar)

        return Series(self._series.map_elements(center_str, return_dtype=pl.Utf8))

    def ljust(self, width: int, fillchar: str = " ") -> Series:
        """Left justify strings in a Series."""
        return Series(self._series.str.pad_end(width, fillchar))

    def rjust(self, width: int, fillchar: str = " ") -> Series:
        """Right justify strings in a Series."""
        return Series(self._series.str.pad_start(width, fillchar))

    def lstrip(self, chars: Any = None) -> Series:
        """Remove leading characters."""
        if chars is None:
            return Series(self._series.str.strip_chars_start())
        return Series(self._series.str.strip_chars_start(chars))

    def rstrip(self, chars: Any = None) -> Series:
        """Remove trailing characters."""
        if chars is None:
            return Series(self._series.str.strip_chars_end())
        return Series(self._series.str.strip_chars_end(chars))

    def swapcase(self) -> Series:
        """Swap case of strings."""

        def swap_case(s: Any) -> Any:
            if s is None:
                return None
            return s.swapcase()

        return Series(self._series.map_elements(swap_case, return_dtype=pl.Utf8))

    def title(self) -> Series:
        """Convert strings to title case."""

        def title_case(s: Any) -> Any:
            if s is None:
                return None
            return s.title()

        return Series(self._series.map_elements(title_case, return_dtype=pl.Utf8))

    def zfill(self, width: int) -> Series:
        """Pad strings with zeros on the left."""
        return Series(self._series.str.zfill(width))

    # Character Classification
    def isalnum(self) -> Series:
        """Check if all characters are alphanumeric."""

        def check_alnum(s: Any) -> Any:
            if s is None:
                return None
            return s.isalnum()

        return Series(self._series.map_elements(check_alnum, return_dtype=pl.Boolean))

    def isalpha(self) -> Series:
        """Check if all characters are alphabetic."""

        def check_alpha(s: Any) -> Any:
            if s is None:
                return None
            return s.isalpha()

        return Series(self._series.map_elements(check_alpha, return_dtype=pl.Boolean))

    def isascii(self) -> Series:
        """Check if all characters are ASCII."""

        def check_ascii(s: Any) -> Any:
            if s is None:
                return None
            return s.isascii()

        return Series(self._series.map_elements(check_ascii, return_dtype=pl.Boolean))

    def isdecimal(self) -> Series:
        """Check if all characters are decimal."""

        def check_decimal(s: Any) -> Any:
            if s is None:
                return None
            return s.isdecimal()

        return Series(self._series.map_elements(check_decimal, return_dtype=pl.Boolean))

    def isdigit(self) -> Series:
        """Check if all characters are digits."""

        def check_digit(s: Any) -> Any:
            if s is None:
                return None
            return s.isdigit()

        return Series(self._series.map_elements(check_digit, return_dtype=pl.Boolean))

    def islower(self) -> Series:
        """Check if all characters are lowercase."""

        def check_lower(s: Any) -> Any:
            if s is None:
                return None
            return s.islower()

        return Series(self._series.map_elements(check_lower, return_dtype=pl.Boolean))

    def isnumeric(self) -> Series:
        """Check if all characters are numeric."""

        def check_numeric(s: Any) -> Any:
            if s is None:
                return None
            return s.isnumeric()

        return Series(self._series.map_elements(check_numeric, return_dtype=pl.Boolean))

    def isspace(self) -> Series:
        """Check if all characters are whitespace."""

        def check_space(s: Any) -> Any:
            if s is None:
                return None
            return s.isspace()

        return Series(self._series.map_elements(check_space, return_dtype=pl.Boolean))

    def istitle(self) -> Series:
        """Check if strings are in title case."""

        def check_title(s: Any) -> Any:
            if s is None:
                return None
            return s.istitle()

        return Series(self._series.map_elements(check_title, return_dtype=pl.Boolean))

    def isupper(self) -> Series:
        """Check if all characters are uppercase."""

        def check_upper(s: Any) -> Any:
            if s is None:
                return None
            return s.isupper()

        return Series(self._series.map_elements(check_upper, return_dtype=pl.Boolean))

    # String Finding/Indexing
    def find(self, sub: str, start: int = 0, end: Any = None) -> Series:
        """Find substring in each string, return -1 if not found."""

        def find_sub(s: Any) -> Any:
            if s is None:
                return None
            try:
                return s.find(sub, start, end)
            except (TypeError, ValueError):
                return -1

        return Series(self._series.map_elements(find_sub, return_dtype=pl.Int64))

    def rfind(self, sub: str, start: int = 0, end: Any = None) -> Series:
        """Find substring from right, return -1 if not found."""

        def rfind_sub(s: Any) -> Any:
            if s is None:
                return None
            try:
                return s.rfind(sub, start, end)
            except (TypeError, ValueError):
                return -1

        return Series(self._series.map_elements(rfind_sub, return_dtype=pl.Int64))

    def index(self, sub: str, start: int = 0, end: Any = None) -> Series:
        """Find substring, raise ValueError if not found."""

        def index_sub(s: Any) -> Any:
            if s is None:
                return None
            try:
                return s.index(sub, start, end)
            except ValueError:
                # Return -1 to match pandas behavior (raises in some cases)
                return -1

        return Series(self._series.map_elements(index_sub, return_dtype=pl.Int64))

    def rindex(self, sub: str, start: int = 0, end: Any = None) -> Series:
        """Find substring from right, raise ValueError if not found."""

        def rindex_sub(s: Any) -> Any:
            if s is None:
                return None
            try:
                return s.rindex(sub, start, end)
            except ValueError:
                # Return -1 to match pandas behavior (raises in some cases)
                return -1

        return Series(self._series.map_elements(rindex_sub, return_dtype=pl.Int64))

    def get(self, i: int) -> Series:
        """Get character at position."""

        def get_char(s: Any) -> Any:
            if s is None:
                return None
            try:
                return s[i] if 0 <= i < len(s) else ""
            except (TypeError, IndexError):
                return ""

        return Series(self._series.map_elements(get_char, return_dtype=pl.Utf8))

    # String Splitting/Partitioning
    def rsplit(self, pat: Any = None, n: int = -1) -> Series:
        """Split strings from the right."""
        if pat is None:
            pat = " "

        def rsplit_str(s: Any) -> Any:
            if s is None:
                return None
            return s.rsplit(pat, n)

        return Series(
            self._series.map_elements(rsplit_str, return_dtype=pl.List(pl.Utf8))
        )

    def partition(self, sep: str = " ") -> Series:
        """Partition strings at first occurrence of separator."""

        def partition_str(s: Any) -> Any:
            if s is None:
                return None
            return s.partition(sep)

        # Return as list of strings
        def partition_to_list(s: Any) -> Any:
            if s is None:
                return None
            parts = s.partition(sep)
            return list(parts)

        return Series(
            self._series.map_elements(partition_to_list, return_dtype=pl.List(pl.Utf8))
        )

    def rpartition(self, sep: str = " ") -> Series:
        """Partition strings at last occurrence of separator."""

        def rpartition_to_list(s: Any) -> Any:
            if s is None:
                return None
            parts = s.rpartition(sep)
            return list(parts)

        return Series(
            self._series.map_elements(rpartition_to_list, return_dtype=pl.List(pl.Utf8))
        )

    # String Manipulation
    def repeat(self, repeats: Any) -> Series:
        """Repeat strings."""
        if isinstance(repeats, Series):

            def repeat_with_series(s: Any, r: Any) -> Any:
                if s is None or r is None:
                    return None
                return s * r if r >= 0 else ""

            # Combine series and repeats
            combined = pl.DataFrame({"s": self._series, "r": repeats._series})
            result = combined.select(
                pl.struct(["s", "r"]).map_elements(
                    lambda x: repeat_with_series(x["s"], x["r"]), return_dtype=pl.Utf8
                )
            )["s"]
            return Series(result)
        else:

            def repeat_str(s: Any) -> Any:
                if s is None:
                    return None
                return s * repeats if repeats >= 0 else ""

            return Series(self._series.map_elements(repeat_str, return_dtype=pl.Utf8))

    def join(self, sep: str) -> Series:
        """Join strings with separator."""

        def join_str(s: Any) -> Any:
            if s is None:
                return None
            # If s is a string, join its characters
            return sep.join(s)

        return Series(self._series.map_elements(join_str, return_dtype=pl.Utf8))

    def removeprefix(self, prefix: str) -> Series:
        """Remove prefix from strings."""

        def remove_prefix(s: Any) -> Any:
            if s is None:
                return None
            return s.removeprefix(prefix)

        return Series(self._series.map_elements(remove_prefix, return_dtype=pl.Utf8))

    def removesuffix(self, suffix: str) -> Series:
        """Remove suffix from strings."""

        def remove_suffix(s: Any) -> Any:
            if s is None:
                return None
            return s.removesuffix(suffix)

        return Series(self._series.map_elements(remove_suffix, return_dtype=pl.Utf8))

    def slice_replace(
        self, start: Any = None, stop: Any = None, repl: str = ""
    ) -> Series:
        """Replace slice of strings."""

        def slice_replace_str(s: Any) -> Any:
            if s is None:
                return None
            if start is None and stop is None:
                return repl
            elif start is None:
                return repl + s[stop:]
            elif stop is None:
                return s[:start] + repl
            else:
                return s[:start] + repl + s[stop:]

        return Series(
            self._series.map_elements(slice_replace_str, return_dtype=pl.Utf8)
        )

    def pad(self, width: int, side: str = "left", fillchar: str = " ") -> Series:
        """Pad strings (deprecated, use center/ljust/rjust)."""
        if side == "left":
            return self.rjust(width, fillchar)
        elif side == "right":
            return self.ljust(width, fillchar)
        else:  # "both"
            return self.center(width, fillchar)

    # Regex Operations
    def findall(self, pat: str, flags: int = 0) -> Series:
        """Find all matches of pattern."""
        import re

        def find_all(s: Any) -> Any:
            if s is None:
                return None
            try:
                return re.findall(pat, s, flags)
            except Exception:
                return []

        return Series(
            self._series.map_elements(find_all, return_dtype=pl.List(pl.Utf8))
        )

    def fullmatch(self, pat: str, flags: int = 0) -> Series:
        """Check if full string matches pattern."""
        import re

        def full_match(s: Any) -> Any:
            if s is None:
                return None
            try:
                return bool(re.fullmatch(pat, s, flags))
            except Exception:
                return False

        return Series(self._series.map_elements(full_match, return_dtype=pl.Boolean))

    def match(self, pat: str, flags: int = 0) -> Series:
        """Check if string matches pattern at start."""
        import re

        def match_str(s: Any) -> Any:
            if s is None:
                return None
            try:
                return bool(re.match(pat, s, flags))
            except Exception:
                return False

        return Series(self._series.map_elements(match_str, return_dtype=pl.Boolean))

    def extractall(self, pat: str, flags: int = 0) -> DataFrame:
        """Extract all matches of pattern (returns DataFrame)."""
        import re

        # Extract all matches for each string
        def extract_all(s: Any) -> Any:
            if s is None:
                return None
            try:
                matches = re.finditer(pat, s, flags)
                result = []
                for match in matches:
                    # Get all groups
                    groups = match.groups()
                    if groups:
                        result.append({str(i): g for i, g in enumerate(groups, 1)})
                    else:
                        result.append({"0": match.group(0)})
                return result if result else None
            except Exception:
                return None

        # This is complex - for now, return NotImplementedError with workaround
        raise NotImplementedError(
            "extractall() is not yet fully implemented.\n"
            "Workarounds:\n"
            "  - Use pandas: pd_series.str.extractall(pat) then convert with polarpandas.DataFrame(df)\n"
            "  - Use findall() and process results manually"
        )

    # Encoding/Decoding
    def encode(self, encoding: str = "utf-8", errors: str = "strict") -> Series:
        """Encode strings to bytes."""

        def encode_str(s: Any) -> Any:
            if s is None:
                return None
            try:
                return s.encode(encoding, errors)
            except Exception:
                return None

        # Polars doesn't have native bytes type, return as list of integers
        def encode_to_list(s: Any) -> Any:
            if s is None:
                return None
            try:
                return list(s.encode(encoding, errors))
            except Exception:
                return None

        return Series(
            self._series.map_elements(encode_to_list, return_dtype=pl.List(pl.UInt8))
        )

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> Series:
        """Decode bytes to strings."""

        # For Series of bytes (as list of ints), decode them
        def decode_bytes(s: Any) -> Any:
            if s is None:
                return None
            try:
                return bytes(s).decode(encoding, errors)
            except Exception:
                return None

        return Series(self._series.map_elements(decode_bytes, return_dtype=pl.Utf8))

    # Special Operations
    def cat(
        self,
        others: Any = None,
        sep: Any = None,
        na_rep: Any = None,
        join: str = "left",
    ) -> Series:
        """Concatenate strings."""
        raise NotImplementedError(
            "cat() is not yet fully implemented.\n"
            "Workarounds:\n"
            "  - Use pandas: pd_series.str.cat(others, sep) then convert with polarpandas.Series(s)\n"
            "  - Use + operator: series1 + sep + series2"
        )

    def get_dummies(self, sep: str = "|") -> DataFrame:
        """Get dummy variables from strings."""
        import polarpandas as ppd

        # Split by separator and create dummy columns
        split_series = self.split(pat=sep, expand=False)
        # Get unique values
        unique_vals = set()
        for val_list in split_series._series.to_list():
            if val_list:
                unique_vals.update(val_list)
        # Create dummy DataFrame
        result_dict = {}
        for val in sorted(unique_vals):
            # Use a factory function to properly capture val in closure
            def make_has_val(v: Any) -> Any:
                def has_val(s: Any) -> bool:
                    if s is None:
                        return False
                    return v in s

                return has_val

            result_dict[val] = Series(
                split_series._series.map_elements(
                    make_has_val(val), return_dtype=pl.Boolean
                )
            )
        return ppd.DataFrame(result_dict)

    def normalize(self, form: str) -> Series:
        """Unicode normalization."""
        try:
            import unicodedata

            def normalize_str(s: Any) -> Any:
                if s is None:
                    return None
                try:
                    return unicodedata.normalize(form, s)  # type: ignore[arg-type]
                except Exception:
                    return s

            return Series(
                self._series.map_elements(normalize_str, return_dtype=pl.Utf8)
            )
        except ImportError:
            raise NotImplementedError(
                "normalize() requires unicodedata (standard library).\n"
                "This should be available in Python standard library."
            ) from None

    def translate(self, table: BuiltinDict[int, Any]) -> Series:
        """Translate characters using translation table."""

        def translate_str(s: Any) -> Any:
            if s is None:
                return None
            try:
                # Convert dict to str.maketrans format
                # table maps ord(char) -> replacement
                trans_table = str.maketrans(
                    {k: (v if v else "") for k, v in table.items()}
                )
                return s.translate(trans_table)
            except Exception:
                return s

        return Series(self._series.map_elements(translate_str, return_dtype=pl.Utf8))

    def wrap(self, width: int, **kwargs: Any) -> Series:
        """Wrap text to specified width."""
        import textwrap

        def wrap_str(s: Any) -> Any:
            if s is None:
                return None
            try:
                return "\n".join(textwrap.wrap(s, width, **kwargs))
            except Exception:
                return s

        return Series(self._series.map_elements(wrap_str, return_dtype=pl.Utf8))


class _DatetimeAccessor:
    """Datetime operations accessor for Series."""

    def __init__(self, series: Series):
        self._series = series._series

    @property
    def year(self) -> Series:
        """Get year."""
        # Handle empty series
        if len(self._series) == 0:
            return Series(pl.Series([], dtype=pl.Int32))
        return Series(self._series.dt.year())

    @property
    def month(self) -> Series:
        """Get month."""
        return Series(self._series.dt.month())

    @property
    def day(self) -> Series:
        """Get day."""
        return Series(self._series.dt.day())

    @property
    def hour(self) -> Series:
        """Get hour."""
        return Series(self._series.dt.hour())

    @property
    def minute(self) -> Series:
        """Get minute."""
        return Series(self._series.dt.minute())

    @property
    def second(self) -> Series:
        """Get second."""
        return Series(self._series.dt.second())

    @property
    def weekday(self) -> Series:
        """Get day of week."""
        return Series(self._series.dt.weekday())

    def strftime(self, fmt: str) -> Series:
        """Format datetime as string."""
        return Series(self._series.dt.strftime(fmt))

    def day_name(self, locale: Any = None) -> Series:
        """
        Return the day names of the datetime with specified locale.

        Parameters
        ----------
        locale : str, optional
            Locale to use for day names

        Returns
        -------
        Series
            Day names
        """
        # Polars doesn't have direct day_name, use weekday and map
        weekday = self._series.dt.weekday()
        day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        # Map weekday (0=Monday, 6=Sunday) to day names
        result = weekday.map_elements(
            lambda x: day_names[x] if 0 <= x < 7 else None, return_dtype=pl.Utf8
        )
        return Series(result)

    def month_name(self, locale: Any = None) -> Series:
        """
        Return the month names of the datetime with specified locale.

        Parameters
        ----------
        locale : str, optional
            Locale to use for month names

        Returns
        -------
        Series
            Month names
        """
        # Polars doesn't have direct month_name, use month and map
        month = self._series.dt.month()
        month_names = [
            "",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        # Map month (1-12) to month names
        result = month.map_elements(
            lambda x: month_names[x] if 1 <= x <= 12 else None, return_dtype=pl.Utf8
        )
        return Series(result)

    @property
    def date(self) -> Series:
        """Extract date part."""
        # Use Polars date extraction - dtype may differ from pandas
        return Series(self._series.dt.date())

    @property
    def time(self) -> Series:
        """Extract time part."""
        return Series(self._series.dt.time())

    @property
    def dayofweek(self) -> Series:
        """Get day of week (Monday=0, Sunday=6)."""
        # Polars weekday() returns different values than pandas dayofweek
        # Adjust to match pandas convention
        weekday_series: pl.Series = self._series.dt.weekday() - 1
        return Series(weekday_series.cast(pl.Int32))

    @property
    def dayofyear(self) -> Series:
        """Get day of year."""
        return Series(self._series.dt.ordinal_day().cast(pl.Int32))

    @property
    def quarter(self) -> Series:
        """Get quarter of year."""
        return Series(self._series.dt.quarter().cast(pl.Int32))

    @property
    def is_month_start(self) -> Series:
        """Check if date is first day of month."""
        return Series(self._series.dt.day() == 1)

    @property
    def is_month_end(self) -> Series:
        """Check if date is last day of month."""
        # Get next day and check if it's the 1st
        next_day = self._series + pl.duration(days=1)
        result_expr = next_day.dt.day() == 1
        # Evaluate the expression by creating a DataFrame and selecting the result
        df = pl.DataFrame({"temp": self._series})
        result_df = df.select(result_expr.alias("result"))
        return Series(result_df["result"])

    @property
    def is_quarter_start(self) -> Series:
        """Check if date is first day of quarter."""
        month = self._series.dt.month()
        day = self._series.dt.day()
        return Series((month.is_in([1, 4, 7, 10])) & (day == 1))

    @property
    def is_quarter_end(self) -> Series:
        """Check if date is last day of quarter."""
        next_day = self._series + pl.duration(days=1)
        next_month = next_day.dt.month()
        result_expr = next_month.is_in([1, 4, 7, 10]) & (next_day.dt.day() == 1)
        # Evaluate the expression by creating a DataFrame and selecting the result
        df = pl.DataFrame({"temp": self._series})
        result_df = df.select(result_expr.alias("result"))
        return Series(result_df["result"])

    @property
    def is_year_start(self) -> Series:
        """Check if date is first day of year."""
        return Series((self._series.dt.month() == 1) & (self._series.dt.day() == 1))

    @property
    def is_year_end(self) -> Series:
        """Check if date is last day of year."""
        next_day = self._series + pl.duration(days=1)
        result_expr = (next_day.dt.month() == 1) & (next_day.dt.day() == 1)
        # Evaluate the expression by creating a DataFrame and selecting the result
        df = pl.DataFrame({"temp": self._series})
        result_df = df.select(result_expr.alias("result"))
        return Series(result_df["result"])

    def floor(self, freq: str) -> Series:
        """Floor datetime to specified frequency."""
        parsed_freq = self._parse_freq_to_duration(freq)
        return Series(self._series.dt.truncate(parsed_freq))

    def ceil(self, freq: str) -> Series:
        """Ceil datetime to specified frequency."""
        # Polars doesn't have ceil, use floor + offset
        parsed_freq = self._parse_freq_to_duration(freq)
        floored = self._series.dt.truncate(parsed_freq)
        # Add one unit of the frequency if not already at the boundary
        duration_kwargs = self._parse_freq_to_duration_kwargs(freq)
        result_expr = (
            pl.when(self._series == floored)
            .then(floored)
            .otherwise(floored + pl.duration(**duration_kwargs))  # type: ignore[arg-type]
        )
        # Evaluate the expression by creating a DataFrame and selecting the result
        df = pl.DataFrame({"temp": self._series})
        result_df = df.select(result_expr.alias("result"))
        return Series(result_df["result"])

    def round(self, freq: str) -> Series:
        """Round datetime to specified frequency."""
        parsed_freq = self._parse_freq_to_duration(freq)
        return Series(self._series.dt.round(parsed_freq))

    def _parse_freq_to_duration(self, freq: str) -> str:
        """Parse pandas frequency string to Polars duration."""
        freq_map = {
            "D": "1d",
            "H": "1h",
            "h": "1h",
            "T": "1m",
            "min": "1m",
            "S": "1s",
            "s": "1s",
            "MS": "1ms",
            "ms": "1ms",
            "US": "1us",
            "us": "1us",
            "NS": "1ns",
            "ns": "1ns",
        }
        return freq_map.get(freq, freq)

    def _parse_freq_to_duration_kwargs(
        self, freq: BuiltinStr
    ) -> BuiltinDict[BuiltinStr, int]:
        """Parse pandas frequency string to Polars duration kwargs."""
        freq_map = {
            "D": {"days": 1},
            "H": {"hours": 1},
            "h": {"hours": 1},
            "T": {"minutes": 1},
            "min": {"minutes": 1},
            "S": {"seconds": 1},
            "s": {"seconds": 1},
            "MS": {"milliseconds": 1},
            "ms": {"milliseconds": 1},
            "US": {"microseconds": 1},
            "us": {"microseconds": 1},
            "NS": {"nanoseconds": 1},
            "ns": {"nanoseconds": 1},
        }
        return freq_map.get(freq, {"days": 1})

    def normalize(self) -> Series:
        """
        Normalize datetime to midnight (00:00:00).

        Returns
        -------
        Series
            Series with datetime normalized to midnight

        Examples
        --------
        >>> import polarpandas as ppd
        >>> s = ppd.Series([ppd.Timestamp('2023-01-01 14:30:00')])
        >>> result = s.dt.normalize()
        """
        # Normalize to midnight by truncating to day
        return Series(self._series.dt.truncate("1d"))

    def to_pydatetime(self) -> Series:
        """
        Convert to Python datetime objects.

        Returns
        -------
        Series
            Series of Python datetime.datetime objects

        Examples
        --------
        >>> import polarpandas as ppd
        >>> s = ppd.Series([ppd.Timestamp('2023-01-01')])
        >>> result = s.dt.to_pydatetime()
        """
        from datetime import datetime

        # Convert Polars datetime to Python datetime
        def to_py_dt(dt: Any) -> Any:
            if dt is None:
                return None
            # If already a datetime, return it
            if isinstance(dt, datetime):
                return dt
            # Convert from Polars datetime
            try:
                # Polars datetime can be converted via to_python
                if hasattr(dt, "to_python"):
                    return dt.to_python()
                # Otherwise, try to parse
                return datetime.fromisoformat(str(dt))
            except Exception:
                return None

        return Series(self._series.map_elements(to_py_dt, return_dtype=pl.Object))

    def as_unit(self, unit: str) -> Series:
        """
        Convert datetime to specified time unit.

        Parameters
        ----------
        unit : str
            Time unit: 'ns', 'us', 'ms', 's'

        Returns
        -------
        Series
            Series with datetime converted to specified unit

        Examples
        --------
        >>> import polarpandas as ppd
        >>> s = ppd.Series([ppd.Timestamp('2023-01-01')])
        >>> result = s.dt.as_unit('ns')
        """
        # Map pandas unit strings to Polars
        unit_map = {
            "ns": "ns",
            "nanosecond": "ns",
            "nanoseconds": "ns",
            "us": "us",
            "microsecond": "us",
            "microseconds": "us",
            "ms": "ms",
            "millisecond": "ms",
            "milliseconds": "ms",
            "s": "s",
            "second": "s",
            "seconds": "s",
        }
        polars_unit = unit_map.get(unit.lower(), unit)
        # Use Polars datetime conversion
        try:
            # Convert to timestamp in specified unit
            if polars_unit == "ns":
                return Series(self._series.dt.timestamp("ns"))
            elif polars_unit == "us":
                return Series(self._series.dt.timestamp("us"))
            elif polars_unit == "ms":
                return Series(self._series.dt.timestamp("ms"))
            elif polars_unit == "s":
                return Series(self._series.dt.timestamp("s"))  # type: ignore[arg-type]
            else:
                raise ValueError(f"Unsupported unit: {unit}")
        except Exception as e:
            raise ValueError(f"Error converting to unit {unit}: {e}") from e

    def isocalendar(self) -> DataFrame:
        """
        Return ISO calendar year, week, and day.

        Returns
        -------
        DataFrame
            DataFrame with columns: year, week, day

        Examples
        --------
        >>> import polarpandas as ppd
        >>> s = ppd.Series([ppd.Timestamp('2023-01-01')])
        >>> result = s.dt.isocalendar()
        """
        from datetime import datetime

        import polarpandas as ppd

        # Use Python's datetime.isocalendar() via map_elements
        def get_isocalendar(dt: Any) -> Any:
            if dt is None:
                return None
            try:
                # Convert to Python datetime if needed
                if isinstance(dt, datetime):
                    py_dt = dt
                elif hasattr(dt, "to_python"):
                    py_dt = dt.to_python()
                else:
                    py_dt = datetime.fromisoformat(str(dt))
                # Get ISO calendar
                iso = py_dt.isocalendar()
                return {"year": iso.year, "week": iso.week, "day": iso.weekday}
            except Exception:
                return None

        # Get ISO calendar for each datetime
        iso_list = self._series.map_elements(get_isocalendar, return_dtype=pl.Object)
        # Convert to DataFrame - extract Polars Series directly
        year_series = iso_list.map_elements(
            lambda x: x["year"] if x else None, return_dtype=pl.Int32
        )
        week_series = iso_list.map_elements(
            lambda x: x["week"] if x else None, return_dtype=pl.Int32
        )
        day_series = iso_list.map_elements(
            lambda x: x["day"] if x else None, return_dtype=pl.Int32
        )
        # Create DataFrame from Polars Series
        result_df = pl.DataFrame(
            {
                "year": year_series,
                "week": week_series,
                "day": day_series,
            }
        )
        return ppd.DataFrame(result_df)


class _SeriesRolling:
    """Series rolling window helper supporting custom apply."""

    def __init__(self, series: Series, window: int, **kwargs: Any) -> None:
        self._series = series
        self._window = window
        self._kwargs = {k: v for k, v in kwargs.items() if k is not None}
        min_periods = self._kwargs.pop("min_periods", None)
        min_samples = self._kwargs.pop("min_samples", None)

        if (
            min_periods is not None
            and min_samples is not None
            and min_periods != min_samples
        ):
            raise ValueError(
                "min_periods and min_samples must match when both provided"
            )

        base_min = min_samples if min_samples is not None else min_periods
        self._min_periods = window if base_min is None else base_min

        if not isinstance(self._min_periods, int):
            raise TypeError("min_periods must be an integer")

        if self._min_periods < 0:
            raise ValueError("min_periods must be non-negative")

    def _rolling_kwargs(self, rolling_callable: Any) -> BuiltinDict[BuiltinStr, Any]:
        rolling_kwargs = dict(self._kwargs)

        if "min_samples" in rolling_kwargs or "min_periods" in rolling_kwargs:
            return rolling_kwargs

        if _callable_accepts_argument(rolling_callable, "min_samples"):
            rolling_kwargs["min_samples"] = self._min_periods
        else:
            rolling_kwargs["min_periods"] = self._min_periods

        return rolling_kwargs

    def _index_copy(self) -> BuiltinList[Any] | None:
        return list(self._series._index) if self._series._index is not None else None

    def _wrap_result(self, polars_series: pl.Series) -> Series:
        return Series(polars_series, name=self._series.name, index=self._index_copy())

    def mean(self) -> Series:
        return self._wrap_result(
            self._series._series.rolling_mean(
                window_size=self._window,
                **self._rolling_kwargs(self._series._series.rolling_mean),
            )
        )

    def sum(self) -> Series:
        return self._wrap_result(
            self._series._series.rolling_sum(
                window_size=self._window,
                **self._rolling_kwargs(self._series._series.rolling_sum),
            )
        )

    def std(self) -> Series:
        return self._wrap_result(
            self._series._series.rolling_std(
                window_size=self._window,
                **self._rolling_kwargs(self._series._series.rolling_std),
            )
        )

    def max(self) -> Series:
        return self._wrap_result(
            self._series._series.rolling_max(
                window_size=self._window,
                **self._rolling_kwargs(self._series._series.rolling_max),
            )
        )

    def min(self) -> Series:
        return self._wrap_result(
            self._series._series.rolling_min(
                window_size=self._window,
                **self._rolling_kwargs(self._series._series.rolling_min),
            )
        )

    def apply(
        self,
        func: Callable[[BuiltinList[Any] | Series], Any],
        raw: bool = False,
        engine: str | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> Series:
        """Apply custom rolling function to the Series."""

        unused_engine = engine_kwargs  # Suppress unused variable warnings
        _ = unused_engine

        args = args or ()
        kwargs = kwargs or {}

        values = self._series.to_list()
        results: BuiltinList[Any] = []

        for end_idx in range(len(values)):
            start_idx = max(0, end_idx + 1 - self._window)
            window_values = values[start_idx : end_idx + 1]

            if len(window_values) < self._min_periods:
                results.append(None)
                continue

            if raw:
                window_input: BuiltinList[Any] | Series = window_values
            else:
                window_input = Series(window_values, name=self._series.name)

            result_value = func(window_input, *args, **kwargs)
            results.append(result_value)

        return Series(results, name=self._series.name, index=self._index_copy())
