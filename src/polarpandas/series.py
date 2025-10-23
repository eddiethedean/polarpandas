"""
Series implementation wrapping Polars Series with pandas-like API.
"""

import builtins
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import polars as pl

if TYPE_CHECKING:
    from .frame import DataFrame


class Series:
    """
    A mutable Series wrapper around Polars Series with pandas-like API.

    This class wraps a Polars Series and provides a pandas-compatible interface
    with in-place mutation support.
    """

    def __init__(
        self,
        data: Optional[Union[List[Any], pl.Series]] = None,
        name: Optional[str] = None,
        index: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a Series from various data sources.

        Parameters
        ----------
        data : list, pl.Series, or None
            Data to initialize the Series with
        name : str, optional
            Name for the Series
        index : array-like, optional
            Index for the Series
        """
        # Store index information
        self._index = index
        self._index_name = None
        self._original_name = None  # Store original name type for restoration

        if data is None:
            self._series = pl.Series(name=name or "", values=[])
        elif isinstance(data, pl.Series):
            self._series = data
        else:
            # Handle list or other array-like data
            # Pass through kwargs to pl.Series constructor
            self._series = pl.Series(name or "", data, **kwargs)

    @property
    def name(self) -> Optional[str]:
        """Get the name of the Series."""
        return self._series.name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        """Set the name of the Series."""
        if value is not None:
            self._series = self._series.rename(value)

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

    def __len__(self) -> int:
        """Return the length of the Series."""
        return len(self._series)

    @property
    def shape(self) -> Tuple[int]:
        """Return the shape of the Series."""
        return (len(self._series),)

    @property
    def size(self) -> int:
        """Return the size of the Series."""
        return len(self._series)

    # Arithmetic operations
    def __add__(self, other: Union["Series", Any]) -> "Series":
        """Add Series or scalar."""
        if isinstance(other, Series):
            return Series(self._series + other._series)
        return Series(self._series + other)

    def __sub__(self, other: Union["Series", Any]) -> "Series":
        """Subtract Series or scalar."""
        if isinstance(other, Series):
            return Series(self._series - other._series)
        return Series(self._series - other)

    def __mul__(self, other: Union["Series", Any]) -> "Series":
        """Multiply Series or scalar."""
        if isinstance(other, Series):
            return Series(self._series * other._series)
        return Series(self._series * other)

    def __truediv__(self, other: Union["Series", Any]) -> "Series":
        """Divide Series or scalar."""
        if isinstance(other, Series):
            return Series(self._series / other._series)
        return Series(self._series / other)

    def __radd__(self, other: Union["Series", Any]) -> "Series":
        """Right add (for scalar + Series)."""
        return Series(other + self._series)  # type: ignore[arg-type]

    def __rsub__(self, other: Union["Series", Any]) -> "Series":
        """Right subtract (for scalar - Series)."""
        return Series(other - self._series)  # type: ignore[arg-type]

    def __rmul__(self, other: Union["Series", Any]) -> "Series":
        """Right multiply (for scalar * Series)."""
        return Series(other * self._series)  # type: ignore[arg-type]

    def __rtruediv__(self, other: Union["Series", Any]) -> "Series":
        """Right divide (for scalar / Series)."""
        return Series(other / self._series)  # type: ignore[arg-type]

    # Comparison operators
    def __gt__(self, other: Union["Series", Any]) -> "Series":
        """Greater than comparison."""
        if isinstance(other, Series):
            result = Series(self._series > other._series)
        else:
            result = Series(self._series > other)
        # Set name to empty string to match pandas behavior
        result._series = result._series.alias("")
        return result

    def __lt__(self, other: Union["Series", Any]) -> "Series":
        """Less than comparison."""
        if isinstance(other, Series):
            result = Series(self._series < other._series)
        else:
            result = Series(self._series < other)
        result._series = result._series.alias("")
        return result

    def __ge__(self, other: Union["Series", Any]) -> "Series":
        """Greater than or equal comparison."""
        if isinstance(other, Series):
            result = Series(self._series >= other._series)
        else:
            result = Series(self._series >= other)
        result._series = result._series.alias("")
        return result

    def __le__(self, other: Union["Series", Any]) -> "Series":
        """Less than or equal comparison."""
        if isinstance(other, Series):
            result = Series(self._series <= other._series)
        else:
            result = Series(self._series <= other)
        result._series = result._series.alias("")
        return result

    def __eq__(self, other: Union["Series", Any]) -> "Series":  # type: ignore[override]
        """Equal comparison."""
        if isinstance(other, Series):
            result = Series(self._series == other._series)
        else:
            result = Series(self._series == other)
        result._series = result._series.alias("")
        return result

    def __ne__(self, other: Union["Series", Any]) -> "Series":  # type: ignore[override]
        """Not equal comparison."""
        if isinstance(other, Series):
            result = Series(self._series != other._series)
        else:
            result = Series(self._series != other)
        result._series = result._series.alias("")
        return result

    # Accessor properties
    @property
    def str(self) -> "_StringAccessor":
        """String accessor for string operations."""
        return _StringAccessor(self)

    @property
    def dt(self) -> "_DatetimeAccessor":
        """Datetime accessor for datetime operations."""
        return _DatetimeAccessor(self)

    # Methods
    def apply(self, func: Callable[..., Any]) -> "Series":
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

    def map(self, arg: Union[Dict[Any, Any], Callable[..., Any], "Series"]) -> "Series":
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
            return Series(self._series.map_elements(arg, return_dtype=pl.Float64))  # type: ignore[arg-type]

    def isna(self) -> "Series":
        """
        Detect missing values.

        Returns
        -------
        Series
            Boolean series indicating null values
        """
        return Series(self._series.is_null())

    def notna(self) -> "Series":
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
        inclusive: Literal["both", "neither", "left", "right"] = "both",
    ) -> "Series":
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

    def clip(
        self, lower: Optional[Any] = None, upper: Optional[Any] = None
    ) -> "Series":
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
        method: builtins.str = "average",
        ascending: bool = True,
        na_option: builtins.str = "keep",
        pct: bool = False,
    ) -> "Series":
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

    def sort_values(
        self, ascending: bool = True, inplace: bool = False
    ) -> Optional["Series"]:
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

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins: Optional[int] = None,
        dropna: bool = True,
    ) -> "Series":
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

        return Series(result)  # type: ignore[arg-type]

    def unique(self) -> "Series":
        """
        Return unique values in the series.

        Returns
        -------
        Series
            Series with unique values
        """
        return Series(self._series.unique())

    def copy(self) -> "Series":
        """
        Make a copy of the Series.

        Returns
        -------
        Series
            Copy of the Series
        """
        return Series(self._series.clone())

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

        # Set index if we have one
        if self._index is not None:
            pandas_series.index = self._index

        # Handle name conversion - empty string becomes None in pandas
        if self._series.name == "":
            pandas_series.name = None

        return pandas_series


class _StringAccessor:
    """String operations accessor for Series."""

    def __init__(self, series: Series):
        self._series = series._series

    def lower(self) -> "Series":
        """Convert to lowercase."""
        return Series(self._series.str.to_lowercase())

    def upper(self) -> "Series":
        """Convert to uppercase."""
        return Series(self._series.str.to_uppercase())

    def contains(self, pat: str) -> "Series":
        """Check if pattern is contained."""
        # Handle empty series
        if len(self._series) == 0:
            return Series(pl.Series([], dtype=pl.Boolean))
        return Series(self._series.str.contains(pat))

    def startswith(self, pat: str) -> "Series":
        """Check if starts with pattern."""
        return Series(self._series.str.starts_with(pat))

    def endswith(self, pat: str) -> "Series":
        """Check if ends with pattern."""
        return Series(self._series.str.ends_with(pat))

    def len(self) -> "Series":
        """Get length of strings."""
        return Series(self._series.str.len_chars())

    def strip(self) -> "Series":
        """Strip whitespace."""
        return Series(self._series.str.strip_chars())

    def replace(self, pat: str, repl: str) -> "Series":
        """Replace pattern with replacement."""
        return Series(self._series.str.replace_all(pat, repl))

    def split(
        self, pat: Optional[str] = None, n: int = -1, expand: bool = False
    ) -> Union["Series", "DataFrame"]:
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

    def extract(
        self, pat: str, flags: int = 0, expand: bool = True
    ) -> Union["Series", "DataFrame"]:
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
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ) -> "Series":
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
            def apply_step_slice(s: Optional[str]) -> Optional[str]:
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

        return Series(self._series.str.slice(start, length))  # type: ignore[arg-type]


class _DatetimeAccessor:
    """Datetime operations accessor for Series."""

    def __init__(self, series: Series):
        self._series = series._series

    @property
    def year(self) -> "Series":
        """Get year."""
        # Handle empty series
        if len(self._series) == 0:
            return Series(pl.Series([], dtype=pl.Int32))
        return Series(self._series.dt.year())

    @property
    def month(self) -> "Series":
        """Get month."""
        return Series(self._series.dt.month())

    @property
    def day(self) -> "Series":
        """Get day."""
        return Series(self._series.dt.day())

    @property
    def hour(self) -> "Series":
        """Get hour."""
        return Series(self._series.dt.hour())

    @property
    def minute(self) -> "Series":
        """Get minute."""
        return Series(self._series.dt.minute())

    @property
    def second(self) -> "Series":
        """Get second."""
        return Series(self._series.dt.second())

    @property
    def weekday(self) -> "Series":
        """Get day of week."""
        return Series(self._series.dt.weekday())

    def strftime(self, fmt: str) -> "Series":
        """Format datetime as string."""
        return Series(self._series.dt.strftime(fmt))

    @property
    def date(self) -> "Series":
        """Extract date part."""
        # Use Polars date extraction - dtype may differ from pandas
        return Series(self._series.dt.date())

    @property
    def time(self) -> "Series":
        """Extract time part."""
        return Series(self._series.dt.time())

    @property
    def dayofweek(self) -> "Series":
        """Get day of week (Monday=0, Sunday=6)."""
        # Polars weekday() returns different values than pandas dayofweek
        # Adjust to match pandas convention
        return Series((self._series.dt.weekday() - 1).cast(pl.Int32))

    @property
    def dayofyear(self) -> "Series":
        """Get day of year."""
        return Series(self._series.dt.ordinal_day().cast(pl.Int32))

    @property
    def quarter(self) -> "Series":
        """Get quarter of year."""
        return Series(self._series.dt.quarter().cast(pl.Int32))

    @property
    def is_month_start(self) -> "Series":
        """Check if date is first day of month."""
        return Series(self._series.dt.day() == 1)

    @property
    def is_month_end(self) -> "Series":
        """Check if date is last day of month."""
        # Get next day and check if it's the 1st
        next_day = self._series + pl.duration(days=1)
        result_expr = next_day.dt.day() == 1
        # Evaluate the expression by creating a DataFrame and selecting the result
        df = pl.DataFrame({"temp": self._series})
        result_df = df.select(result_expr.alias("result"))
        return Series(result_df["result"])

    @property
    def is_quarter_start(self) -> "Series":
        """Check if date is first day of quarter."""
        month = self._series.dt.month()
        day = self._series.dt.day()
        return Series((month.is_in([1, 4, 7, 10])) & (day == 1))

    @property
    def is_quarter_end(self) -> "Series":
        """Check if date is last day of quarter."""
        next_day = self._series + pl.duration(days=1)
        next_month = next_day.dt.month()
        result_expr = next_month.is_in([1, 4, 7, 10]) & (next_day.dt.day() == 1)
        # Evaluate the expression by creating a DataFrame and selecting the result
        df = pl.DataFrame({"temp": self._series})
        result_df = df.select(result_expr.alias("result"))
        return Series(result_df["result"])

    @property
    def is_year_start(self) -> "Series":
        """Check if date is first day of year."""
        return Series((self._series.dt.month() == 1) & (self._series.dt.day() == 1))

    @property
    def is_year_end(self) -> "Series":
        """Check if date is last day of year."""
        next_day = self._series + pl.duration(days=1)
        result_expr = (next_day.dt.month() == 1) & (next_day.dt.day() == 1)
        # Evaluate the expression by creating a DataFrame and selecting the result
        df = pl.DataFrame({"temp": self._series})
        result_df = df.select(result_expr.alias("result"))
        return Series(result_df["result"])

    def floor(self, freq: str) -> "Series":
        """Floor datetime to specified frequency."""
        parsed_freq = self._parse_freq_to_duration(freq)
        return Series(self._series.dt.truncate(parsed_freq))

    def ceil(self, freq: str) -> "Series":
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

    def round(self, freq: str) -> "Series":
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

    def _parse_freq_to_duration_kwargs(self, freq: str) -> Dict[str, int]:
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
