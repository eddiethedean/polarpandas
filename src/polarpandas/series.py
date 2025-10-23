"""
Series implementation wrapping Polars Series with pandas-like API.
"""

from typing import Any, List, Optional, Union, Tuple
import polars as pl


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
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize a Series from various data sources.

        Parameters
        ----------
        data : list, pl.Series, or None
            Data to initialize the Series with
        name : str, optional
            Name for the Series
        """
        if data is None:
            self._series = pl.Series(name=name or "", values=[])
        elif isinstance(data, pl.Series):
            self._series = data
        else:
            # Handle list or other array-like data
            self._series = pl.Series(name or "", data)

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
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

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
    def __add__(self, other):
        """Add Series or scalar."""
        if isinstance(other, Series):
            return self._series + other._series
        return self._series + other

    def __sub__(self, other):
        """Subtract Series or scalar."""
        if isinstance(other, Series):
            return self._series - other._series
        return self._series - other

    def __mul__(self, other):
        """Multiply Series or scalar."""
        if isinstance(other, Series):
            return self._series * other._series
        return self._series * other

    def __truediv__(self, other):
        """Divide Series or scalar."""
        if isinstance(other, Series):
            return self._series / other._series
        return self._series / other

    def __radd__(self, other):
        """Right add (for scalar + Series)."""
        return other + self._series

    def __rsub__(self, other):
        """Right subtract (for scalar - Series)."""
        return other - self._series

    def __rmul__(self, other):
        """Right multiply (for scalar * Series)."""
        return other * self._series

    def __rtruediv__(self, other):
        """Right divide (for scalar / Series)."""
        return other / self._series

    # Accessor properties
    @property
    def str(self):
        """String accessor for string operations."""
        return _StringAccessor(self)

    @property
    def dt(self):
        """Datetime accessor for datetime operations."""
        return _DatetimeAccessor(self)

    # Methods
    def apply(self, func):
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
        return self._series.map_elements(func, return_dtype=pl.Float64)

    def map(self, arg):
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
            return self._series.replace(arg, default=None)
        else:
            # Use map_elements for functions
            return self._series.map_elements(arg, return_dtype=pl.Float64)

    def isna(self):
        """
        Detect missing values.

        Returns
        -------
        Series
            Boolean series indicating null values
        """
        return Series(self._series.is_null())

    def notna(self):
        """
        Detect non-missing values.

        Returns
        -------
        Series
            Boolean series indicating non-null values
        """
        return Series(self._series.is_not_null())


class _StringAccessor:
    """String operations accessor for Series."""

    def __init__(self, series: Series):
        self._series = series._series

    def lower(self):
        """Convert to lowercase."""
        return self._series.str.to_lowercase()

    def upper(self):
        """Convert to uppercase."""
        return self._series.str.to_uppercase()

    def contains(self, pat):
        """Check if pattern is contained."""
        return self._series.str.contains(pat)

    def startswith(self, pat):
        """Check if starts with pattern."""
        return self._series.str.starts_with(pat)

    def endswith(self, pat):
        """Check if ends with pattern."""
        return self._series.str.ends_with(pat)

    def len(self):
        """Get length of strings."""
        return self._series.str.len_chars()

    def strip(self):
        """Strip whitespace."""
        return self._series.str.strip_chars()

    def replace(self, pat, repl):
        """Replace pattern with replacement."""
        return self._series.str.replace_all(pat, repl)


class _DatetimeAccessor:
    """Datetime operations accessor for Series."""

    def __init__(self, series: Series):
        self._series = series._series

    @property
    def year(self):
        """Get year."""
        return self._series.dt.year()

    @property
    def month(self):
        """Get month."""
        return self._series.dt.month()

    @property
    def day(self):
        """Get day."""
        return self._series.dt.day()

    @property
    def hour(self):
        """Get hour."""
        return self._series.dt.hour()

    @property
    def minute(self):
        """Get minute."""
        return self._series.dt.minute()

    @property
    def second(self):
        """Get second."""
        return self._series.dt.second()

    @property
    def weekday(self):
        """Get day of week."""
        return self._series.dt.weekday()

    def strftime(self, fmt):
        """Format datetime as string."""
        return self._series.dt.strftime(fmt)
