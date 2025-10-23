"""
Index implementation wrapping Polars Series with pandas-like API.
"""

from typing import Any, List, Optional, Union, Tuple
import polars as pl


class Index:
    """
    An Index wrapper around Polars Series with pandas-like API.

    This class wraps a Polars Series to represent DataFrame indices.
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
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __repr__(self) -> str:
        """Return string representation of the Index."""
        return repr(self._series)

    def __str__(self) -> str:
        """Return string representation of the Index."""
        return str(self._series)

    def __len__(self) -> int:
        """Return the length of the Index."""
        return len(self._series)

    @property
    def shape(self) -> Tuple[int]:
        """Return the shape of the Index."""
        return (len(self._series),)

    @property
    def size(self) -> int:
        """Return the size of the Index."""
        return len(self._series)
