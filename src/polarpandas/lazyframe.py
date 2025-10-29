"""
A LazyFrame object that provides lazy execution with pandas-compatible API.

This class wraps a Polars LazyFrame and provides a pandas-compatible interface
with lazy execution. Operations are deferred until materialization via .collect().
"""

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

from polarpandas.index import Index

if TYPE_CHECKING:
    from .series import Series
    from .frame import DataFrame


class LazyFrame:
    """
    A LazyFrame wrapper around Polars LazyFrame with pandas-like API.

    This class wraps a Polars LazyFrame and provides a pandas-compatible interface
    with lazy execution. Operations are deferred until materialization via .collect().
    """

    _index: Optional[List[Any]]
    _index_name: Optional[Union[str, Tuple[str, ...]]]
    _columns_index: Optional[Any]
    _df: pl.LazyFrame

    def _materialize(self) -> pl.DataFrame:
        """
        Materialize the LazyFrame to DataFrame.

        Returns
        -------
        pl.DataFrame
            Materialized DataFrame
        """
        return self._df.collect()

    def __init__(
        self,
        data: Optional[Union[Dict[str, Any], List[Any], pl.DataFrame, pl.LazyFrame]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a LazyFrame from various data sources.

        Parameters
        ----------
        data : dict, list, pl.DataFrame, pl.LazyFrame, or None
            Data to initialize the LazyFrame with. Can be:
            - Dictionary of column names to values
            - List of dictionaries
            - Existing Polars DataFrame (converted to LazyFrame)
            - Existing Polars LazyFrame
            - None for empty LazyFrame
        index : array-like, optional
            Index to use for resulting frame
        """
        if data is None:
            # Handle columns and index parameters for empty LazyFrame
            columns = kwargs.pop("columns", None)
            index = kwargs.pop("index", None)

            if index is not None and columns is not None:
                # Create empty LazyFrame with specified columns and index
                self._df = pl.DataFrame({col: [] for col in columns}).lazy()
                self._index = index
                self._index_name = None
            elif index is not None:
                # Create empty LazyFrame with specified index
                self._df = pl.DataFrame().lazy()
                self._index = index
                self._index_name = None
            else:
                # Create empty LazyFrame
                self._df = pl.DataFrame().lazy()
                self._index = None
                self._index_name = None
        elif isinstance(data, pl.LazyFrame):
            # Already a LazyFrame
            self._df = data
            self._index = kwargs.pop("index", None)
            self._index_name = kwargs.pop("index_name", None)
        elif isinstance(data, pl.DataFrame):
            # Convert DataFrame to LazyFrame
            self._df = data.lazy()
            self._index = kwargs.pop("index", None)
            self._index_name = kwargs.pop("index_name", None)
        else:
            # Handle polarpandas DataFrame or other types
            if hasattr(data, '_df'):
                # It's a polarpandas DataFrame
                if isinstance(data._df, pl.DataFrame):
                    self._df = data._df.lazy()
                else:
                    self._df = data._df  # Already a LazyFrame
                self._index = kwargs.pop("index", None)
                self._index_name = kwargs.pop("index_name", None)
            else:
                # Create LazyFrame from data
                self._df = pl.DataFrame(data, *args, **kwargs).lazy()
                self._index = kwargs.pop("index", None)
                self._index_name = kwargs.pop("index_name", None)

    def collect(self) -> "DataFrame":
        """
        Materialize the LazyFrame to an eager DataFrame.

        Returns
        -------
        DataFrame
            Eager DataFrame with materialized data
        """
        from polarpandas.frame import DataFrame
        materialized = self._materialize()
        return DataFrame(materialized, index=self._index, index_name=self._index_name)

    def __repr__(self) -> str:
        """String representation of the LazyFrame."""
        materialized = self._materialize()
        return materialized.__repr__()

    def __str__(self) -> str:
        """String representation of the LazyFrame."""
        materialized = self._materialize()
        return materialized.__str__()

    def __len__(self) -> int:
        """Length of the LazyFrame."""
        materialized = self._materialize()
        return len(materialized)

    def __getitem__(self, key: Union[str, List[str]]) -> Union["LazyFrame", "Series"]:
        """
        Get a column or subset of the LazyFrame.

        Parameters
        ----------
        key : str or other
            Column name or selection key

        Returns
        -------
        Column data or LazyFrame subset
        """
        try:
            from polarpandas.series import Series
            if isinstance(key, str):
                # Single column - materialize and return Series
                materialized = self._materialize()
                return Series(materialized[key])
            elif isinstance(key, list):
                # Multiple columns - stay lazy
                result_lazy = self._df.select(key)
                return LazyFrame(result_lazy)
            elif hasattr(key, '__iter__') and not isinstance(key, str) or isinstance(key, Series):
                # Boolean indexing with Series or array-like
                # Materialize to handle boolean indexing
                materialized = self._materialize()
                if isinstance(key, Series):
                    # Convert polarpandas Series to polars Series for indexing
                    polars_key = key._series
                else:
                    polars_key = key
                # Use filter for boolean indexing
                filtered = materialized.filter(polars_key)
                return LazyFrame(filtered.lazy())
            else:
                # Other key types - delegate to Polars
                return self._df.__getitem__(key)  # type: ignore[return-value]
        except Exception as e:
            # Convert Polars exceptions to pandas-compatible ones
            if "not found" in str(e).lower() or "ColumnNotFoundError" in str(type(e)):
                raise KeyError(str(e)) from e
            raise

    @property
    def columns(self) -> List[str]:
        """Column names of the LazyFrame."""
        return self._df.collect_schema().names()

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the LazyFrame."""
        materialized = self._materialize()
        return materialized.shape

    @property
    def height(self) -> int:
        """Number of rows in the LazyFrame."""
        materialized = self._materialize()
        return materialized.height

    @property
    def width(self) -> int:
        """Number of columns in the LazyFrame."""
        return len(self.columns)

    @property
    def dtypes(self) -> Dict[str, Any]:
        """Data types of columns."""
        schema = self._df.collect_schema()
        return {name: dtype for name, dtype in zip(schema.names(), schema.dtypes())}

    def head(self, n: int = 5) -> "LazyFrame":
        """Return first n rows."""
        result_lazy = self._df.head(n)
        return LazyFrame(result_lazy)

    def tail(self, n: int = 5) -> "LazyFrame":
        """Return last n rows."""
        result_lazy = self._df.tail(n)
        return LazyFrame(result_lazy)

    def filter(self, predicate: Any) -> "LazyFrame":
        """Filter rows based on predicate."""
        result_lazy = self._df.filter(predicate)
        return LazyFrame(result_lazy)

    def select(self, *columns: Union[str, pl.Expr]) -> "LazyFrame":
        """Select columns."""
        result_lazy = self._df.select(*columns)
        return LazyFrame(result_lazy)

    def with_columns(self, *columns: Union[pl.Expr, List[pl.Expr]]) -> "LazyFrame":
        """Add or modify columns."""
        result_lazy = self._df.with_columns(*columns)
        return LazyFrame(result_lazy)

    def group_by(self, *by: Union[str, List[str]]) -> Any:
        """Group by columns."""
        return self._df.group_by(*by)

    def sort(self, by: Union[str, List[str]], descending: bool = False) -> "LazyFrame":
        """Sort by columns."""
        result_lazy = self._df.sort(by, descending=descending)
        return LazyFrame(result_lazy)

    def join(
        self,
        other: Union["LazyFrame", "DataFrame"],
        on: Optional[Union[str, List[str]]] = None,
        how: str = "inner",
        **kwargs: Any,
    ) -> "LazyFrame":
        """Join with another LazyFrame or DataFrame."""
        if hasattr(other, '_df'):
            other_df = other._df
            if isinstance(other_df, pl.DataFrame):
                other_df = other_df.lazy()
        else:
            other_df = other.lazy() if hasattr(other, 'lazy') else other
        
        result_lazy = self._df.join(other_df, on=on, how=how, **kwargs)
        return LazyFrame(result_lazy)

    def to_pandas(self) -> Any:
        """Convert to pandas DataFrame."""
        materialized = self._materialize()
        return materialized.to_pandas()

    def to_csv(self, path: str, **kwargs: Any) -> None:
        """Write to CSV file."""
        materialized = self._materialize()
        materialized.write_csv(path, **kwargs)

    def to_parquet(self, path: str, **kwargs: Any) -> None:
        """Write to Parquet file."""
        materialized = self._materialize()
        materialized.write_parquet(path, **kwargs)

    def to_json(self, path: str, **kwargs: Any) -> None:
        """Write to JSON file."""
        materialized = self._materialize()
        materialized.write_json(path, **kwargs)

    def to_feather(self, path: str, **kwargs: Any) -> None:
        """Write to Feather file."""
        materialized = self._materialize()
        materialized.write_ipc(path, **kwargs)

    def to_sql(self, name: str, con: Any, **kwargs: Any) -> None:
        """Write to SQL database."""
        materialized = self._materialize()
        materialized.write_database(name, con, **kwargs)

    def info(self) -> None:
        """Print information about the LazyFrame."""
        materialized = self._materialize()
        print(f"LazyFrame shape: {materialized.shape}")
        print(f"Columns: {materialized.columns}")
        print(f"Data types:")
        schema = materialized.schema
        for col, dtype in schema.items():
            print(f"  {col}: {dtype}")

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the underlying LazyFrame."""
        return getattr(self._df, name)
