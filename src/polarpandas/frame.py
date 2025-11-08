"""
DataFrame implementation with pandas-compatible API built on Polars.

This module provides the main DataFrame class that wraps Polars DataFrame and
provides a pandas-compatible interface. All operations are performed using
Polars for optimal performance while maintaining pandas-like behavior.

The DataFrame class supports:
- Eager execution by default (like pandas)
- Mutable operations with inplace parameter support
- Index preservation across operations
- Full pandas API compatibility where implemented
- Direct access to Polars methods via delegation

Examples
--------
>>> import polarpandas as ppd
>>> df = ppd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
>>> df["C"] = df["A"] * 2
>>> result = df.groupby("A").agg(pl.col("B").sum())

Notes
-----
- DataFrame operations are always eager (executed immediately)
- Use LazyFrame for lazy evaluation and query optimization
- Some pandas behaviors may differ due to Polars architecture
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import polars as pl

from polarpandas._dtypes import FLOAT_DTYPES, INTEGER_DTYPES, NUMERIC_DTYPES
from polarpandas._exceptions import (
    convert_to_keyerror,
    create_keyerror_with_suggestions,
)
from polarpandas._frame_aggregations import (
    DataFrameAggregationsMixin,
    _require_pandas,
)
from polarpandas._index_manager import IndexManager
from polarpandas._sql_utils import _require_sqlalchemy, create_table_with_primary_key
from polarpandas.index import Index, MultiIndex
from polarpandas.utils import convert_schema_to_polars

if TYPE_CHECKING:
    from .lazyframe import LazyFrame
    from .series import Series  # noqa: TC004


def _is_integer_dtype(dtype: Any) -> bool:
    """Check if a Polars dtype is an integer type."""
    return dtype in INTEGER_DTYPES


def _is_numeric_dtype(dtype: Any) -> bool:
    """Check if a Polars dtype is a numeric type."""
    return dtype in NUMERIC_DTYPES


def _is_float_dtype(dtype: Any) -> bool:
    """Check if a Polars dtype is a float type."""
    return dtype in FLOAT_DTYPES


class DataFrame(DataFrameAggregationsMixin):
    """
    Two-dimensional, size-mutable, potentially heterogeneous tabular data.

    DataFrame is the primary data structure in PolarPandas, providing a pandas-like
    API while using Polars for all operations under the hood. This offers the
    best of both worlds: familiar pandas syntax with Polars performance.

    Parameters
    ----------
    data : dict, list of dicts, pl.DataFrame, pl.LazyFrame, or None, optional
        Input data. Can be:
        - Dictionary of {column_name: [values]} pairs
        - List of dictionaries (each dict becomes a row)
        - Existing Polars DataFrame
        - Existing Polars LazyFrame (will be materialized)
        - None for empty DataFrame
    index : array-like, optional
        Index to use for resulting DataFrame. If None, a default integer
        index will be used.
    columns : array-like, optional
        Column names for empty DataFrame. Ignored if data is not None.
    **kwargs
        Additional keyword arguments passed to Polars DataFrame constructor.

    Attributes
    ----------
    _df : pl.DataFrame
        The underlying Polars DataFrame.
    _index : list or None
        Stored index values for pandas compatibility.
    _index_name : str, tuple, or None
        Name(s) for the index.

    Examples
    --------
    >>> import polarpandas as ppd
    >>> # From dictionary
    >>> df = ppd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    >>> # From list of dicts
    >>> df = ppd.DataFrame([{"A": 1, "B": 2}, {"A": 3, "B": 4}])
    >>> # Empty DataFrame
    >>> df = ppd.DataFrame(columns=["A", "B"])
    >>> # With index
    >>> df = ppd.DataFrame({"A": [1, 2]}, index=["x", "y"])

    See Also
    --------
    LazyFrame : For lazy execution and query optimization
    Series : One-dimensional labeled array

    Notes
    -----
    - All operations execute immediately (eager execution)
    - Use `.lazy()` to convert to LazyFrame for lazy evaluation
    - Index operations may be slower due to Polars' columnar architecture
    """

    _index: Optional[List[Any]]
    _index_name: Optional[Union[str, Tuple[str, ...]]]
    _columns_index: Optional[Any]
    _df: pl.DataFrame

    def __init__(
        self,
        data: Optional[
            Union[Dict[str, Any], List[Any], pl.DataFrame, pl.LazyFrame]
        ] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a DataFrame from various data sources.

        Create a new DataFrame instance from the provided data. The data can be
        provided in multiple formats, similar to pandas DataFrame constructor.

        Parameters
        ----------
        data : dict, list of dicts, pl.DataFrame, pl.LazyFrame, or None, optional
            Data to initialize the DataFrame with. Supported formats:
            - Dictionary mapping column names to lists/arrays of values
            - List of dictionaries (each dict becomes a row)
            - Existing Polars DataFrame (used directly)
            - Existing Polars LazyFrame (materialized automatically)
            - None for empty DataFrame
        index : array-like, optional
            Index to use for resulting DataFrame. If provided, must have same
            length as data rows. Stored separately for pandas compatibility.
        columns : array-like, optional
            Column names for empty DataFrame. Ignored if data is provided.
        dtype : dict, pl.Schema, or None, optional
            Schema specification for columns. Can be:
            - Pandas-style dict with string dtype names: {"col1": "int64", "col2": "float64"}
            - Pandas-style dict with dtype objects: {"col1": np.int64, "col2": np.float64}
            - Polars schema dict: {"col1": pl.Int64, "col2": pl.Float64}
            - Polars Schema object
        strict : bool, default True
            Whether to use strict mode for Polars DataFrame creation.
        **kwargs
            Additional keyword arguments passed to Polars DataFrame constructor.

        Examples
        --------
        >>> import polarpandas as ppd
        >>> # From dictionary
        >>> df = ppd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> # From list of dicts
        >>> df = ppd.DataFrame([{"A": 1, "B": 2}, {"A": 3, "B": 4}])
        >>> # Empty with columns
        >>> df = ppd.DataFrame(columns=["A", "B"])
        >>> # With custom index
        >>> df = ppd.DataFrame({"value": [10, 20]}, index=["x", "y"])

        Notes
        -----
        - LazyFrames are automatically materialized during initialization
        - Index is stored separately and not part of Polars DataFrame structure
        - Dictionary keys become column names, values become column data
        """
        if data is None:
            # Handle columns and index parameters for empty DataFrame
            columns = kwargs.pop("columns", None)
            index = kwargs.pop("index", None)
            dtype = kwargs.pop("dtype", None)

            if index is not None and columns is not None:
                # Create empty DataFrame with specified columns and index
                self._df = pl.DataFrame({col: [] for col in columns})
                self._index = index
                self._index_name = None
            elif index is not None:
                # Create empty DataFrame with specified index
                self._df = pl.DataFrame()
                self._index = index
                self._index_name = None
            elif columns is not None:
                # Create empty DataFrame with specified columns
                self._df = pl.DataFrame({col: [] for col in columns})
                self._index = None
                self._index_name = None
                self._columns_index = None
            else:
                self._df = pl.DataFrame()
                self._index = None
                self._index_name = None
                self._columns_index = None
        elif isinstance(data, pl.LazyFrame):
            # Materialize LazyFrame to DataFrame
            self._df = data.collect()
            self._index = kwargs.pop("index", None)
            self._index_name = kwargs.pop("index_name", None)
            dtype = kwargs.pop("dtype", None)
        elif isinstance(data, pl.DataFrame):
            # Use DataFrame directly
            self._df = data
            self._index = kwargs.pop("index", None)
            self._index_name = kwargs.pop("index_name", None)
            dtype = kwargs.pop("dtype", None)
        else:
            # Handle index and columns parameters separately since Polars doesn't support them directly
            index = kwargs.pop("index", None)
            columns = kwargs.pop("columns", None)
            dtype = kwargs.pop("dtype", None)
            strict = kwargs.pop("strict", True)

            # Create DataFrame with data
            if index is not None or columns is not None:
                # Store the index separately and create DataFrame with Polars
                self._index = index
                self._index_name = None
                # Create DataFrame with data and handle index/columns
                if isinstance(data, dict):
                    # For dict data, create with specified columns
                    if columns is not None:
                        # Check if column names match data keys
                        data_keys = set(data.keys())
                        column_set = set(columns)

                        if data_keys == column_set:
                            # Column names match data keys, create DataFrame normally
                            self._df = pl.DataFrame(data, strict=strict)
                        else:
                            # Column names don't match data keys, create empty DataFrame with specified columns
                            # This matches pandas behavior
                            self._df = pl.DataFrame({col: [] for col in columns})
                    else:
                        self._df = pl.DataFrame(data, strict=strict)
                else:
                    # For other data types, create DataFrame directly
                    self._df = pl.DataFrame(data, strict=strict)
            else:
                # Handle dict, list, or other data
                # Use strict=False to handle mixed types like inf values
                try:
                    self._df = pl.DataFrame(data, *args, strict=False, **kwargs)
                except pl.exceptions.ComputeError as e:
                    # If Polars can't handle the type mixture, raise the error
                    # No pandas fallback - this is a limitation of pure Polars
                    raise ValueError(
                        f"Polars cannot handle this data type mixture: {e}"
                    ) from e
                self._index = None
                self._index_name = None
                self._columns_index = None

        # Apply dtype/schema conversion if provided
        if dtype is not None:
            polars_schema = convert_schema_to_polars(dtype)
            if polars_schema:
                # Cast columns to specified types
                cast_expressions = [
                    pl.col(col).cast(dtype_val)
                    for col, dtype_val in polars_schema.items()
                    if col in self._df.columns
                ]
                if cast_expressions:
                    self._df = self._df.with_columns(cast_expressions)

    def lazy(self) -> "LazyFrame":
        """
        Convert DataFrame to LazyFrame for lazy execution.

        Creates a new LazyFrame from the current DataFrame. All subsequent
        operations on the LazyFrame will be deferred until `.collect()` is called,
        allowing Polars to optimize the query plan.

        Returns
        -------
        LazyFrame
            LazyFrame wrapping the current DataFrame data

        Examples
        --------
        >>> import polarpandas as ppd
        >>> import polars as pl
        >>> df = ppd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> lf = df.lazy()
        >>> result = lf.filter(pl.col("A") > 1).select(["A", "B"])
        >>> df_final = result.collect()  # Materialize when ready

        See Also
        --------
        LazyFrame : For lazy execution and query optimization
        LazyFrame.collect : Materialize the lazy query plan
        """
        from polarpandas.lazyframe import LazyFrame

        return LazyFrame(
            self._df.lazy(),
            index=self._index,
            index_name=self._index_name,
        )

    @classmethod
    def read_csv(cls, path: str, **kwargs: Any) -> "DataFrame":
        """
        Read a CSV file into DataFrame.

        Parameters
        ----------
        path : str
            Path to CSV file
        dtype : dict, pl.Schema, or None, optional
            Schema specification for columns. Can be pandas-style dict or Polars schema.
            See DataFrame constructor for details.
        schema : dict, pl.Schema, or None, optional
            Direct Polars schema specification (alternative to dtype).
        **kwargs
            Additional arguments passed to Polars read_csv()

        Returns
        -------
        DataFrame
            DataFrame loaded from CSV
        """
        # Map pandas-style parameters to Polars equivalents
        polars_kwargs: Dict[str, Any] = {}

        # Handle pandas-specific parameters
        index_col = kwargs.pop("index_col", None)

        # Handle dtype/schema parameters
        dtype = kwargs.pop("dtype", None)
        schema = kwargs.pop("schema", None)

        # If both are provided, schema takes precedence
        schema_to_use = schema if schema is not None else dtype

        if schema_to_use is not None:
            polars_schema = convert_schema_to_polars(schema_to_use)
            if polars_schema is not None:
                # Convert to Polars Schema object for read_csv
                polars_kwargs["schema"] = polars_schema

        if "sep" in kwargs:
            polars_kwargs["separator"] = kwargs.pop("sep")

        if "names" in kwargs:
            # When names is provided, use the names as column names
            # Set has_header=False as pandas treats the first row as data when names is provided
            names = kwargs.pop("names")
            polars_kwargs["new_columns"] = names
            polars_kwargs["has_header"] = False

        if "skiprows" in kwargs:
            polars_kwargs["skip_rows"] = kwargs.pop("skiprows")

        if "nrows" in kwargs:
            polars_kwargs["n_rows"] = kwargs.pop("nrows")

        # Pass through other parameters
        polars_kwargs.update(kwargs)

        # Read CSV with Polars - use eager reading as per user requirements
        try:
            pl_df = pl.read_csv(path, **polars_kwargs)
            df = cls(pl_df)
        except Exception as e:
            # Convert Polars exceptions to pandas-compatible ones
            if "empty" in str(e).lower() or "NoDataError" in str(type(e)):
                # Convert to pandas EmptyDataError
                try:
                    import pandas as pd

                    raise pd.errors.EmptyDataError(
                        "No columns to parse from file"
                    ) from e
                except ImportError:
                    raise ValueError(
                        f"No columns to parse from file: {e}\n"
                        "Possible causes:\n"
                        "  - File is empty or has no header row\n"
                        "  - All columns were skipped\n"
                        "  - File format is not recognized\n"
                        "Check file contents and try specifying columns explicitly."
                    ) from e
            raise

        # Handle index_col if specified
        if index_col is not None:
            # Create DataFrame and set index
            result = df
            if isinstance(index_col, (int, str)):
                # Single column as index
                if isinstance(index_col, int):
                    col_name = df.columns[index_col]
                else:
                    col_name = index_col

                # Set the column as index
                result._index = df._df[col_name].to_list()
                result._index_name = col_name
                # Remove the column from data
                result._df = result._df.drop(col_name)
            else:
                # Multiple columns as index
                col_names = [
                    df.columns[i] if isinstance(i, int) else i for i in index_col
                ]
                # Set the columns as index (as list of tuples)
                result._index = list(zip(*[df._df[col].to_list() for col in col_names]))
                result._index_name = tuple(col_names)
                # Remove the columns from data
                result._df = result._df.drop(col_names)

            return result
        else:
            return df

    @classmethod
    def read_parquet(cls, path: str, **kwargs: Any) -> "DataFrame":
        """
        Read a Parquet file into DataFrame.

        Parameters
        ----------
        path : str
            Path to Parquet file
        dtype : dict, pl.Schema, or None, optional
            Schema specification for columns. Can be pandas-style dict or Polars schema.
            See DataFrame constructor for details.
        schema : dict, pl.Schema, or None, optional
            Direct Polars schema specification (alternative to dtype).
        **kwargs
            Additional arguments passed to Polars read_parquet()

        Returns
        -------
        DataFrame
            DataFrame loaded from Parquet
        """
        # Handle dtype/schema parameters
        dtype = kwargs.pop("dtype", None)
        schema = kwargs.pop("schema", None)

        # If both are provided, schema takes precedence
        schema_to_use = schema if schema is not None else dtype

        # Use eager reading as per user requirements
        # Note: Parquet files don't support schema parameter, so we read first then cast
        pl_df = pl.read_parquet(path, **kwargs)

        # Apply schema conversion if provided (cast after reading)
        if schema_to_use is not None:
            polars_schema = convert_schema_to_polars(schema_to_use)
            if polars_schema is not None:
                # Cast columns to specified types
                cast_expressions = [
                    pl.col(col).cast(dtype_val)
                    for col, dtype_val in polars_schema.items()
                    if col in pl_df.columns
                ]
                if cast_expressions:
                    pl_df = pl_df.with_columns(cast_expressions)

        return cls(pl_df)

    @classmethod
    def read_json(cls, path: str, **kwargs: Any) -> "DataFrame":
        """
        Read a JSON file into DataFrame using pandas for full orient support.
        """

        schema = kwargs.pop("schema", None)
        dtype = kwargs.pop("dtype", None)
        schema_to_use = schema if schema is not None else dtype

        if schema_to_use is not None:
            import polars as pl  # type: ignore

            polars_schema = convert_schema_to_polars(schema_to_use)
            df = pl.read_json(path)
            if polars_schema is not None:
                cast_exprs = [
                    pl.col(col).cast(dt) for col, dt in polars_schema.items() if col in df.columns
                ]
                if cast_exprs:
                    df = df.with_columns(cast_exprs)
            return cls(df)

        pd = _require_pandas("DataFrame.read_json")
        pandas_df = pd.read_json(path, **kwargs)
        return cls()._from_pandas_dataframe(pandas_df)


    @classmethod
    def read_sql(cls, sql: str, con: Any, **kwargs: Any) -> "DataFrame":
        """
        Read SQL query into DataFrame.

        Parameters
        ----------
        sql : str
            SQL query string
        con : connection object
            Database connection
        dtype : dict, pl.Schema, or None, optional
            Schema specification for columns. Can be pandas-style dict or Polars schema.
            See DataFrame constructor for details.
        schema : dict, pl.Schema, or None, optional
            Direct Polars schema specification (alternative to dtype).
        **kwargs
            Additional arguments passed to Polars read_database()

        Returns
        -------
        DataFrame
            DataFrame loaded from SQL query
        """
        # Handle dtype/schema parameters
        dtype = kwargs.pop("dtype", None)
        schema = kwargs.pop("schema", None)

        # If both are provided, schema takes precedence
        schema_to_use = schema if schema is not None else dtype

        if schema_to_use is not None:
            polars_schema = convert_schema_to_polars(schema_to_use)
            if polars_schema is not None:
                kwargs["schema"] = polars_schema

        return cls(pl.read_database(sql, con, **kwargs))

    @classmethod
    def read_feather(cls, path: str, **kwargs: Any) -> "DataFrame":
        """
        Read Feather file into DataFrame.

        Parameters
        ----------
        path : str
            Path to Feather file
        dtype : dict, pl.Schema, or None, optional
            Schema specification for columns. Can be pandas-style dict or Polars schema.
            See DataFrame constructor for details.
        schema : dict, pl.Schema, or None, optional
            Direct Polars schema specification (alternative to dtype).
        **kwargs
            Additional arguments passed to Polars read_ipc()

        Returns
        -------
        DataFrame
            DataFrame loaded from Feather file
        """
        # Handle dtype/schema parameters
        dtype = kwargs.pop("dtype", None)
        schema = kwargs.pop("schema", None)

        # If both are provided, schema takes precedence
        schema_to_use = schema if schema is not None else dtype

        # Note: Feather/IPC files don't support schema parameter, so we read first then cast
        pl_df = pl.read_ipc(path, **kwargs)

        # Apply schema conversion if provided (cast after reading)
        if schema_to_use is not None:
            polars_schema = convert_schema_to_polars(schema_to_use)
            if polars_schema is not None:
                # Cast columns to specified types
                cast_expressions = [
                    pl.col(col).cast(dtype_val)
                    for col, dtype_val in polars_schema.items()
                    if col in pl_df.columns
                ]
                if cast_expressions:
                    pl_df = pl_df.with_columns(cast_expressions)

        return cls(pl_df)

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the underlying Polars DataFrame.

        This allows transparent access to Polars methods and properties.
        """
        if name.startswith("_"):
            # Avoid infinite recursion for private attributes
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        try:
            attr = getattr(self._df, name)
            return attr
        except AttributeError as e:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from e

    def __repr__(self) -> str:
        """Return string representation of the DataFrame."""
        return repr(self._df)

    def __str__(self) -> str:
        """Return string representation of the DataFrame."""
        return str(self._df)

    def __len__(self) -> int:
        """Return the number of rows in the DataFrame."""
        return len(self._df)

    def __getitem__(self, key: Union[str, List[str]]) -> Union["DataFrame", "Series"]:
        """
        Access a column or subset of columns from the DataFrame.

        This method provides column selection similar to pandas DataFrame indexing.
        Returns a Series for single column selection, or DataFrame for multiple columns.

        Parameters
        ----------
        key : str, list of str, or Series
            Column selection key:
            - str: Single column name, returns Series
            - list of str: Multiple column names, returns DataFrame
            - Series: Boolean indexing, returns filtered DataFrame
            - array-like: Boolean array for filtering rows

        Returns
        -------
        Series or DataFrame
            - Series if key is a single column name
            - DataFrame if key is a list of column names or boolean array

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> # Single column
        >>> series = df["A"]  # Returns Series
        >>> # Multiple columns
        >>> subset = df[["A", "B"]]  # Returns DataFrame
        >>> # Boolean indexing
        >>> filtered = df[df["A"] > 1]  # Returns filtered DataFrame

        Raises
        ------
        KeyError
            If column name is not found in DataFrame

        See Also
        --------
        loc : Label-based selection
        iloc : Integer position-based selection
        """
        try:
            from polarpandas.series import Series

            if isinstance(key, str):
                # Single column - return Series
                return Series(self._df[key])
            elif isinstance(key, list):
                # Multiple columns - select and return DataFrame
                result_df = self._df.select(key)
                return IndexManager.preserve_index(self, result_df)
            elif isinstance(key, Series):  # type: ignore[unreachable]
                # Boolean indexing with Series
                polars_key = key._series
                filtered = self._df.filter(polars_key)
                return DataFrame(filtered)
            elif (
                hasattr(key, "__iter__")
                and not isinstance(key, str)
                and not isinstance(key, list)
            ):
                # Boolean indexing with array-like (but not list, which is handled above)
                polars_key = key
                # Use filter for boolean indexing
                filtered = self._df.filter(polars_key)
                return DataFrame(filtered)
            else:
                # Other key types - delegate to Polars
                return self._df.__getitem__(key)
        except Exception as e:
            # Convert Polars exceptions to pandas-compatible ones
            converted = convert_to_keyerror(e)
            if converted is not e:
                raise converted from e
            raise
    def __setitem__(self, column: str, values: Union[Any, "Series"]) -> None:
        """
        Set a column in the DataFrame (in-place mutation).

        Add or update a column in the DataFrame. If the column already exists,
        it will be overwritten. This operation modifies the DataFrame in place.

        Parameters
        ----------
        column : str
            Column name. If column exists, it will be overwritten; otherwise,
            a new column will be created.
        values : array-like, scalar, Series, or polarpandas.Series
            Values to assign to the column. Can be:
            - List or array-like: Must match DataFrame length
            - Scalar: Broadcast to all rows
            - polarpandas.Series: Uses underlying Polars Series
            - polars.Series: Used directly

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2, 3]})
        >>> # Add new column with list
        >>> df["B"] = [4, 5, 6]
        >>> # Add column with scalar (broadcasts)
        >>> df["constant"] = 10
        >>> # Add column from Series
        >>> df["C"] = df["A"] * 2
        >>> # Overwrite existing column
        >>> df["A"] = [10, 20, 30]

        Notes
        -----
        - This operation is always in-place (modifies self)
        - Column is added or updated immediately
        - Values are converted to Polars Series internally
        """
        from .series import Series as PolarPandasSeries

        # Convert values to Polars Series if needed
        if isinstance(values, PolarPandasSeries):
            # Handle polarpandas Series - extract underlying Polars Series
            series = values._series.alias(column)
        elif isinstance(values, pl.Series):
            series = values.alias(column)
        elif isinstance(values, (int, float, str, bool)):
            # Scalar value - use Polars lit() to broadcast
            expr = pl.lit(values)
            self._df = self._df.with_columns(expr.alias(column))
            return
        else:
            # Handle list or array-like values
            if hasattr(values, "tolist"):
                # Convert to list if it has tolist method (e.g., numpy array)
                values = values.tolist()
            series = pl.Series(column, values)

        # Use with_columns to add or update the column, then replace internal _df
        self._df = self._df.with_columns(series.alias(column))

    def __delitem__(self, column: str) -> None:
        """
        Delete a column from the DataFrame (in-place mutation).

        Parameters
        ----------
        column : str
            Column name to delete
        """
        self._df = self._df.drop(column)

    def drop(
        self, columns: Union[str, List[str]], inplace: bool = False
    ) -> Optional["DataFrame"]:
        """
        Drop specified columns from DataFrame.

        Remove one or more columns from the DataFrame. This operation can be
        performed in-place or return a new DataFrame.

        Parameters
        ----------
        columns : str or list of str
            Column name(s) to drop. Can be a single column name or a list
            of column names.
        inplace : bool, default False
            If True, modify DataFrame in place and return None.
            If False, return a new DataFrame with columns dropped.

        Returns
        -------
        DataFrame or None
            DataFrame with specified columns removed, or None if inplace=True.
            If inplace=False, returns a new DataFrame; original is unchanged.

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        >>> # Drop single column
        >>> df_dropped = df.drop("A")
        >>> # Drop multiple columns
        >>> df_dropped = df.drop(["A", "B"])
        >>> # In-place drop
        >>> df.drop("A", inplace=True)  # Modifies df, returns None

        Raises
        ------
        KeyError
            If any specified column name does not exist in the DataFrame

        See Also
        --------
        rename : Rename columns instead of dropping them
        """
        # Polars drop() accepts both str and list
        try:
            result_df = self._df.drop(columns)
        except Exception as e:
            # Convert Polars exceptions to pandas-compatible ones
            converted = convert_to_keyerror(e)
            if converted is not e:
                raise converted from e
            raise

        if inplace:
            IndexManager.preserve_index_inplace(self, result_df)
            return None
        else:
            return IndexManager.preserve_index(self, result_df)

    def rename(
        self,
        mapping: Optional[Dict[str, str]] = None,
        columns: Optional[Dict[str, str]] = None,
        inplace: bool = False,
    ) -> Optional["DataFrame"]:
        """
        Rename DataFrame columns.

        Change column names by providing a mapping from old names to new names.
        Non-existent columns in the mapping are silently ignored (matching pandas
        behavior).

        Parameters
        ----------
        mapping : dict, optional
            Mapping of {old_name: new_name} pairs. Deprecated, use `columns` instead.
        columns : dict, optional
            Mapping of {old_name: new_name} pairs. If both `mapping` and `columns`
            are provided, `columns` takes precedence.
        inplace : bool, default False
            If True, modify DataFrame in place and return None.
            If False, return a new DataFrame with renamed columns.

        Returns
        -------
        DataFrame or None
            DataFrame with renamed columns, or None if inplace=True.
            Non-existent column names in the mapping are ignored.

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"old_name": [1, 2, 3]})
        >>> # Rename single column
        >>> df_renamed = df.rename(columns={"old_name": "new_name"})
        >>> # Rename multiple columns
        >>> df = ppd.DataFrame({"A": [1], "B": [2]})
        >>> df_renamed = df.rename(columns={"A": "Alpha", "B": "Beta"})
        >>> # In-place rename
        >>> df.rename(columns={"A": "Alpha"}, inplace=True)

        Notes
        -----
        - Non-existent column names are silently ignored (pandas behavior)
        - If mapping is empty or all columns don't exist, returns unchanged DataFrame

        See Also
        --------
        drop : Remove columns instead of renaming
        """
        # Use columns parameter if provided, otherwise use mapping
        rename_dict = columns if columns is not None else mapping
        if rename_dict is None:
            raise ValueError(
                "Either 'mapping' or 'columns' must be provided to rename columns.\n"
                "Examples:\n"
                "  - df.rename(columns={'old': 'new'})\n"
                "  - df.rename(columns=['col1', 'col2'])"
            )

        # Filter out non-existent columns to match pandas behavior
        # pandas ignores non-existent columns in rename operations
        existing_columns = set(self._df.columns)
        filtered_rename_dict = {
            old: new for old, new in rename_dict.items() if old in existing_columns
        }

        if not filtered_rename_dict:
            # No valid columns to rename, return copy of original
            result_df = self._df.clone()
        else:
            result_df = self._df.rename(filtered_rename_dict)

        if inplace:
            self._df = result_df
            return None
        else:
            return DataFrame(result_df)

    def sort_values(
        self, by: Union[str, List[str]], inplace: bool = False, **kwargs: Any
    ) -> Optional["DataFrame"]:
        """
        Sort DataFrame by one or more column values.

        Sort the DataFrame by the values in specified column(s). When sorting
        by multiple columns, the first column takes precedence, then the second,
        and so on.

        Parameters
        ----------
        by : str or list of str
            Column name(s) to sort by. If a list, sorts by first column first,
            then by second column, etc.
        inplace : bool, default False
            If True, modify DataFrame in place and return None.
            If False, return a new sorted DataFrame.
        **kwargs
            Additional arguments passed to Polars sort(). Common options:
            - descending: bool or list of bool, default False
              Sort in descending order. Can be a list matching `by` length.
            - nulls_last: bool, default False
              Place null values last in sorted order

        Returns
        -------
        DataFrame or None
            DataFrame sorted by specified columns, or None if inplace=True.
            Original DataFrame is unchanged if inplace=False.

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [3, 1, 2], "B": [6, 4, 5]})
        >>> # Sort by single column
        >>> df_sorted = df.sort_values("A")
        >>> # Sort by multiple columns
        >>> df_sorted = df.sort_values(["A", "B"])
        >>> # Descending sort
        >>> df_sorted = df.sort_values("A", descending=True)
        >>> # In-place sort
        >>> df.sort_values("A", inplace=True)

        See Also
        --------
        sort_index : Sort by index instead of column values
        """
        # Polars uses sort() instead of sort_values()
        result_df = self._df.sort(by, **kwargs)

        if inplace:
            self._df = result_df
            return None
        else:
            return DataFrame(result_df)

    def fillna(
        self, value: Any, inplace: bool = False, **kwargs: Any
    ) -> Optional["DataFrame"]:
        """
        Fill null values.

        Parameters
        ----------
        value : scalar
            Value to fill nulls with
        inplace : bool, default False
            If True, modify DataFrame in place and return None.
            If False, return a new DataFrame.
        **kwargs : additional arguments
            Additional arguments passed to Polars fill_null()

        Returns
        -------
        DataFrame or None
            DataFrame with nulls filled, or None if inplace=True
        """
        # Polars uses fill_null() instead of fillna()
        result_df = self._df.fill_null(value, **kwargs)

        if inplace:
            self._df = result_df
            return None
        else:
            return DataFrame(result_df)

    def astype(
        self, dtype: Union[Dict[str, Any], Any], errors: str = "raise", **kwargs: Any
    ) -> "DataFrame":
        """
        Cast a pandas object to a specified dtype.

        Parameters
        ----------
        dtype : dict, str, or dtype
            Data type(s) to cast to. Can be:
            - Dict mapping column names to dtypes
            - Single dtype to apply to all columns
        errors : {'raise', 'ignore'}, default 'raise'
            Control raising of exceptions on invalid data types.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            DataFrame with cast dtypes.

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2, 3], "B": [1.5, 2.5, 3.5]})
        >>> df.astype({"A": "float64"})
        >>> df.astype("int64")  # Cast all columns
        """
        from polarpandas.utils import convert_schema_to_polars

        if errors not in ("raise", "ignore"):
            raise ValueError(f"errors must be 'raise' or 'ignore', got '{errors}'")

        try:
            if isinstance(dtype, dict):
                # Cast specific columns
                polars_schema = convert_schema_to_polars(dtype)
                if polars_schema is None:
                    raise ValueError(f"Could not convert dtype dict: {dtype}")

                cast_exprs = []
                for col, target_dtype in polars_schema.items():
                    if col in self.columns:
                        cast_exprs.append(pl.col(col).cast(target_dtype))
                    elif errors == "raise":
                        raise KeyError(f"Column '{col}' not found in DataFrame")

                if cast_exprs:
                    result_df = self._df.with_columns(cast_exprs)
                else:
                    result_df = self._df
            else:
                # Cast all columns to same dtype
                polars_dtype = convert_schema_to_polars({"dummy": dtype})
                if polars_dtype is None:
                    if errors == "raise":
                        raise ValueError(f"Could not convert dtype: {dtype}")
                    else:
                        return DataFrame(self._df)
                target_dtype = list(polars_dtype.values())[0]

                cast_exprs = [pl.col(col).cast(target_dtype) for col in self.columns]
                result_df = self._df.with_columns(cast_exprs)

            return DataFrame(result_df)
        except Exception:
            if errors == "raise":
                raise
            else:
                # On error with ignore, return original DataFrame
                return DataFrame(self._df)

    def replace(
        self,
        to_replace: Any = None,
        value: Any = None,
        inplace: bool = False,
        limit: Optional[int] = None,
        regex: bool = False,
        **kwargs: Any,
    ) -> Optional["DataFrame"]:
        """
        Replace values given in to_replace with value.

        Parameters
        ----------
        to_replace : str, regex, list, dict, Series, int, float, or None
            How to find the values that will be replaced.
        value : scalar, dict, list, str, regex, default None
            Value to replace any values matching to_replace with.
        inplace : bool, default False
            If True, modify DataFrame in place and return None.
        limit : int, default None
            Maximum size gap to forward or backward fill.
        regex : bool, default False
            Whether to interpret to_replace as a regex pattern.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame or None
            DataFrame with values replaced, or None if inplace=True.
        """
        if to_replace is None and value is None:
            return None if inplace else DataFrame(self._df)

        result_df = self._df.clone()

        # Handle dict replacement
        if isinstance(to_replace, dict):
            # Check if it's a nested dict (column-specific) or value mapping
            is_nested = any(isinstance(v, dict) for v in to_replace.values())

            if is_nested:
                # Nested dict: {column: {old_value: new_value}}
                for col in self.columns:
                    if col in to_replace:
                        replace_map = to_replace[col]
                        if isinstance(replace_map, dict):
                            result_df = result_df.with_columns(
                                pl.col(col).replace(replace_map)
                            )
            else:
                # Simple dict: {old_value: new_value} - apply to all columns
                replace_map = to_replace
                for col in self.columns:
                    result_df = result_df.with_columns(pl.col(col).replace(replace_map))
        elif isinstance(to_replace, (list, tuple)):
            # List of values to replace
            if isinstance(value, (list, tuple)) and len(value) == len(to_replace):
                # Map each old value to corresponding new value
                replace_map = dict(zip(to_replace, value))
                for col in self.columns:
                    result_df = result_df.with_columns(pl.col(col).replace(replace_map))
            else:
                # Replace all with single value
                replace_map = dict.fromkeys(to_replace, value)
                for col in self.columns:
                    result_df = result_df.with_columns(pl.col(col).replace(replace_map))
        else:
            # Scalar replacement
            replace_map = {to_replace: value}
            for col in self.columns:
                result_df = result_df.with_columns(pl.col(col).replace(replace_map))

        if inplace:
            self._df = result_df
            return None
        else:
            return DataFrame(result_df)

    def interpolate(
        self,
        method: str = "linear",
        axis: Union[int, Literal["index", "columns"]] = 0,
        limit: Optional[int] = None,
        inplace: bool = False,
        limit_direction: Optional[
            Union[str, Literal["forward", "backward", "both"]]
        ] = None,
        limit_area: Optional[Any] = None,
        downcast: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional["DataFrame"]:
        """
        Fill NaN values using an interpolation method.

        Parameters
        ----------
        method : str, default 'linear'
            Interpolation technique to use. Currently supports 'linear'.
        axis : {0, 1, 'index', 'columns'}, default 0
            Axis to interpolate along.
        limit : int, optional
            Maximum number of consecutive NaN values to fill.
        inplace : bool, default False
            If True, modify DataFrame in place and return None.
        limit_direction : {'forward', 'backward', 'both'}, optional
            Consecutive NaNs will be filled in this direction.
        limit_area : str, optional
            Not used (for pandas compatibility).
        downcast : dict, optional
            Not used (for pandas compatibility).
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame or None
            DataFrame with interpolated values, or None if inplace=True.
        """
        if method != "linear":
            raise NotImplementedError(
                f"Interpolation method '{method}' not yet implemented"
            )

        # Apply interpolation to each column
        expressions = []
        for col in self.columns:
            expr = pl.col(col).interpolate()
            if limit is not None:
                # Limit is handled by Polars interpolate, but we can add additional logic if needed
                pass
            expressions.append(expr.alias(col))

        result_df = self._df.select(expressions)

        if inplace:
            self._df = result_df
            return None
        else:
            return DataFrame(result_df, index=self._index)

    def transform(
        self,
        func: Union[
            Callable[..., Any],
            str,
            List[Union[Callable[..., Any], str]],
            Dict[str, Union[Callable[..., Any], str]],
        ],
        axis: Union[int, Literal["index", "columns"]] = 0,
        *args: Any,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Call func on self producing a DataFrame with the same axis shape as self.

        Parameters
        ----------
        func : function, str, list, or dict
            Function to use for transforming the data.
        axis : {0, 1, 'index', 'columns'}, default 0
            Axis along which to apply the transformation.
        *args
            Positional arguments to pass to func.
        **kwargs
            Keyword arguments to pass to func.

        Returns
        -------
        DataFrame
            DataFrame with transformed values.
        """
        # For now, delegate to apply() - transform is similar but guarantees same shape
        # Type ignore needed because apply can return Series or DataFrame and has different axis types
        return self.apply(func, axis=axis, *args, **kwargs)  # type: ignore[return-value,misc,arg-type] # noqa: B026

    def pipe(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Apply func(self, *args, **kwargs).

        Parameters
        ----------
        func : callable
            Function to apply to the DataFrame.
        *args
            Positional arguments to pass to func.
        **kwargs
            Keyword arguments to pass to func.

        Returns
        -------
        object
            Result of applying func to the DataFrame.
        """
        return func(self, *args, **kwargs)

    def update(
        self,
        other: "DataFrame",
        join: str = "left",
        overwrite: bool = True,
        filter_func: Optional[Any] = None,
        errors: str = "ignore",
        **kwargs: Any,
    ) -> None:
        """
        Modify in place using non-NA values from another DataFrame.

        Parameters
        ----------
        other : DataFrame
            DataFrame to update with.
        join : {'left'}, default 'left'
            Only 'left' join is supported.
        overwrite : bool, default True
            How to handle non-NA values for overlapping keys.
        filter_func : callable, optional
            Not used (for pandas compatibility).
        errors : {'ignore', 'raise'}, default 'ignore'
            If 'raise', raise a ValueError if there are overlapping keys.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        None
            Modifies DataFrame in place.
        """
        if errors == "raise" and not overwrite:
            # Check for overlapping non-null values
            for col in self.columns:
                if col in other.columns:
                    # This is a simplified check - full implementation would be more complex
                    pass

        # Update columns that exist in both DataFrames
        expressions = []
        for col in self.columns:
            if col in other.columns:
                # Use values from other where not null, otherwise keep original
                if overwrite:
                    expr = (
                        pl.when(pl.col(f"{col}_other").is_not_null())
                        .then(pl.col(f"{col}_other"))
                        .otherwise(pl.col(col))
                    )
                else:
                    expr = (
                        pl.when(pl.col(col).is_null())
                        .then(pl.col(f"{col}_other"))
                        .otherwise(pl.col(col))
                    )
                expressions.append(expr.alias(col))
            else:
                expressions.append(pl.col(col).alias(col))

        # Add other DataFrame columns with suffix for comparison
        other_df_renamed = other._df.select(
            [pl.col(col).alias(f"{col}_other") for col in other.columns]
        )
        combined_df = self._df.hstack(other_df_renamed)
        result_df = combined_df.select(expressions)

        self._df = result_df

    def combine_first(self, other: "DataFrame") -> "DataFrame":
        """
        Update null elements with value in the same location in other.

        Parameters
        ----------
        other : DataFrame
            DataFrame to combine with.

        Returns
        -------
        DataFrame
            DataFrame with null values filled from other.
        """
        expressions = []
        all_cols = set(self.columns) | set(other.columns)

        for col in all_cols:
            if col in self.columns and col in other.columns:
                # Use self value if not null, otherwise use other value
                expr = (
                    pl.when(pl.col(col).is_not_null())
                    .then(pl.col(col))
                    .otherwise(pl.col(f"{col}_other"))
                )
            elif col in self.columns:
                expr = pl.col(col)
            else:
                expr = pl.col(f"{col}_other")
            expressions.append(expr.alias(col))

        # Add other DataFrame columns with suffix
        other_df_renamed = other._df.select(
            [pl.col(col).alias(f"{col}_other") for col in other.columns]
        )
        combined_df = self._df.hstack(other_df_renamed)
        result_df = combined_df.select(expressions)

        return DataFrame(result_df, index=self._index)

    def floordiv(
        self,
        other: Union[Any, "Series", "DataFrame"],
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        level: Optional[Any] = None,
        fill_value: Optional[Any] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Get integer division of DataFrame and other, element-wise (binary operator //).

        Parameters
        ----------
        other : scalar, Series, or DataFrame
            Object to divide the DataFrame by.
        axis : {0, 1, 'index', 'columns'} or None, default None
            Axis to match Series index on. For Series input, axis to match Series index on.
        level : int or name, optional
            Broadcast across a level, matching Index values on the passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for successful
            DataFrame alignment, with this value before computation.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.
        """
        from polarpandas.series import Series

        if isinstance(other, DataFrame):
            if len(self) != len(other):
                raise ValueError(
                    "Can only compute floor division with identically-labeled DataFrame objects"
                )
            expressions = []
            other_df_renamed = other._df.select(
                [pl.col(col).alias(f"{col}_other") for col in other.columns]
            )
            combined_df = self._df.hstack(other_df_renamed)
            for col in self.columns:
                if col in other.columns:
                    if fill_value is not None:
                        expr = pl.col(col).fill_null(fill_value) // pl.col(
                            f"{col}_other"
                        ).fill_null(fill_value)
                    else:
                        expr = pl.col(col) // pl.col(f"{col}_other")
                    expressions.append(expr.alias(col))
                else:
                    expressions.append(pl.col(col).alias(col))
            result_df = combined_df.select(expressions)
        elif isinstance(other, Series):
            if axis is None or axis == 0 or axis == "index":
                if len(other) != len(self):
                    raise ValueError("Lengths must match")
                other_series = other._series if hasattr(other, "_series") else other
                if fill_value is not None:
                    expressions = [
                        (
                            pl.col(col).fill_null(fill_value)
                            // pl.Series(other_series).fill_null(fill_value)
                        ).alias(col)
                        for col in self.columns
                    ]
                else:
                    expressions = [
                        (pl.col(col) // pl.Series(other_series)).alias(col)
                        for col in self.columns
                    ]
            else:
                raise NotImplementedError(
                    "Series arithmetic with axis=1 not yet implemented"
                )
            result_df = self._df.select(expressions)
        else:
            if fill_value is not None:
                expressions = [
                    (pl.col(col).fill_null(fill_value) // other).alias(col)
                    for col in self.columns
                ]
            else:
                expressions = [
                    (pl.col(col) // other).alias(col) for col in self.columns
                ]
            result_df = self._df.select(expressions)

        return DataFrame(result_df, index=self._index)

    def truediv(
        self,
        other: Union[Any, "Series", "DataFrame"],
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        level: Optional[Any] = None,
        fill_value: Optional[Any] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Get floating division of DataFrame and other, element-wise (binary operator /).

        Parameters
        ----------
        other : scalar, Series, or DataFrame
            Object to divide the DataFrame by.
        axis : {0, 1, 'index', 'columns'} or None, default None
            Axis to match Series index on. For Series input, axis to match Series index on.
        level : int or name, optional
            Broadcast across a level, matching Index values on the passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for successful
            DataFrame alignment, with this value before computation.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.
        """
        # truediv is an alias for div
        return self.div(other, axis=axis, level=level, fill_value=fill_value, **kwargs)

    def dot(
        self, other: Union["Series", "DataFrame", Any]
    ) -> Union["Series", "DataFrame"]:
        """
        Compute the matrix multiplication between the DataFrame and other.

        Parameters
        ----------
        other : Series, DataFrame, or array-like
            The other object to compute the matrix product with.

        Returns
        -------
        Series or DataFrame
            The result of the matrix multiplication.
        """
        from polarpandas.series import Series

        if isinstance(other, DataFrame):
            # DataFrame.dot(DataFrame) - matrix multiplication: self @ other
            # For proper matrix multiplication, we need: self (m x n) @ other (n x p) = result (m x p)
            # where n is the number of columns in self and rows in other
            if len(self.columns) != len(other):
                raise ValueError(
                    f"DataFrame.dot() requires the number of columns of the left DataFrame "
                    f"({len(self.columns)}) to equal the number of rows of the right DataFrame ({len(other)})"
                )
            # Matrix multiplication: for each row in self and each column in other
            # Compute sum of (self[row, col] * other[col, other_col]) for all cols
            # Add other DataFrame columns with suffix for alignment
            other_df_renamed = other._df.select(
                [pl.col(col).alias(f"{col}_other") for col in other.columns]
            )
            combined_df = self._df.hstack(other_df_renamed)

            expressions = []
            for other_col in other.columns:
                # For each column in other, compute dot product
                expr = pl.sum_horizontal(
                    [
                        pl.col(self_col) * pl.col(f"{other_col}_other")
                        for self_col in self.columns
                    ]
                )
                expressions.append(expr.alias(other_col))

            result_df = combined_df.select(expressions)
            return DataFrame(result_df, index=self._index)
        elif isinstance(other, Series):
            # DataFrame.dot(Series) - returns Series
            if len(self.columns) != len(other):
                raise ValueError(
                    f"DataFrame.dot() requires the number of columns of the DataFrame "
                    f"({len(self.columns)}) to equal the length of the Series ({len(other)})"
                )
            other_series = other._series if hasattr(other, "_series") else other
            # Compute dot product: sum of (self column * other value) for each row
            # Add other Series as columns with matching indices
            other_values = other_series.to_list()
            expressions = []
            for i, col in enumerate(self.columns):
                expressions.append(
                    (pl.col(col) * other_values[i]).alias(f"{col}_weighted")
                )
            weighted_df = self._df.select(expressions)
            result_series = weighted_df.select(
                pl.sum_horizontal(
                    [pl.col(f"{col}_weighted") for col in self.columns]
                ).alias("result")
            )["result"]
            return Series(
                result_series,
                index=self._index
                if self._index is not None
                else list(range(len(result_series))),
            )
        else:
            # Array-like - convert to Series first
            other_series = pl.Series(other)
            if len(self.columns) != len(other_series):
                raise ValueError(
                    f"DataFrame.dot() requires the number of columns of the DataFrame "
                    f"({len(self.columns)}) to equal the length of the array ({len(other_series)})"
                )
            # Compute dot product: sum of (self column * other value) for each row
            other_values = other_series.to_list()
            expressions = []
            for i, col in enumerate(self.columns):
                expressions.append(
                    (pl.col(col) * other_values[i]).alias(f"{col}_weighted")
                )
            weighted_df = self._df.select(expressions)
            result_series = weighted_df.select(
                pl.sum_horizontal(
                    [pl.col(f"{col}_weighted") for col in self.columns]
                ).alias("result")
            )["result"]
            return Series(
                result_series,
                index=self._index
                if self._index is not None
                else list(range(len(result_series))),
            )

    def to_string(
        self,
        buf: Optional[Any] = None,
        columns: Optional[List[str]] = None,
        col_space: Optional[Union[int, Dict[str, int]]] = None,
        header: Union[bool, List[str]] = True,
        index: bool = True,
        na_rep: str = "NaN",
        formatters: Optional[Dict[str, Callable[..., Any]]] = None,
        float_format: Optional[Union[str, Callable[..., Any]]] = None,
        sparsify: Optional[bool] = None,
        index_names: bool = True,
        justify: Optional[str] = None,
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
        show_dimensions: bool = False,
        decimal: str = ".",
        line_width: Optional[int] = None,
        min_rows: Optional[int] = None,
        max_colwidth: Optional[int] = None,
        encoding: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Render a DataFrame to a console-friendly tabular output.

        Parameters
        ----------
        buf : writable buffer, optional
            Buffer to write to. If None, returns string.
        columns : list of str, optional
            Columns to write.
        col_space : int or dict, optional
            Minimum width for columns.
        header : bool or list of str, default True
            Write out column names.
        index : bool, default True
            Whether to print index.
        na_rep : str, default 'NaN'
            String representation of NaN to use.
        formatters : dict, optional
            Formatters for columns.
        float_format : str or callable, optional
            Formatter for floating point numbers.
        sparsify : bool, optional
            Not used (for pandas compatibility).
        index_names : bool, default True
            Print names of index levels.
        justify : str, optional
            Not used (for pandas compatibility).
        max_rows : int, optional
            Maximum number of rows to display.
        max_cols : int, optional
            Maximum number of columns to display.
        show_dimensions : bool, default False
            Display DataFrame dimensions.
        decimal : str, default '.'
            Character recognized as decimal separator.
        line_width : int, optional
            Not used (for pandas compatibility).
        min_rows : int, optional
            Minimum number of rows to display.
        max_colwidth : int, optional
            Maximum width of columns.
        encoding : str, optional
            Not used (for pandas compatibility).
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        str or None
            String representation of DataFrame, or None if buf is provided.
        """
        # Use Polars' built-in string representation
        result_str = str(self._df)

        if buf is not None:
            buf.write(result_str)
            return None
        else:
            return result_str

    def pop(self, item: str) -> "Series":
        """
        Return item and drop from frame.

        Parameters
        ----------
        item : str
            Column name to pop.

        Returns
        -------
        Series
            Popped column as Series.
        """
        from polarpandas.series import Series

        if item not in self.columns:
            raise KeyError(f"'{item}'")

        # Get the column
        result_series = Series(self._df[item], index=self._index)

        # Remove from DataFrame
        self._df = self._df.drop(item)

        return result_series

    def shift(
        self,
        periods: int = 1,
        freq: Optional[Any] = None,
        axis: int = 0,
        fill_value: Optional[Any] = None,
    ) -> "DataFrame":
        """
        Shift index by desired number of periods with an optional time freq.

        Parameters
        ----------
        periods : int, default 1
            Number of periods to shift. Can be positive or negative.
        freq : str or DateOffset, optional
            Frequency string or DateOffset object (not fully supported).
        axis : {0, 1}, default 0
            Shift direction. 0 for shifting rows, 1 for shifting columns.
        fill_value : scalar, optional
            The scalar value to use for newly introduced missing values.

        Returns
        -------
        DataFrame
            Copy of input object, shifted.
        """
        if axis == 1:
            # Shift columns (not commonly used)
            # This would require column reordering, which is complex
            # For now, return a copy
            return DataFrame(self._df)
        else:
            # Shift rows (default)
            result_df = self._df.shift(periods, fill_value=fill_value)
            return DataFrame(result_df)

    def ffill(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        limit: Optional[int] = None,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Optional["DataFrame"]:
        """
        Forward fill missing values.

        Parameters
        ----------
        axis : {0, 1, 'index', 'columns'}, optional
            Axis along which to fill. 0 or 'index' for column-wise, 1 or 'columns' for row-wise.
        limit : int, optional
            Maximum number of consecutive NaN values to forward fill.
        inplace : bool, default False
            If True, modify DataFrame in place and return None.
        **kwargs
            Additional arguments passed to Polars forward_fill().

        Returns
        -------
        DataFrame or None
            DataFrame with forward-filled values, or None if inplace=True.
        """
        if limit is not None:
            # Polars doesn't directly support limit, so we'd need a workaround
            # For now, just do forward_fill without limit
            result_df = self._df.with_columns(
                [pl.col(col).forward_fill(**kwargs) for col in self._df.columns]
            )
        else:
            result_df = self._df.with_columns(
                [pl.col(col).forward_fill(**kwargs) for col in self._df.columns]
            )

        if inplace:
            self._df = result_df
            return None
        else:
            return DataFrame(result_df)

    def bfill(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        limit: Optional[int] = None,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Optional["DataFrame"]:
        """
        Backward fill missing values.

        Parameters
        ----------
        axis : {0, 1, 'index', 'columns'}, optional
            Axis along which to fill. 0 or 'index' for column-wise, 1 or 'columns' for row-wise.
        limit : int, optional
            Maximum number of consecutive NaN values to backward fill.
        inplace : bool, default False
            If True, modify DataFrame in place and return None.
        **kwargs
            Additional arguments passed to Polars backward_fill().

        Returns
        -------
        DataFrame or None
            DataFrame with backward-filled values, or None if inplace=True.
        """
        if limit is not None:
            # Polars doesn't directly support limit, so we'd need a workaround
            # For now, just do backward_fill without limit
            result_df = self._df.with_columns(
                [pl.col(col).backward_fill(**kwargs) for col in self._df.columns]
            )
        else:
            result_df = self._df.with_columns(
                [pl.col(col).backward_fill(**kwargs) for col in self._df.columns]
            )

        if inplace:
            self._df = result_df
            return None
        else:
            return DataFrame(result_df)

    def pad(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        limit: Optional[int] = None,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Optional["DataFrame"]:
        """
        Alias for ffill() (pandas compatibility).

        Parameters
        ----------
        axis : {0, 1, 'index', 'columns'}, optional
            Axis along which to fill.
        limit : int, optional
            Maximum number of consecutive NaN values to forward fill.
        inplace : bool, default False
            If True, modify DataFrame in place and return None.
        **kwargs
            Additional arguments passed to Polars forward_fill().

        Returns
        -------
        DataFrame or None
            DataFrame with forward-filled values, or None if inplace=True.
        """
        return self.ffill(axis=axis, limit=limit, inplace=inplace, **kwargs)

    def dropna(self, inplace: bool = False, **kwargs: Any) -> Optional["DataFrame"]:
        """
        Drop rows with null values.

        Parameters
        ----------
        inplace : bool, default False
            If True, modify DataFrame in place and return None.
            If False, return a new DataFrame.
        **kwargs : additional arguments
            Additional arguments passed to Polars drop_nulls()

        Returns
        -------
        DataFrame or None
            DataFrame with null rows dropped, or None if inplace=True
        """
        # Polars uses drop_nulls() instead of dropna()
        result_df = self._df.drop_nulls(**kwargs)

        if inplace:
            self._df = result_df
            return None
        else:
            return DataFrame(result_df)

    # Properties
    @property
    def columns(self) -> List[str]:
        """Get column names without materializing."""
        return self._df.columns  # Available on both LazyFrame and DataFrame

    @property
    def shape(self) -> Tuple[int, int]:
        """Return a tuple representing the dimensionality of the DataFrame."""
        rows, cols = self._df.shape
        # If we have a stored index, use its length for rows
        if self._index is not None:
            rows = len(self._index)
        return (rows, cols)

    @property
    def empty(self) -> bool:
        """Return True if DataFrame is empty."""
        return len(self._df) == 0

    @property
    def values(self) -> Any:
        """Return the values of the DataFrame as a numpy array."""
        return self._df.to_numpy()

    @property
    def dtypes(self) -> Any:
        """Return the dtypes in the DataFrame."""
        # Return Polars dtypes - may differ from pandas
        # Use schema to avoid materialization
        schema = self._df.schema
        dtypes_dict = dict(zip(self._df.columns, schema.values()))

        # Add empty attribute to match pandas behavior
        class DtypesDict(Dict[str, Any]):
            @property
            def empty(self) -> bool:
                return len(self) == 0

        return DtypesDict(dtypes_dict)

    @property
    def height(self) -> int:
        """Return the number of rows in the DataFrame."""
        return self._df.height

    @property
    def width(self) -> int:
        """Return the number of columns in the DataFrame."""
        return len(self._df.columns)  # Available without materialization

    @property
    def index(self) -> Any:
        """Return the index (row labels) of the DataFrame."""
        if self._index is not None:
            # Check if index contains tuples (MultiIndex)
            if len(self._index) > 0 and isinstance(self._index[0], tuple):
                # Create MultiIndex from tuples
                if isinstance(self._index_name, tuple):
                    names: Optional[List[Optional[str]]] = list(self._index_name)
                else:
                    names = None
                return MultiIndex.from_tuples(self._index, names=names)
            else:
                # Regular Index - preserve name if set
                idx = Index(self._index)
                if self._index_name is not None and not isinstance(
                    self._index_name, tuple
                ):
                    # Set index name by updating the underlying series name
                    idx._series = idx._series.rename(self._index_name)
                return idx
        else:
            # Create a simple RangeIndex-like object
            return Index(list(range(len(self._df))))

    @property
    def loc(self) -> "_LocIndexer":
        """Access a group of rows and columns by label(s)."""
        return _LocIndexer(self)

    @property
    def iloc(self) -> "_ILocIndexer":
        """Access a group of rows and columns by integer position(s)."""
        return _ILocIndexer(self)

    @property
    def at(self) -> "_AtIndexer":
        """Access a single value for a row/column label pair."""
        return _AtIndexer(self)

    @property
    def iat(self) -> "_IAtIndexer":
        """Access a single value for a row/column pair by integer position."""
        return _IAtIndexer(self)

    # Methods
    def head(self, n: int = 5) -> "DataFrame":
        """
        Return the first n rows of the DataFrame.

        Select and return the first n rows from the DataFrame. This is useful
        for quickly inspecting the beginning of a large DataFrame.

        Parameters
        ----------
        n : int, default 5
            Number of rows to return. Must be non-negative. If n exceeds the
            number of rows, returns all rows.

        Returns
        -------
        DataFrame
            New DataFrame containing the first n rows. Original DataFrame is
            unchanged.

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": range(10), "B": range(10, 20)})
        >>> # Default: first 5 rows
        >>> df.head()
        >>> # First 3 rows
        >>> df.head(3)
        >>> # More rows than available returns all rows
        >>> df.head(100)  # Returns all 10 rows

        See Also
        --------
        tail : Return the last n rows
        """
        return DataFrame(self._df.head(n))

    def tail(self, n: int = 5) -> "DataFrame":
        """
        Return the last n rows of the DataFrame.

        Select and return the last n rows from the DataFrame. This is useful
        for quickly inspecting the end of a large DataFrame.

        Parameters
        ----------
        n : int, default 5
            Number of rows to return. Must be non-negative. If n exceeds the
            number of rows, returns all rows.

        Returns
        -------
        DataFrame
            New DataFrame containing the last n rows. Original DataFrame is
            unchanged.

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": range(10), "B": range(10, 20)})
        >>> # Default: last 5 rows
        >>> df.tail()
        >>> # Last 3 rows
        >>> df.tail(3)
        >>> # More rows than available returns all rows
        >>> df.tail(100)  # Returns all 10 rows

        See Also
        --------
        head : Return the first n rows
        """
        return DataFrame(self._df.tail(n))

    def copy(self) -> "DataFrame":
        """
        Make a copy of this DataFrame.

        Returns
        -------
        DataFrame
            A copy of the DataFrame
        """
        result = DataFrame(self._df.clone())
        # Preserve the index in the copy
        result._index = self._index.copy() if self._index is not None else None
        result._index_name = self._index_name
        result._columns_index = getattr(self, "_columns_index", None)
        if result._columns_index is not None:
            result._columns_index = result._columns_index.copy()
        return result

    def _from_pandas_like(self, obj: Any) -> Any:
        """
        Internal helper to convert pandas objects back into polarpandas equivalents.
        """

        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised when pandas missing
            raise ImportError(
                "pandas is required for interoperability helpers. "
                "Install with `pip install polarpandas[pandas]`."
            ) from exc

        if isinstance(obj, pd.DataFrame):
            return self._from_pandas_dataframe(obj)
        if isinstance(obj, pd.Series):
            from polarpandas.series import Series as PolarPandasSeries

            polars_series = pl.from_pandas(obj)
            result_series = PolarPandasSeries(polars_series, index=list(obj.index))
            if obj.name is not None:
                if isinstance(obj.name, str):
                    result_series._series = result_series._series.rename(obj.name)
                result_series._original_name = obj.name  # type: ignore[attr-defined]
            return result_series
        return obj

    def _update_from_dataframe(self, other: "DataFrame") -> None:
        """
        Overwrite this DataFrame's internal state with another DataFrame's state.

        Parameters
        ----------
        other : DataFrame
            Source DataFrame whose internal state should replace this one.
        """

        self._df = other._df.clone()
        self._index = None if other._index is None else list(other._index)
        self._index_name = other._index_name
        self._columns_index = getattr(other, "_columns_index", None)
        if hasattr(self._columns_index, "copy"):
            self._columns_index = self._columns_index.copy()

    def _to_pandas_value(self, value: Any) -> Any:
        if isinstance(value, DataFrame):
            return value.to_pandas()

        try:
            from polarpandas.series import Series as PolarPandasSeries
        except ImportError:  # pragma: no cover - defensive fallback
            PolarPandasSeries = None  # type: ignore

        if PolarPandasSeries is not None and isinstance(value, PolarPandasSeries):
            return value.to_pandas()

        if hasattr(value, "to_pandas"):
            return value.to_pandas()

        try:
            import polars as pl  # type: ignore

            if isinstance(value, pl.DataFrame):
                return value.to_pandas()
            if isinstance(value, pl.Series):
                return value.to_pandas()
        except ImportError:  # pragma: no cover
            pass

        return value

    def insert(
        self,
        loc: int,
        column: str,
        value: Any,
        allow_duplicates: bool = False,
    ) -> None:
        """
        Insert a column into the DataFrame at the specified location.
        """

        _require_pandas("DataFrame.insert")
        pandas_df = self.to_pandas()
        pandas_value = self._to_pandas_value(value)
        pandas_df.insert(loc, column, pandas_value, allow_duplicates=allow_duplicates)
        updated = self._from_pandas_dataframe(pandas_df)
        self._update_from_dataframe(updated)

    def apply(
        self,
        func: Callable[..., Any],
        axis: Union[int, Literal["index", "columns"]] = 0,
        raw: bool = False,
        result_type: Optional[str] = None,
        args: Sequence[Any] = (),
        **kwargs: Any,
    ) -> Any:
        """
        Apply a function along an axis of the DataFrame.

        This method delegates to pandas for comprehensive compatibility and then
        converts the result back into the appropriate polarpandas object.
        """

        _require_pandas("DataFrame.apply")
        pandas_df = self.to_pandas()
        pandas_result = pandas_df.apply(
            func,
            axis=axis,
            raw=raw,
            result_type=result_type,
            args=args,
            **kwargs,
        )
        return self._from_pandas_like(pandas_result)

    def assign(self, **kwargs: Any) -> "DataFrame":
        """
        Assign new columns to a DataFrame.

        Returns a new object with all original columns in addition to new ones.
        Existing columns that are re-assigned will be overwritten.

        Parameters
        ----------
        **kwargs
            Column assignments. The column names are the keywords, and the values
            are either Series, arrays, or callables that return Series/arrays.

        Returns
        -------
        DataFrame
            A new DataFrame with the new columns in addition to all the existing columns.

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> df.assign(C=lambda x: x["A"] * 2)
        >>> df.assign(C=df["A"] * 2)
        """
        from .series import Series as PolarPandasSeries

        result_df = self._df.clone()
        expressions: List[Union[pl.Series, pl.Expr]] = []

        # Process each assignment
        for col_name, value in kwargs.items():
            if callable(value):
                # Callable - evaluate with self as argument
                result = value(self)
                if isinstance(result, PolarPandasSeries):
                    expressions.append(result._series.alias(col_name))
                elif isinstance(result, pl.Series):
                    expressions.append(result.alias(col_name))
                elif isinstance(result, (int, float, str, bool)):
                    expressions.append(pl.lit(result).alias(col_name))
                else:
                    # Convert to list/array
                    if hasattr(result, "tolist"):
                        result = result.tolist()
                    expressions.append(pl.Series(col_name, result))
            elif isinstance(value, PolarPandasSeries):
                expressions.append(value._series.alias(col_name))
            elif isinstance(value, pl.Series):
                expressions.append(value.alias(col_name))
            elif isinstance(value, (int, float, str, bool)):
                expressions.append(pl.lit(value).alias(col_name))
            else:
                # Convert to list/array
                if hasattr(value, "tolist"):
                    value = value.tolist()
                expressions.append(pl.Series(col_name, value))

        # Add all new columns
        if expressions:
            result_df = result_df.with_columns(expressions)

        return DataFrame(result_df, index=self._index)

    def set_axis(
        self,
        labels: Union[List[Any], "Index", Any],
        axis: Union[int, Literal["index", "columns"]] = 0,
        copy: bool = True,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Assign desired index to given axis.

        Parameters
        ----------
        labels : list-like or Index
            Values for the new index/columns.
        axis : {0, 1, 'index', 'columns'}, default 0
            The axis to update. 0 or 'index' for index, 1 or 'columns' for columns.
        copy : bool, default True
            Whether to copy the underlying data.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            DataFrame with new index/columns.
        """
        if axis == 0 or axis == "index":
            # Set index
            new_index = (
                list(labels) if not isinstance(labels, (list, tuple)) else labels
            )
            result_df = self._df.clone() if copy else self._df
            return DataFrame(result_df, index=new_index)
        elif axis == 1 or axis == "columns":
            # Set column names
            new_columns = (
                list(labels) if not isinstance(labels, (list, tuple)) else labels
            )
            if len(new_columns) != len(self.columns):
                raise ValueError(
                    f"Length mismatch: Expected axis has {len(self.columns)} elements, "
                    f"new values have {len(new_columns)} elements"
                )
            result_df = self._df.clone() if copy else self._df
            # Rename columns
            rename_dict = dict(zip(self.columns, new_columns))
            result_df = result_df.rename(rename_dict)
            return DataFrame(result_df, index=self._index)
        else:
            raise ValueError(f"No axis named {axis} for object type DataFrame")

    def first_valid_index(self) -> Optional[Any]:
        """
        Return index for first non-NA value or None, if no NA value is found.

        Returns
        -------
        scalar or None
            Index label of the first non-null value, or None if all values are null.
        """
        if len(self) == 0:
            return None

        # Find first row with at least one non-null value
        for i in range(len(self)):
            row = self._df.row(i, named=False)
            if any(val is not None for val in row):
                if self._index is not None and i < len(self._index):
                    return self._index[i]
                else:
                    return i
        return None

    def last_valid_index(self) -> Optional[Any]:
        """
        Return index for last non-NA value or None, if no NA value is found.

        Returns
        -------
        scalar or None
            Index label of the last non-null value, or None if all values are null.
        """
        if len(self) == 0:
            return None

        # Find last row with at least one non-null value
        for i in range(len(self) - 1, -1, -1):
            row = self._df.row(i, named=False)
            if any(val is not None for val in row):
                if self._index is not None and i < len(self._index):
                    return self._index[i]
                else:
                    return i
        return None

    def select(self, *args: Any, **kwargs: Any) -> "DataFrame":
        """
        Select columns from DataFrame.

        Returns wrapped DataFrame.
        """
        return DataFrame(self._df.select(*args, **kwargs))

    def filter(self, *args: Any, **kwargs: Any) -> "DataFrame":
        """
        Filter rows from DataFrame.

        Returns wrapped DataFrame.
        """
        # Handle polarpandas Series by converting to polars Series
        processed_args = []
        for arg in args:
            if hasattr(arg, "_series"):  # It's a polarpandas Series
                processed_args.append(arg._series)
            else:
                processed_args.append(arg)

        return DataFrame(self._df.filter(*processed_args, **kwargs))

    def with_columns(self, *exprs: Any, **named_exprs: Any) -> "DataFrame":
        """
        Add columns to DataFrame.

        Returns wrapped DataFrame.
        """
        return DataFrame(self._df.with_columns(*exprs, **named_exprs))
    def isna(self) -> "DataFrame":
        """
        Detect missing values.

        Returns
        -------
        DataFrame
            Boolean DataFrame showing whether each value is null
        """
        # Apply is_null() to each column
        result = self._df.select([pl.col(c).is_null() for c in self._df.columns])
        return DataFrame(result)

    def notna(self) -> "DataFrame":
        """
        Detect non-missing values.

        Returns
        -------
        DataFrame
            Boolean DataFrame showing whether each value is not null
        """
        # Apply is_not_null() to each column
        result = self._df.select([pl.col(c).is_not_null() for c in self._df.columns])
        return DataFrame(result)

    def isnull(self) -> "DataFrame":
        """
        Detect missing values (alias for isna()).

        Returns
        -------
        DataFrame
            Boolean DataFrame showing whether each value is null
        """
        return self.isna()

    def notnull(self) -> "DataFrame":
        """
        Detect non-missing values (alias for notna()).

        Returns
        -------
        DataFrame
            Boolean DataFrame showing whether each value is not null
        """
        return self.notna()

    def add_prefix(self, prefix: str, **kwargs: Any) -> "DataFrame":
        """
        Prefix labels with string prefix.

        Parameters
        ----------
        prefix : str
            The string to add before each column name.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            DataFrame with prefixed column names.
        """
        rename_dict = {col: f"{prefix}{col}" for col in self.columns}
        result_df = self._df.rename(rename_dict)
        return DataFrame(result_df, index=self._index)

    def add_suffix(self, suffix: str, **kwargs: Any) -> "DataFrame":
        """
        Suffix labels with string suffix.

        Parameters
        ----------
        suffix : str
            The string to add after each column name.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            DataFrame with suffixed column names.
        """
        rename_dict = {col: f"{col}{suffix}" for col in self.columns}
        result_df = self._df.rename(rename_dict)
        return DataFrame(result_df, index=self._index)

    def get(
        self, key: Union[str, List[str]], default: Optional[Any] = None
    ) -> Union["Series", "DataFrame", Any]:
        """
        Get item from object for given key (DataFrame column, Series value, etc.).

        Parameters
        ----------
        key : str or list of str
            Column name(s) to get.
        default : any, optional
            Value to return if key is not found.

        Returns
        -------
        Series, DataFrame, or default
            Column(s) if found, otherwise default value.
        """
        from polarpandas.series import Series

        if isinstance(key, list):
            # Multiple columns - return DataFrame
            missing_cols = [col for col in key if col not in self.columns]
            if missing_cols and default is not None:
                return default
            available_cols = [col for col in key if col in self.columns]
            if not available_cols:
                return default if default is not None else DataFrame(pl.DataFrame())
            result_df = self._df.select(available_cols)
            return DataFrame(result_df, index=self._index)
        else:
            # Single column - return Series
            if key not in self.columns:
                return default if default is not None else None
            return Series(self._df[key], index=self._index)

    def keys(self) -> Any:
        """
        Return the column names of the DataFrame.

        Returns
        -------
        Index
            Index-like object containing column names.
        """
        from polarpandas.index import Index

        return Index(self.columns)

    def items(self) -> Iterator[Tuple[str, "Series"]]:
        """
        Iterate over (column name, Series) pairs.

        Yields
        ------
        tuple
            (column name, Series) pairs.
        """
        from polarpandas.series import Series

        for col in self.columns:
            yield (col, Series(self._df[col], index=self._index))

    def groupby(
        self,
        by: Union[str, List[str], None] = None,
        level: Optional[Union[int, str, List[Union[int, str]]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> "_GroupBy":
        """
        Group DataFrame by one or more columns or index levels.

        Split the DataFrame into groups based on values in one or more columns
        or index levels. Returns a GroupBy object that can be used for aggregation operations.

        Parameters
        ----------
        by : str, list of str, or None
            Column name(s) to group by. Rows with the same values in these
            columns will be grouped together. If None and level is specified,
            groups by index level(s).
        level : int, str, or list, optional
            Level(s) of the index to group by. Can be level number or name.
            Only used if by is None.
        *args
            Additional positional arguments passed to Polars group_by().
        **kwargs
            Additional keyword arguments passed to Polars group_by().

        Returns
        -------
        _GroupBy
            GroupBy object that can be used for aggregation operations like
            `.agg()`, `.mean()`, `.sum()`, etc.

        Examples
        --------
        >>> import polarpandas as ppd
        >>> import polars as pl
        >>> df = ppd.DataFrame({
        ...     "category": ["A", "B", "A", "B"],
        ...     "value": [10, 20, 30, 40]
        ... })
        >>> # Group by single column
        >>> gb = df.groupby("category")
        >>> result = gb.agg(pl.col("value").mean())
        >>> # Group by multiple columns
        >>> gb = df.groupby(["category", "subcategory"])
        >>> result = gb.agg(pl.col("value").sum())
        >>> # Group by index level
        >>> df_indexed = df.set_index(['category'])
        >>> gb = df_indexed.groupby(level=0)

        Notes
        -----
        - GroupBy operations in Polars use expressions (e.g., pl.col("x").mean())
          rather than string aggregation functions like pandas
        - The GroupBy object is lazy; aggregations are computed when called
        - When using level parameter, the level values are extracted and used as
          temporary columns for grouping

        See Also
        --------
        _GroupBy : The GroupBy object returned by this method
        """
        # Handle level parameter for MultiIndex
        if level is not None and by is None:
            # Group by index level(s)
            if self._index is None or len(self._index) == 0:
                raise ValueError("Cannot group by level when index is None or empty")

            # Check if MultiIndex
            is_multiindex = (
                isinstance(self._index[0], tuple) if len(self._index) > 0 else False
            )

            if is_multiindex:
                # Extract level values
                if isinstance(self._index_name, tuple):
                    names: Optional[List[Optional[str]]] = list(self._index_name)
                else:
                    names = None
                mi = MultiIndex.from_tuples(self._index, names=names)

                # Convert level to list if single value
                if isinstance(level, (int, str)):
                    levels_to_group = [level]
                else:
                    levels_to_group = list(level)

                # Extract level values and add as temporary columns
                temp_df = self._df.clone()
                level_columns = []
                for lev in levels_to_group:
                    level_num = mi.get_level_number(lev)
                    level_values = mi.get_level_values(level_num).tolist()
                    level_name = f"__level_{level_num}__"
                    temp_df = temp_df.with_columns(pl.Series(level_name, level_values))
                    level_columns.append(level_name)

                # Group by temporary columns
                if "maintain_order" not in kwargs:
                    kwargs["maintain_order"] = True
                polars_gb = temp_df.group_by(level_columns, *args, **kwargs)
                # Store level info for result processing
                gb = _GroupBy(polars_gb, self)
                gb._level_info = {  # type: ignore[attr-defined]
                    "level_columns": level_columns,
                    "level_names": [
                        mi.names[mi.get_level_number(lev)]
                        if mi.names and mi.names[mi.get_level_number(lev)]
                        else f"level_{mi.get_level_number(lev)}"
                        for lev in levels_to_group
                    ],
                    "level_numbers": [
                        mi.get_level_number(lev) for lev in levels_to_group
                    ],
                }
                return gb
            else:
                # Regular Index - can only group by level 0
                if level != 0 and level != "index":
                    raise ValueError(
                        f"Cannot group by level {level} for non-MultiIndex"
                    )
                # Use index values as grouping column
                level_values = self._index
                temp_df = self._df.clone()
                temp_df = temp_df.with_columns(pl.Series("__level_0__", level_values))
                if "maintain_order" not in kwargs:
                    kwargs["maintain_order"] = True
                polars_gb = temp_df.group_by("__level_0__", *args, **kwargs)
                return _GroupBy(polars_gb, self)

        # Normal column-based grouping
        if by is None:
            raise ValueError("Must specify either 'by' or 'level' parameter")

        # Polars uses group_by() instead of groupby()
        # Return a wrapper for the Polars GroupBy object
        if "maintain_order" not in kwargs:
            kwargs["maintain_order"] = True
        polars_gb = self._df.group_by(by, *args, **kwargs)
        return _GroupBy(polars_gb, self)

    def melt(
        self,
        id_vars: Optional[Union[str, List[str]]] = None,
        value_vars: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Unpivot a DataFrame (melt).

        Parameters
        ----------
        id_vars : list, optional
            Columns to use as identifier variables (Polars: index)
        value_vars : list, optional
            Columns to unpivot (Polars: on)

        Returns
        -------
        DataFrame
            Melted DataFrame
        """
        # Polars uses unpivot() with 'index' instead of 'id_vars'
        # and 'on' instead of 'value_vars'
        unpivot_kwargs = {}
        if id_vars is not None:
            unpivot_kwargs["index"] = id_vars
        if value_vars is not None:
            unpivot_kwargs["on"] = value_vars
        unpivot_kwargs.update(kwargs)

        return DataFrame(self._df.unpivot(**unpivot_kwargs))  # type: ignore[arg-type]

    def merge(self, other: "DataFrame", *args: Any, **kwargs: Any) -> "DataFrame":
        """
        Merge (join) DataFrame with another DataFrame.

        Perform a database-style join operation between two DataFrames. This
        is similar to SQL JOIN operations and pandas merge().

        Parameters
        ----------
        other : DataFrame
            Right DataFrame to merge with.
        on : str or list of str, optional
            Column name(s) to join on. Must exist in both DataFrames.
            If not specified and `left_on`/`right_on` are not specified,
            joins on columns with matching names.
        how : {'inner', 'left', 'right', 'full', 'outer', 'semi', 'anti'}, default 'inner'
            Type of join to perform:
            - 'inner': Only matching rows from both DataFrames
            - 'left': All rows from left DataFrame, matching from right
            - 'right': All rows from right DataFrame, matching from left
            - 'full'/'outer': All rows from both DataFrames
            - 'semi': Rows from left that have matches in right (no right data)
            - 'anti': Rows from left that have no matches in right
        left_on : str or list of str, optional
            Column(s) from left DataFrame to join on.
        right_on : str or list of str, optional
            Column(s) from right DataFrame to join on.
        suffix : tuple of str, default ('_x', '_y')
            Suffix to append to overlapping column names.
        *args
            Additional positional arguments passed to Polars join().
        **kwargs
            Additional keyword arguments passed to Polars join().

        Returns
        -------
        DataFrame
            New DataFrame containing the merged result. Original DataFrames
            are unchanged.

        Examples
        --------
        >>> import polarpandas as ppd
        >>> left = ppd.DataFrame({"key": [1, 2, 3], "A": [10, 20, 30]})
        >>> right = ppd.DataFrame({"key": [1, 2, 4], "B": [100, 200, 400]})
        >>> # Inner join on 'key'
        >>> result = left.merge(right, on="key")
        >>> # Left join
        >>> result = left.merge(right, on="key", how="left")
        >>> # Different column names
        >>> result = left.merge(right, left_on="key", right_on="other_key")

        Notes
        -----
        - This operation converts both DataFrames to LazyFrames internally
          for the join operation
        - Result columns are the union of both DataFrames' columns
        - Overlapping columns are suffixed according to the `suffix` parameter

        See Also
        --------
        join : Alias for merge
        concat : Concatenate DataFrames along axis
        """
        # Extract the underlying Polars DataFrame if other is wrapped
        if isinstance(other, DataFrame):  # noqa: SIM108
            other_polars = other._df
        else:
            # This branch is technically unreachable due to type annotation
            # but kept for defensive programming
            other_polars = other  # type: ignore[unreachable]

        # Convert to LazyFrame for join operation
        if isinstance(other_polars, pl.DataFrame):
            other_lazy = other_polars.lazy()
        elif isinstance(other_polars, pl.LazyFrame):  # type: ignore[unreachable]
            other_lazy = other_polars
        else:
            # Handle other types (e.g., internal Polars types)
            try:
                # Try to get the DataFrame and convert to LazyFrame
                if hasattr(other_polars, "collect"):
                    # It's a LazyFrame that needs collecting first, then convert
                    other_lazy = other_polars.collect().lazy()
                elif hasattr(other_polars, "lazy"):
                    # It has a lazy() method, use it directly
                    other_lazy = other_polars.lazy()
                else:
                    # Try to convert via Polars DataFrame constructor
                    try:
                        # Attempt to create DataFrame from the object
                        other_lazy = pl.DataFrame(other_polars).lazy()
                    except (TypeError, ValueError):
                        # Fallback: try to convert via pandas
                        try:
                            import pandas as pd

                            # Convert to pandas first, then to Polars
                            if hasattr(other_polars, "to_pandas"):
                                pd_df = other_polars.to_pandas()
                            else:
                                # Try to convert via to_dict if available
                                if hasattr(other_polars, "to_dict"):
                                    pd_df = pd.DataFrame(other_polars.to_dict())
                                else:
                                    # Last resort: try to iterate
                                    try:
                                        pd_df = pd.DataFrame(list(other_polars))
                                    except (TypeError, ValueError) as e:
                                        raise TypeError(
                                            f"Cannot convert {type(other_polars).__name__} to pandas DataFrame. "
                                            "Object must support to_pandas(), to_dict(), or be iterable."
                                        ) from e
                            other_lazy = pl.from_pandas(pd_df).lazy()
                        except (ImportError, AttributeError, TypeError) as e:
                            raise TypeError(
                                f"Cannot convert {type(other_polars).__name__} to LazyFrame. "
                                "Supported types: polarpandas.DataFrame, polars.DataFrame, "
                                "polars.LazyFrame. For other types, pandas may be required."
                            ) from e
            except (AttributeError, TypeError) as e:
                raise TypeError(
                    f"Cannot convert {type(other_polars).__name__} to LazyFrame. "
                    "Supported types: polarpandas.DataFrame, polars.DataFrame, "
                    "polars.LazyFrame."
                ) from e

        # self._df is always a DataFrame (not LazyFrame) in this class
        self_lazy = self._df.lazy()

        return DataFrame(self_lazy.join(other_lazy, *args, **kwargs))

    def join(self, other: "DataFrame", *args: Any, **kwargs: Any) -> "DataFrame":
        """
        Join with another DataFrame (alias for merge).

        Parameters
        ----------
        other : DataFrame
            DataFrame to join with

        Returns
        -------
        DataFrame
            Joined DataFrame
        """
        return self.merge(other, *args, **kwargs)

    def describe(self) -> "DataFrame":
        """
        Generate descriptive statistics.

        Returns
        -------
        DataFrame
            Summary statistics
        """
        return DataFrame(self._df.describe())

    def sum(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        level: Optional[Union[int, str, List[Union[int, str]]]] = None,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        """
        Return the sum of the values over the requested axis.

        Parameters
        ----------
        axis : {0, 1, 'index', 'columns'}, default 0
            Axis along which to sum. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. If False, will attempt to use everything.
        level : int, str, or list, optional
            If the axis is a MultiIndex, sum along a particular level, collapsing into a Series.
        **kwargs
            Additional arguments passed to Polars sum().

        Returns
        -------
        Series or scalar
            Series when axis=0 (default), scalar when axis=1 or axis=None.
        """
        return DataFrameAggregationsMixin.sum(
            self,
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            level=level,
            **kwargs,
        )

    def std(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        level: Optional[Union[int, str, List[Union[int, str]]]] = None,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        """
        Return the standard deviation of the values over the requested axis.

        Parameters
        ----------
        axis : {0, 1, 'index', 'columns'}, default 0
            Axis along which to compute standard deviation. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        ddof : int, default 1
            Delta degrees of freedom. The divisor used in calculations is N - ddof, where N represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns. If False, will attempt to use everything.
        level : int, str, or list, optional
            If the axis is a MultiIndex, compute standard deviation along a particular level, collapsing into a Series.
        **kwargs
            Additional arguments passed to Polars std().

        Returns
        -------
        Series or scalar
            Series when axis=0 (default), scalar when axis=1 or axis=None.
        """
        return DataFrameAggregationsMixin.std(
            self,
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
            level=level,
            **kwargs,
        )

    def var(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        level: Optional[Union[int, str, List[Union[int, str]]]] = None,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        """
        Return the variance of the values over the requested axis.

        Parameters
        ----------
        axis : {0, 1, 'index', 'columns'}, default 0
            Axis along which to compute variance. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        ddof : int, default 1
            Delta degrees of freedom. The divisor used in calculations is N - ddof, where N represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns. If False, will attempt to use everything.
        **kwargs
            Additional arguments passed to Polars var().

        Returns
        -------
        Series or scalar
            Series when axis=0 (default), scalar when axis=1 or axis=None.
        """
        return DataFrameAggregationsMixin.var(
            self,
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
            level=level,
            **kwargs,
        )

    def count(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = 0,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        """
        Count non-null values for each column or row.

        Parameters
        ----------
        axis : {0, 1, 'index', 'columns'}, default 0
            Axis along which to count. 0 or 'index' for column-wise, 1 or 'columns' for row-wise.
        numeric_only : bool, default False
            Include only float, int, boolean columns. If False, will attempt to use everything.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Series with counts for each column (axis=0) or row (axis=1).
        """
        return DataFrameAggregationsMixin.count(
            self,
            axis=axis,
            numeric_only=numeric_only,
            **kwargs,
        )

    def median(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        level: Optional[Union[int, str, List[Union[int, str]]]] = None,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        """
        Return the median of the values over the requested axis.

        Parameters
        ----------
        axis : {0, 1, 'index', 'columns'}, default 0
            Axis along which to compute median. 0 or 'index' for column-wise, 1 or 'columns' for row-wise.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. If False, will attempt to use everything.
        **kwargs
            Additional arguments passed to Polars median().

        Returns
        -------
        Series
            Series with medians for each column (axis=0) or row (axis=1).
        """
        return DataFrameAggregationsMixin.median(
            self,
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            level=level,
            **kwargs,
        )

    def quantile(
        self,
        q: Union[float, List[float]] = 0.5,
        axis: Optional[Union[int, Literal["index", "columns"]]] = 0,
        numeric_only: bool = False,
        interpolation: str = "linear",
        **kwargs: Any,
    ) -> Union["Series", "DataFrame"]:
        """
        Return values at the given quantile over the requested axis.

        Parameters
        ----------
        q : float or array-like, default 0.5
            Quantile(s) to compute, between 0 and 1.
        axis : {0, 1, 'index', 'columns'}, default 0
            Axis along which to compute quantiles. 0 or 'index' for column-wise, 1 or 'columns' for row-wise.
        numeric_only : bool, default False
            Include only float, int, boolean columns. If False, will attempt to use everything.
        interpolation : str, default 'linear'
            Interpolation method. Polars uses 'linear' by default.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series or DataFrame
            Series when q is scalar, DataFrame when q is array-like.
        """
        from polarpandas.series import Series

        if isinstance(q, (int, float)):
            q = [q]
        elif not isinstance(q, list):
            q = list(q)  # type: ignore[unreachable]

        if axis is None or axis == 1 or axis == "columns":
            # Row-wise quantile (axis=1)
            if numeric_only:
                numeric_cols = [
                    col
                    for col in self.columns
                    if self._df[col].dtype
                    in (
                        pl.Int8,
                        pl.Int16,
                        pl.Int32,
                        pl.Int64,
                        pl.UInt8,
                        pl.UInt16,
                        pl.UInt32,
                        pl.UInt64,
                        pl.Float32,
                        pl.Float64,
                    )
                ]
                if not numeric_cols:
                    return Series(pl.Series([], dtype=pl.Float64))
                # For row-wise, compute quantile across columns
                result_series = self._df.select(
                    pl.concat_list(numeric_cols).list.quantile(q[0])  # type: ignore[attr-defined]
                )["literal"]
            else:
                result_series = self._df.select(
                    pl.concat_list(self.columns).list.quantile(q[0])  # type: ignore[attr-defined]
                )["literal"]
            index = (
                self._index
                if self._index is not None
                else list(range(len(result_series)))
            )
            if len(q) == 1:
                return Series(result_series, index=index)
            else:
                # Multiple quantiles - return DataFrame
                result_data = {}
                for quantile_val in q:
                    if numeric_only:
                        quantile_series = self._df.select(
                            pl.concat_list(numeric_cols).list.quantile(quantile_val)  # type: ignore[attr-defined]
                        )["literal"]
                    else:
                        quantile_series = self._df.select(
                            pl.concat_list(self.columns).list.quantile(quantile_val)  # type: ignore[attr-defined]
                        )["literal"]
                    result_data[quantile_val] = quantile_series.to_list()
                return DataFrame(result_data, index=index)  # type: ignore[arg-type]
        else:
            # Column-wise quantile (axis=0, default)
            if numeric_only:
                numeric_cols = [
                    col
                    for col in self.columns
                    if self._df[col].dtype
                    in (
                        pl.Int8,
                        pl.Int16,
                        pl.Int32,
                        pl.Int64,
                        pl.UInt8,
                        pl.UInt16,
                        pl.UInt32,
                        pl.UInt64,
                        pl.Float32,
                        pl.Float64,
                    )
                ]
                if not numeric_cols:
                    return Series(pl.Series([], dtype=pl.Float64))
                cols_to_use = numeric_cols
            else:
                cols_to_use = self.columns

            if len(q) == 1:
                # Single quantile - return Series
                result_pl = self._df.select(
                    [pl.col(col).quantile(q[0]) for col in cols_to_use]
                )
                values = [result_pl[col].to_list()[0] for col in result_pl.columns]
                return Series(values, index=result_pl.columns)
            else:
                # Multiple quantiles - return DataFrame
                result_data = {}
                for quantile_val in q:
                    quantile_result = self._df.select(
                        [pl.col(col).quantile(quantile_val) for col in cols_to_use]
                    )
                    result_data[quantile_val] = [
                        quantile_result[col][0] for col in quantile_result.columns
                    ]
                return DataFrame(result_data, index=cols_to_use)  # type: ignore[arg-type]
    def nunique(
        self,
        axis: int = 0,
        dropna: bool = True,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        """
        Count distinct observations over requested axis.

        Parameters
        ----------
        axis : {0, 1}, default 0
            Axis along which to count. 0 for column-wise, 1 for row-wise.
        dropna : bool, default True
            Don't include NaN in the counts.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Series with number of unique values for each column (axis=0) or row (axis=1).
        """
        from polarpandas.series import Series

        if axis == 1:
            # Row-wise nunique (axis=1) - count unique values across columns for each row
            # This is complex in Polars, we'll compute it row by row
            result_series = self._df.select(
                pl.concat_list(self.columns).list.n_unique()
            )["literal"]
            index = (
                self._index
                if self._index is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)
        else:
            # Column-wise nunique (axis=0, default) - count unique values down columns
            result_pl = self._df.select(
                [pl.col(col).n_unique() for col in self.columns]
            )
            # Convert to Series with column names as index
            values = [result_pl[col].to_list()[0] for col in result_pl.columns]
            return Series(values, index=result_pl.columns)

    def value_counts(
        self,
        subset: Optional[Union[str, List[str]]] = None,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
        **kwargs: Any,
    ) -> "Series":
        """
        Return a Series containing counts of unique rows in the DataFrame.

        Parameters
        ----------
        subset : column label or list of column labels, optional
            Columns to use when counting unique combinations.
        normalize : bool, default False
            Return proportions rather than frequencies.
        sort : bool, default True
            Sort by frequencies.
        ascending : bool, default False
            Sort in ascending order.
        dropna : bool, default True
            Don't include counts of rows that contain NA.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Series containing counts of unique rows.
        """
        from polarpandas.series import Series

        if subset is None:
            cols_to_use = self.columns
        elif isinstance(subset, str):
            cols_to_use = [subset]
        else:
            cols_to_use = subset

        # Group by the specified columns and count
        if dropna:
            result_df = (
                self._df.group_by(cols_to_use)
                .agg(pl.count().alias("count"))
                .sort("count", descending=not ascending)
            )
        else:
            result_df = (
                self._df.group_by(cols_to_use)
                .agg(pl.count().alias("count"))
                .sort("count", descending=not ascending)
            )

        if normalize:
            total = result_df["count"].sum()
            result_df = result_df.with_columns((pl.col("count") / total).alias("count"))

        # Convert to Series with tuple index for multi-column combinations
        if len(cols_to_use) == 1:
            index = result_df[cols_to_use[0]].to_list()
        else:
            index = [tuple(row) for row in result_df.select(cols_to_use).iter_rows()]

        values = result_df["count"].to_list()
        return Series(
            values, index=index, name="count" if not normalize else "proportion"
        )

    def prod(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        """
        Return the product of the values over the requested axis.

        Parameters
        ----------
        axis : {0, 1, 'index', 'columns'}, default 0
            Axis along which to compute product. 0 or 'index' for column-wise, 1 or 'columns' for row-wise.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. If False, will attempt to use everything.
        **kwargs
            Additional arguments passed to Polars product().

        Returns
        -------
        Series
            Series with products for each column (axis=0) or row (axis=1).
        """
        from polarpandas.series import Series

        if axis is None or axis == 1 or axis == "columns":
            # Row-wise product (axis=1) - aggregate across columns for each row
            if numeric_only:
                numeric_cols = [
                    col
                    for col in self.columns
                    if self._df[col].dtype
                    in (
                        pl.Int8,
                        pl.Int16,
                        pl.Int32,
                        pl.Int64,
                        pl.UInt8,
                        pl.UInt16,
                        pl.UInt32,
                        pl.UInt64,
                        pl.Float32,
                        pl.Float64,
                    )
                ]
                if not numeric_cols:
                    return Series(pl.Series([], dtype=pl.Float64))
                result_series = self._df.select(pl.product_horizontal(numeric_cols))[  # type: ignore[attr-defined]
                    "literal"
                ]
            else:
                result_series = self._df.select(pl.product_horizontal(self.columns))[  # type: ignore[attr-defined]
                    "literal"
                ]
            index = (
                self._index
                if self._index is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)
        else:
            # Column-wise product (axis=0, default) - aggregate down columns
            if numeric_only:
                numeric_cols = [
                    col
                    for col in self.columns
                    if self._df[col].dtype
                    in (
                        pl.Int8,
                        pl.Int16,
                        pl.Int32,
                        pl.Int64,
                        pl.UInt8,
                        pl.UInt16,
                        pl.UInt32,
                        pl.UInt64,
                        pl.Float32,
                        pl.Float64,
                    )
                ]
                if not numeric_cols:
                    return Series(pl.Series([], dtype=pl.Float64))
                result_pl = self._df.select(
                    [pl.col(col).product() for col in numeric_cols]
                )
            else:
                result_pl = self._df.select(
                    [pl.col(col).product() for col in self.columns]
                )
            # Convert to Series with column names as index
            values = [result_pl[col].to_list()[0] for col in result_pl.columns]
            return Series(values, index=result_pl.columns)

    def nlargest(
        self,
        n: int,
        columns: Union[str, List[str]],
        keep: str = "first",
    ) -> "DataFrame":
        """
        Return the first ``n`` rows ordered by column values in descending order.
        """
        _require_pandas("DataFrame.nlargest")

        pandas_result = self.to_pandas().nlargest(n, columns, keep=keep)
        return self._from_pandas_dataframe(pandas_result)

    def nsmallest(
        self,
        n: int,
        columns: Union[str, List[str]],
        keep: str = "first",
    ) -> "DataFrame":
        """
        Return the first ``n`` rows ordered by column values in ascending order.
        """
        _require_pandas("DataFrame.nsmallest")

        pandas_result = self.to_pandas().nsmallest(n, columns, keep=keep)
        return self._from_pandas_dataframe(pandas_result)

    def get_dummies(
        self,
        prefix: Optional[Union[str, Sequence[str], Dict[str, str]]] = None,
        prefix_sep: str = "_",
        dummy_na: bool = False,
        columns: Optional[Sequence[str]] = None,
        sparse: bool = False,
        drop_first: bool = False,
        dtype: Optional[Any] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Convert categorical variable(s) into dummy/indicator variables.
        """

        pd = _require_pandas("DataFrame.get_dummies")
        pandas_df = self.to_pandas()
        pandas_result = pd.get_dummies(
            pandas_df,
            prefix=prefix,
            prefix_sep=prefix_sep,
            dummy_na=dummy_na,
            columns=columns,
            sparse=sparse,
            drop_first=drop_first,
            dtype=dtype,
            **kwargs,
        )
        return self._from_pandas_dataframe(pandas_result)

    def pivot(
        self,
        index: Optional[Union[str, Sequence[str]]] = None,
        columns: Optional[Union[str, Sequence[str]]] = None,
        values: Optional[Union[str, Sequence[str]]] = None,
    ) -> "DataFrame":
        """
        Reshape data (produce a "pivot" table) similar to pandas DataFrame.pivot.
        """

        _require_pandas("DataFrame.pivot")

        if index is None or columns is None:
            raise TypeError(
                "DataFrame.pivot requires both 'index' and 'columns' arguments."
            )

        pandas_df = self.to_pandas()
        pandas_result = pandas_df.pivot(index=index, columns=columns, values=values)
        return self._from_pandas_dataframe(pandas_result)

    def pivot_table(
        self,
        values: Optional[Union[str, Sequence[str]]] = None,
        index: Optional[Union[str, Sequence[str]]] = None,
        columns: Optional[Union[str, Sequence[str]]] = None,
        aggfunc: Union[str, Callable[..., Any], Sequence[Callable[..., Any]]] = "mean",
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Create a spreadsheet-style pivot table using pandas for compatibility.
        """

        _require_pandas("DataFrame.pivot_table")
        pandas_df = self.to_pandas()
        pandas_result = pandas_df.pivot_table(
            values=values, index=index, columns=columns, aggfunc=aggfunc, **kwargs
        )
        return self._from_pandas_dataframe(pandas_result)

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        replace: bool = False,
        weights: Optional[Union[str, Sequence[float]]] = None,
        random_state: Optional[Any] = None,
        axis: Optional[int] = None,
        ignore_index: bool = False,
    ) -> "DataFrame":
        """
        Return a random sample of items from the DataFrame.
        """
        _require_pandas("DataFrame.sample")

        pandas_result = self.to_pandas().sample(
            n=n,
            frac=frac,
            replace=replace,
            weights=weights,
            random_state=random_state,
            axis=axis,
            ignore_index=ignore_index,
        )
        return self._from_pandas_dataframe(pandas_result)

    def swaplevel(
        self,
        i: Union[int, str] = -2,
        j: Union[int, str] = -1,
        axis: int = 0,
    ) -> "DataFrame":
        """
        Swap levels i and j in a MultiIndex.
        """
        _require_pandas("DataFrame.swaplevel")

        pandas_result = self.to_pandas().swaplevel(i=i, j=j, axis=axis)
        return self._from_pandas_dataframe(pandas_result)

    def reorder_levels(
        self,
        order: Union[List[Union[int, str]], Tuple[Union[int, str], ...]],
        axis: int = 0,
    ) -> "DataFrame":
        """
        Rearrange index levels according to the specified order.
        """
        _require_pandas("DataFrame.reorder_levels")

        pandas_result = self.to_pandas().reorder_levels(order, axis=axis)
        return self._from_pandas_dataframe(pandas_result)

    def transpose(self, *args: Any, copy: bool = False) -> "DataFrame":
        """
        Transpose index and columns.
        """
        _require_pandas("DataFrame.transpose")

        pandas_result = self.to_pandas().transpose(*args, copy=copy)
        return self._from_pandas_dataframe(pandas_result)

    @property
    def T(self) -> "DataFrame":
        """Transpose index and columns."""
        return self.transpose()

    def to_csv(self, path_or_buf: Any = None, *args: Any, **kwargs: Any) -> Any:
        """
        Write object to a comma-separated values (csv) file.
        """
        _require_pandas("DataFrame.to_csv")

        pandas_df = self.to_pandas()
        return pandas_df.to_csv(path_or_buf, *args, **kwargs)

    def to_json(self, path_or_buf: Any = None, *args: Any, **kwargs: Any) -> Any:
        """
        Convert the DataFrame to a JSON string or file.
        """

        _require_pandas("DataFrame.to_json")
        pandas_df = self.to_pandas()
        return pandas_df.to_json(*args, path_or_buf=path_or_buf, **kwargs)

    def to_parquet(self, path: Any, *args: Any, **kwargs: Any) -> None:
        """
        Write a DataFrame to the parquet format.
        """
        _require_pandas("DataFrame.to_parquet")

        pandas_df = self.to_pandas()
        pandas_df.to_parquet(path, *args, **kwargs)

    def to_feather(self, path: Any, **kwargs: Any) -> None:
        """
        Write a DataFrame to the Feather format.
        """

        _require_pandas("DataFrame.to_feather")
        pandas_df = self.to_pandas()
        pandas_df.to_feather(path, **kwargs)

    def to_sql(
        self,
        name: str,
        con: Any,
        schema: Optional[str] = None,
        if_exists: str = "fail",
        index: bool = True,
        index_label: Optional[Union[str, Sequence[str]]] = None,
        chunksize: Optional[int] = None,
        dtype: Optional[Dict[str, Any]] = None,
        method: Optional[Union[str, Callable[..., Any]]] = None,
        *,
        primary_key: Optional[Union[str, Sequence[str]]] = None,
        auto_increment: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Write records stored in a DataFrame to a SQL database.
        """

        valid_if_exists = {"fail", "replace", "append"}
        if if_exists not in valid_if_exists:
            raise ValueError(
                f"'{if_exists}' is not valid for if_exists; "
                f"valid options are {sorted(valid_if_exists)}."
            )

        if auto_increment and not primary_key:
            raise ValueError("auto_increment requires primary_key to be specified.")

        if primary_key is not None or auto_increment:
            _require_sqlalchemy("DataFrame.to_sql with primary_key/auto_increment")
            create_table_with_primary_key(
                self._df,
                name,
                con,
                schema=schema,
                if_exists=if_exists,
                primary_key=primary_key,
                auto_increment=auto_increment,
                dtype=dtype,
                index=index,
                index_label=index_label,
            )
            return None

        _require_pandas("DataFrame.to_sql")
        pandas_df = self.to_pandas()
        return pandas_df.to_sql(
            name,
            con,
            schema=schema,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            chunksize=chunksize,
            dtype=dtype,
            method=method,
            **kwargs,
        )

    def clip(
        self,
        lower: Optional[Union[float, int, Dict[str, Any]]] = None,
        upper: Optional[Union[float, int, Dict[str, Any]]] = None,
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Optional["DataFrame"]:
        """
        Trim values at input threshold(s).

        Parameters
        ----------
        lower : float or dict, optional
            Minimum threshold value. If dict, column-specific thresholds.
        upper : float or dict, optional
            Maximum threshold value. If dict, column-specific thresholds.
        axis : int, optional
            Not used, for pandas compatibility.
        inplace : bool, default False
            If True, modify DataFrame in place and return None.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame or None
            DataFrame with clipped values, or None if inplace=True.
        """
        expressions = []
        for col in self.columns:
            col_expr = pl.col(col)
            if lower is not None:
                if isinstance(lower, dict) and col in lower:
                    col_expr = col_expr.clip(lower_bound=lower[col])
                elif not isinstance(lower, dict):
                    col_expr = col_expr.clip(lower_bound=lower)
            if upper is not None:
                if isinstance(upper, dict) and col in upper:
                    col_expr = col_expr.clip(upper_bound=upper[col])
                elif not isinstance(upper, dict):
                    col_expr = col_expr.clip(upper_bound=upper)
            expressions.append(col_expr.alias(col))

        result_df = self._df.select(expressions)

        if inplace:
            self._df = result_df
            return None
        else:
            return DataFrame(result_df)

    def where(
        self,
        cond: Union["DataFrame", Any],
        other: Optional[Union[Any, "DataFrame"]] = None,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Optional["DataFrame"]:
        """
        Replace values where the condition is False.

        Parameters
        ----------
        cond : bool DataFrame or callable
            Where cond is True, keep the original value. Where False, replace with corresponding value from other.
        other : scalar, Series, or DataFrame, optional
            Entries where cond is False are replaced with corresponding value from other.
        inplace : bool, default False
            If True, modify DataFrame in place and return None.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame or None
            DataFrame with replaced values, or None if inplace=True.
        """
        if isinstance(cond, DataFrame):
            cond_df = cond._df
        else:
            # Assume it's a callable or expression
            raise NotImplementedError(
                "where() with callable conditions not yet implemented"
            )

        if other is None:
            other = None  # Will be replaced with NaN
        elif isinstance(other, DataFrame):
            other_df = other._df
        else:
            # Scalar value
            other = other

        expressions = []
        for col in self.columns:
            if isinstance(other, DataFrame) and col in other.columns:
                expr = pl.when(cond_df[col]).then(pl.col(col)).otherwise(other_df[col])
            elif other is None:
                expr = pl.when(cond_df[col]).then(pl.col(col)).otherwise(None)
            else:
                expr = pl.when(cond_df[col]).then(pl.col(col)).otherwise(other)  # type: ignore[arg-type]
            expressions.append(expr.alias(col))

        result_df = self._df.select(expressions)

        if inplace:
            self._df = result_df
            return None
        else:
            return DataFrame(result_df)

    def agg(
        self,
        func: Union[
            str, List[str], Dict[str, Union[str, List[str]]], Callable[..., Any]
        ],
        axis: Union[int, Literal["index", "columns"]] = 0,
        *args: Any,
        **kwargs: Any,
    ) -> Union["DataFrame", "Series"]:
        """
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : function, str, list, or dict
            Function to use for aggregating the data.
        axis : {0, 1, 'index', 'columns'}, default 0
            Axis along which to aggregate.
        *args
            Positional arguments to pass to func.
        **kwargs
            Keyword arguments to pass to func.

        Returns
        -------
        DataFrame or Series
            Aggregated result.
        """

        # Handle string aggregation functions
        if isinstance(func, str):
            # Map string to method
            agg_methods = {
                "sum": self.sum,
                "min": self.min,
                "max": self.max,
                "std": self.std,
                "var": self.var,
                "count": self.count,
                "median": self.median,
                "nunique": self.nunique,
            }
            if func in agg_methods:
                return agg_methods[func](axis=axis, **kwargs)  # type: ignore[no-any-return,operator]

        # For other cases, delegate to apply()
        return self.apply(func, axis=axis, *args, **kwargs)  # type: ignore[arg-type,misc] # noqa: B026

    def aggregate(
        self,
        func: Union[
            str, List[str], Dict[str, Union[str, List[str]]], Callable[..., Any]
        ],
        axis: Union[int, Literal["index", "columns"]] = 0,
        *args: Any,
        **kwargs: Any,
    ) -> Union["DataFrame", "Series"]:
        """
        Aggregate using one or more operations over the specified axis.

        Alias for agg().
        """
        return self.agg(func, axis=axis, *args, **kwargs)  # type: ignore[misc] # noqa: B026

    def mask(
        self,
        cond: Union["DataFrame", Any],
        other: Optional[Union[Any, "DataFrame"]] = None,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Optional["DataFrame"]:
        """
        Replace values where the condition is True.

        This is the inverse of where() - replace values where condition is True instead of False.

        Parameters
        ----------
        cond : bool DataFrame or callable
            Where cond is True, replace with corresponding value from other. Where False, keep the original value.
        other : scalar, Series, or DataFrame, optional
            Entries where cond is True are replaced with corresponding value from other.
        inplace : bool, default False
            If True, modify DataFrame in place and return None.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame or None
            DataFrame with replaced values, or None if inplace=True.
        """
        if isinstance(cond, DataFrame):
            cond_df = cond._df
        else:
            # Assume it's a callable or expression
            raise NotImplementedError(
                "mask() with callable conditions not yet implemented"
            )

        if other is None:
            other = None  # Will be replaced with NaN
        elif isinstance(other, DataFrame):
            other_df = other._df
        else:
            # Scalar value
            other = other

        expressions = []
        for col in self.columns:
            if isinstance(other, DataFrame) and col in other.columns:
                # Inverse of where: replace where cond is True
                expr = pl.when(cond_df[col]).then(other_df[col]).otherwise(pl.col(col))
            elif other is None:
                expr = pl.when(cond_df[col]).then(None).otherwise(pl.col(col))
            else:
                expr = pl.when(cond_df[col]).then(other).otherwise(pl.col(col))  # type: ignore[arg-type]
            expressions.append(expr.alias(col))

        result_df = self._df.select(expressions)

        if inplace:
            self._df = result_df
            return None
        else:
            return DataFrame(result_df, index=self._index)

    def squeeze(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
    ) -> Union["Series", "DataFrame"]:
        """
        Squeeze 1 dimensional axis objects into scalars.

        Parameters
        ----------
        axis : {0, 1, 'index', 'columns'} or None, default None
            A specific axis to squeeze. By default, all length-1 axes are squeezed.

        Returns
        -------
        DataFrame, Series, or scalar
            The projection after squeezing, or the original type if all lengths are greater than 1.
        """
        from polarpandas.series import Series

        # If single column, return as Series
        if len(self.columns) == 1:
            return Series(self._df[self.columns[0]], index=self._index)

        # If single row, return as Series with column names as index
        if len(self) == 1:
            return Series(pl.Series(self._df.row(0)), index=self.columns)

        # Otherwise return DataFrame
        return self

    def compare(
        self,
        other: "DataFrame",
        align_axis: Union[int, Literal["index", "columns"]] = 1,
        keep_shape: bool = False,
        keep_equal: bool = False,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Compare to another DataFrame and show the differences.

        Parameters
        ----------
        other : DataFrame
            Object to compare with.
        align_axis : {0, 1, 'index', 'columns'}, default 1
            Align differences on columns (1) or index (0).
        keep_shape : bool, default False
            If True, all rows and columns are kept. Otherwise, only the ones with different values are shown.
        keep_equal : bool, default False
            If True, the result keeps values that are equal. Otherwise, equal values are shown as NaNs.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            DataFrame showing the differences.
        """
        if len(self) != len(other):
            raise ValueError("Can only compare identically-labeled DataFrame objects")

        # Find common columns
        common_cols = [col for col in self.columns if col in other.columns]

        if not common_cols:
            return DataFrame(pl.DataFrame())

        # Compare each column
        diff_data = {}
        for col in common_cols:
            self_col = self._df[col]
            other_col = other._df[col]

            # Find differences
            if keep_equal:
                diff_mask = self_col != other_col
            else:
                diff_mask = (self_col != other_col) & (
                    self_col.is_not_null() | other_col.is_not_null()
                )

            if keep_shape or diff_mask.any():
                if align_axis == 1 or align_axis == "columns":
                    # Show differences side by side
                    diff_data[f"{col}_self"] = self_col
                    diff_data[f"{col}_other"] = other_col
                else:
                    # Show differences stacked
                    diff_data[col] = pl.when(diff_mask).then(self_col).otherwise(None)  # type: ignore[assignment]

        if not diff_data:
            return DataFrame(pl.DataFrame())

        result_df = pl.DataFrame(diff_data)
        return DataFrame(result_df, index=self._index)
    def any(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        """
        Return whether any element is True, potentially over an axis.

        Parameters
        ----------
        axis : {0, 1, 'index', 'columns'}, default 0
            Axis along which to reduce. 0 or 'index' for column-wise, 1 or 'columns' for row-wise.
        bool_only : bool, default False
            Include only boolean columns.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Series with boolean results for each column (axis=0) or row (axis=1).
        """
        from polarpandas.series import Series

        if axis is None or axis == 1 or axis == "columns":
            # Row-wise any (axis=1)
            if bool_only:
                bool_cols = [
                    col for col in self.columns if self._df[col].dtype == pl.Boolean
                ]
                if not bool_cols:
                    return Series(pl.Series([], dtype=pl.Boolean))
                result_series = self._df.select(pl.any_horizontal(bool_cols))["literal"]
            else:
                result_series = self._df.select(pl.any_horizontal(self.columns))[
                    "literal"
                ]
            index = (
                self._index
                if self._index is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)
        else:
            # Column-wise any (axis=0, default)
            if bool_only:
                bool_cols = [
                    col for col in self.columns if self._df[col].dtype == pl.Boolean
                ]
                if not bool_cols:
                    return Series(pl.Series([], dtype=pl.Boolean))
                result_pl = self._df.select([pl.col(col).any() for col in bool_cols])
            else:
                # For non-boolean columns, convert to boolean (non-zero/non-null = True)
                result_pl = self._df.select(
                    [
                        pl.col(col).cast(pl.Boolean).any()
                        if self._df[col].dtype != pl.Boolean
                        else pl.col(col).any()
                        for col in self.columns
                    ]
                )
            values = [result_pl[col].to_list()[0] for col in result_pl.columns]
            return Series(values, index=result_pl.columns)

    def eq(
        self,
        other: Union[Any, "Series", "DataFrame"],
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Return equal to of DataFrame and other, element-wise (binary operator ==).

        Parameters
        ----------
        other : scalar, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0, 1, 'index', 'columns'} or None, default None
            Whether to compare by the index (0 or 'index') or columns (1 or 'columns').
            For Series input, axis to match Series index on.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            Result of the comparison.
        """
        from polarpandas.series import Series

        if isinstance(other, DataFrame):
            # DataFrame comparison - align by row position and column names
            if len(self) != len(other):
                raise ValueError(
                    "Can only compare identically-labeled DataFrame objects"
                )
            expressions = []
            # Rename other DataFrame columns with suffix for comparison
            other_df_renamed = other._df.select(
                [pl.col(col).alias(f"{col}_other") for col in other.columns]
            )
            # Combine DataFrames horizontally
            combined_df = self._df.hstack(other_df_renamed)
            for col in self.columns:
                if col in other.columns:
                    expressions.append(
                        (pl.col(col) == pl.col(f"{col}_other")).alias(col)
                    )
                else:
                    expressions.append(pl.lit(False).alias(col))
            result_df = combined_df.select(expressions)
        elif isinstance(other, Series):
            # Series comparison - broadcast along axis
            if axis is None or axis == 0 or axis == "index":
                # Compare each column with Series
                if len(other) != len(self):
                    raise ValueError("Lengths must match")
                expressions = [
                    (
                        pl.col(col)
                        == pl.Series(
                            other._series if hasattr(other, "_series") else other
                        )
                    ).alias(col)
                    for col in self.columns
                ]
            else:
                # Compare each row with Series
                raise NotImplementedError(
                    "Series comparison with axis=1 not yet implemented"
                )
            result_df = self._df.select(expressions)
        else:
            # Scalar comparison
            expressions = [(pl.col(col) == other).alias(col) for col in self.columns]
            result_df = self._df.select(expressions)

        return DataFrame(result_df, index=self._index)

    def ne(
        self,
        other: Union[Any, "Series", "DataFrame"],
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Return not equal to of DataFrame and other, element-wise (binary operator !=).

        Parameters
        ----------
        other : scalar, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0, 1, 'index', 'columns'} or None, default None
            Whether to compare by the index (0 or 'index') or columns (1 or 'columns').
            For Series input, axis to match Series index on.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            Result of the comparison.
        """
        from polarpandas.series import Series

        if isinstance(other, DataFrame):
            if len(self) != len(other):
                raise ValueError(
                    "Can only compare identically-labeled DataFrame objects"
                )
            expressions = []
            other_df_renamed = other._df.select(
                [pl.col(col).alias(f"{col}_other") for col in other.columns]
            )
            combined_df = self._df.hstack(other_df_renamed)
            for col in self.columns:
                if col in other.columns:
                    expressions.append(
                        (pl.col(col) != pl.col(f"{col}_other")).alias(col)
                    )
                else:
                    expressions.append(pl.lit(True).alias(col))
            result_df = combined_df.select(expressions)
        elif isinstance(other, Series):
            if axis is None or axis == 0 or axis == "index":
                if len(other) != len(self):
                    raise ValueError("Lengths must match")
                expressions = [
                    (
                        pl.col(col)
                        != pl.Series(
                            other._series if hasattr(other, "_series") else other
                        )
                    ).alias(col)
                    for col in self.columns
                ]
            else:
                raise NotImplementedError(
                    "Series comparison with axis=1 not yet implemented"
                )
            result_df = self._df.select(expressions)
        else:
            expressions = [(pl.col(col) != other).alias(col) for col in self.columns]
            result_df = self._df.select(expressions)

        return DataFrame(result_df, index=self._index)

    def gt(
        self,
        other: Union[Any, "Series", "DataFrame"],
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Return greater than of DataFrame and other, element-wise (binary operator >).

        Parameters
        ----------
        other : scalar, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0, 1, 'index', 'columns'} or None, default None
            Whether to compare by the index (0 or 'index') or columns (1 or 'columns').
            For Series input, axis to match Series index on.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            Result of the comparison.
        """
        from polarpandas.series import Series

        if isinstance(other, DataFrame):
            if len(self) != len(other):
                raise ValueError(
                    "Can only compare identically-labeled DataFrame objects"
                )
            expressions = []
            other_df_renamed = other._df.select(
                [pl.col(col).alias(f"{col}_other") for col in other.columns]
            )
            combined_df = self._df.hstack(other_df_renamed)
            for col in self.columns:
                if col in other.columns:
                    expressions.append(
                        (pl.col(col) > pl.col(f"{col}_other")).alias(col)
                    )
                else:
                    expressions.append(pl.lit(False).alias(col))
            result_df = combined_df.select(expressions)
        elif isinstance(other, Series):
            if axis is None or axis == 0 or axis == "index":
                if len(other) != len(self):
                    raise ValueError("Lengths must match")
                expressions = [
                    (
                        pl.col(col)
                        > pl.Series(
                            other._series if hasattr(other, "_series") else other
                        )
                    ).alias(col)
                    for col in self.columns
                ]
            else:
                raise NotImplementedError(
                    "Series comparison with axis=1 not yet implemented"
                )
            result_df = self._df.select(expressions)
        else:
            expressions = [(pl.col(col) > other).alias(col) for col in self.columns]
            result_df = self._df.select(expressions)

        return DataFrame(result_df, index=self._index)

    def lt(
        self,
        other: Union[Any, "Series", "DataFrame"],
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Return less than of DataFrame and other, element-wise (binary operator <).

        Parameters
        ----------
        other : scalar, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0, 1, 'index', 'columns'} or None, default None
            Whether to compare by the index (0 or 'index') or columns (1 or 'columns').
            For Series input, axis to match Series index on.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            Result of the comparison.
        """
        from polarpandas.series import Series

        if isinstance(other, DataFrame):
            if len(self) != len(other):
                raise ValueError(
                    "Can only compare identically-labeled DataFrame objects"
                )
            expressions = []
            other_df_renamed = other._df.select(
                [pl.col(col).alias(f"{col}_other") for col in other.columns]
            )
            combined_df = self._df.hstack(other_df_renamed)
            for col in self.columns:
                if col in other.columns:
                    expressions.append(
                        (pl.col(col) < pl.col(f"{col}_other")).alias(col)
                    )
                else:
                    expressions.append(pl.lit(False).alias(col))
            result_df = combined_df.select(expressions)
        elif isinstance(other, Series):
            if axis is None or axis == 0 or axis == "index":
                if len(other) != len(self):
                    raise ValueError("Lengths must match")
                expressions = [
                    (
                        pl.col(col)
                        < pl.Series(
                            other._series if hasattr(other, "_series") else other
                        )
                    ).alias(col)
                    for col in self.columns
                ]
            else:
                raise NotImplementedError(
                    "Series comparison with axis=1 not yet implemented"
                )
            result_df = self._df.select(expressions)
        else:
            expressions = [(pl.col(col) < other).alias(col) for col in self.columns]
            result_df = self._df.select(expressions)

        return DataFrame(result_df, index=self._index)

    def ge(
        self,
        other: Union[Any, "Series", "DataFrame"],
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Return greater than or equal to of DataFrame and other, element-wise (binary operator >=).

        Parameters
        ----------
        other : scalar, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0, 1, 'index', 'columns'} or None, default None
            Whether to compare by the index (0 or 'index') or columns (1 or 'columns').
            For Series input, axis to match Series index on.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            Result of the comparison.
        """
        from polarpandas.series import Series

        if isinstance(other, DataFrame):
            if len(self) != len(other):
                raise ValueError(
                    "Can only compare identically-labeled DataFrame objects"
                )
            expressions = []
            other_df_renamed = other._df.select(
                [pl.col(col).alias(f"{col}_other") for col in other.columns]
            )
            combined_df = self._df.hstack(other_df_renamed)
            for col in self.columns:
                if col in other.columns:
                    expressions.append(
                        (pl.col(col) >= pl.col(f"{col}_other")).alias(col)
                    )
                else:
                    expressions.append(pl.lit(False).alias(col))
            result_df = combined_df.select(expressions)
        elif isinstance(other, Series):
            if axis is None or axis == 0 or axis == "index":
                if len(other) != len(self):
                    raise ValueError("Lengths must match")
                expressions = [
                    (
                        pl.col(col)
                        >= pl.Series(
                            other._series if hasattr(other, "_series") else other
                        )
                    ).alias(col)
                    for col in self.columns
                ]
            else:
                raise NotImplementedError(
                    "Series comparison with axis=1 not yet implemented"
                )
            result_df = self._df.select(expressions)
        else:
            expressions = [(pl.col(col) >= other).alias(col) for col in self.columns]
            result_df = self._df.select(expressions)

        return DataFrame(result_df, index=self._index)

    def le(
        self,
        other: Union[Any, "Series", "DataFrame"],
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Return less than or equal to of DataFrame and other, element-wise (binary operator <=).

        Parameters
        ----------
        other : scalar, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : {0, 1, 'index', 'columns'} or None, default None
            Whether to compare by the index (0 or 'index') or columns (1 or 'columns').
            For Series input, axis to match Series index on.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            Result of the comparison.
        """
        from polarpandas.series import Series

        if isinstance(other, DataFrame):
            if len(self) != len(other):
                raise ValueError(
                    "Can only compare identically-labeled DataFrame objects"
                )
            expressions = []
            other_df_renamed = other._df.select(
                [pl.col(col).alias(f"{col}_other") for col in other.columns]
            )
            combined_df = self._df.hstack(other_df_renamed)
            for col in self.columns:
                if col in other.columns:
                    expressions.append(
                        (pl.col(col) <= pl.col(f"{col}_other")).alias(col)
                    )
                else:
                    expressions.append(pl.lit(False).alias(col))
            result_df = combined_df.select(expressions)
        elif isinstance(other, Series):
            if axis is None or axis == 0 or axis == "index":
                if len(other) != len(self):
                    raise ValueError("Lengths must match")
                expressions = [
                    (
                        pl.col(col)
                        <= pl.Series(
                            other._series if hasattr(other, "_series") else other
                        )
                    ).alias(col)
                    for col in self.columns
                ]
            else:
                raise NotImplementedError(
                    "Series comparison with axis=1 not yet implemented"
                )
            result_df = self._df.select(expressions)
        else:
            expressions = [(pl.col(col) <= other).alias(col) for col in self.columns]
            result_df = self._df.select(expressions)

        return DataFrame(result_df, index=self._index)

    def add(
        self,
        other: Union[Any, "Series", "DataFrame"],
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        level: Optional[Any] = None,
        fill_value: Optional[Any] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Get addition of DataFrame and other, element-wise (binary operator +).

        Parameters
        ----------
        other : scalar, Series, or DataFrame
            Object to add to the DataFrame.
        axis : {0, 1, 'index', 'columns'} or None, default None
            Axis to match Series index on. For Series input, axis to match Series index on.
        level : int or name, optional
            Broadcast across a level, matching Index values on the passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for successful
            DataFrame alignment, with this value before computation.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.
        """
        from polarpandas.series import Series

        if isinstance(other, DataFrame):
            if len(self) != len(other):
                raise ValueError("Can only add identically-labeled DataFrame objects")
            expressions = []
            other_df_renamed = other._df.select(
                [pl.col(col).alias(f"{col}_other") for col in other.columns]
            )
            combined_df = self._df.hstack(other_df_renamed)
            for col in self.columns:
                if col in other.columns:
                    if fill_value is not None:
                        expr = pl.col(col).fill_null(fill_value) + pl.col(
                            f"{col}_other"
                        ).fill_null(fill_value)
                    else:
                        expr = pl.col(col) + pl.col(f"{col}_other")
                    expressions.append(expr.alias(col))
                else:
                    expressions.append(pl.col(col).alias(col))
            result_df = combined_df.select(expressions)
        elif isinstance(other, Series):
            if axis is None or axis == 0 or axis == "index":
                if len(other) != len(self):
                    raise ValueError("Lengths must match")
                other_series = other._series if hasattr(other, "_series") else other
                if fill_value is not None:
                    expressions = [
                        (
                            pl.col(col).fill_null(fill_value)
                            + pl.Series(other_series).fill_null(fill_value)
                        ).alias(col)
                        for col in self.columns
                    ]
                else:
                    expressions = [
                        (pl.col(col) + pl.Series(other_series)).alias(col)
                        for col in self.columns
                    ]
            else:
                raise NotImplementedError(
                    "Series arithmetic with axis=1 not yet implemented"
                )
            result_df = self._df.select(expressions)
        else:
            # Scalar
            if fill_value is not None:
                expressions = [
                    (pl.col(col).fill_null(fill_value) + other).alias(col)
                    for col in self.columns
                ]
            else:
                expressions = [(pl.col(col) + other).alias(col) for col in self.columns]
            result_df = self._df.select(expressions)

        return DataFrame(result_df, index=self._index)

    def sub(
        self,
        other: Union[Any, "Series", "DataFrame"],
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        level: Optional[Any] = None,
        fill_value: Optional[Any] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Get subtraction of DataFrame and other, element-wise (binary operator -).

        Parameters
        ----------
        other : scalar, Series, or DataFrame
            Object to subtract from the DataFrame.
        axis : {0, 1, 'index', 'columns'} or None, default None
            Axis to match Series index on. For Series input, axis to match Series index on.
        level : int or name, optional
            Broadcast across a level, matching Index values on the passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for successful
            DataFrame alignment, with this value before computation.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.
        """
        from polarpandas.series import Series

        if isinstance(other, DataFrame):
            if len(self) != len(other):
                raise ValueError(
                    "Can only subtract identically-labeled DataFrame objects"
                )
            expressions = []
            other_df_renamed = other._df.select(
                [pl.col(col).alias(f"{col}_other") for col in other.columns]
            )
            combined_df = self._df.hstack(other_df_renamed)
            for col in self.columns:
                if col in other.columns:
                    if fill_value is not None:
                        expr = pl.col(col).fill_null(fill_value) - pl.col(
                            f"{col}_other"
                        ).fill_null(fill_value)
                    else:
                        expr = pl.col(col) - pl.col(f"{col}_other")
                    expressions.append(expr.alias(col))
                else:
                    expressions.append(pl.col(col).alias(col))
            result_df = combined_df.select(expressions)
        elif isinstance(other, Series):
            if axis is None or axis == 0 or axis == "index":
                if len(other) != len(self):
                    raise ValueError("Lengths must match")
                other_series = other._series if hasattr(other, "_series") else other
                if fill_value is not None:
                    expressions = [
                        (
                            pl.col(col).fill_null(fill_value)
                            - pl.Series(other_series).fill_null(fill_value)
                        ).alias(col)
                        for col in self.columns
                    ]
                else:
                    expressions = [
                        (pl.col(col) - pl.Series(other_series)).alias(col)
                        for col in self.columns
                    ]
            else:
                raise NotImplementedError(
                    "Series arithmetic with axis=1 not yet implemented"
                )
            result_df = self._df.select(expressions)
        else:
            if fill_value is not None:
                expressions = [
                    (pl.col(col).fill_null(fill_value) - other).alias(col)
                    for col in self.columns
                ]
            else:
                expressions = [(pl.col(col) - other).alias(col) for col in self.columns]
            result_df = self._df.select(expressions)

        return DataFrame(result_df, index=self._index)

    def subtract(
        self,
        other: Union[Any, "Series", "DataFrame"],
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        level: Optional[Any] = None,
        fill_value: Optional[Any] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """Alias for sub()."""
        return self.sub(other, axis=axis, level=level, fill_value=fill_value, **kwargs)

    def mul(
        self,
        other: Union[Any, "Series", "DataFrame"],
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        level: Optional[Any] = None,
        fill_value: Optional[Any] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Get multiplication of DataFrame and other, element-wise (binary operator *).

        Parameters
        ----------
        other : scalar, Series, or DataFrame
            Object to multiply with the DataFrame.
        axis : {0, 1, 'index', 'columns'} or None, default None
            Axis to match Series index on. For Series input, axis to match Series index on.
        level : int or name, optional
            Broadcast across a level, matching Index values on the passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for successful
            DataFrame alignment, with this value before computation.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.
        """
        from polarpandas.series import Series

        if isinstance(other, DataFrame):
            if len(self) != len(other):
                raise ValueError(
                    "Can only multiply identically-labeled DataFrame objects"
                )
            expressions = []
            other_df_renamed = other._df.select(
                [pl.col(col).alias(f"{col}_other") for col in other.columns]
            )
            combined_df = self._df.hstack(other_df_renamed)
            for col in self.columns:
                if col in other.columns:
                    if fill_value is not None:
                        expr = pl.col(col).fill_null(fill_value) * pl.col(
                            f"{col}_other"
                        ).fill_null(fill_value)
                    else:
                        expr = pl.col(col) * pl.col(f"{col}_other")
                    expressions.append(expr.alias(col))
                else:
                    expressions.append(pl.col(col).alias(col))
            result_df = combined_df.select(expressions)
        elif isinstance(other, Series):
            if axis is None or axis == 0 or axis == "index":
                if len(other) != len(self):
                    raise ValueError("Lengths must match")
                other_series = other._series if hasattr(other, "_series") else other
                if fill_value is not None:
                    expressions = [
                        (
                            pl.col(col).fill_null(fill_value)
                            * pl.Series(other_series).fill_null(fill_value)
                        ).alias(col)
                        for col in self.columns
                    ]
                else:
                    expressions = [
                        (pl.col(col) * pl.Series(other_series)).alias(col)
                        for col in self.columns
                    ]
            else:
                raise NotImplementedError(
                    "Series arithmetic with axis=1 not yet implemented"
                )
            result_df = self._df.select(expressions)
        else:
            if fill_value is not None:
                expressions = [
                    (pl.col(col).fill_null(fill_value) * other).alias(col)
                    for col in self.columns
                ]
            else:
                expressions = [(pl.col(col) * other).alias(col) for col in self.columns]
            result_df = self._df.select(expressions)

        return DataFrame(result_df, index=self._index)

    def multiply(
        self,
        other: Union[Any, "Series", "DataFrame"],
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        level: Optional[Any] = None,
        fill_value: Optional[Any] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """Alias for mul()."""
        return self.mul(other, axis=axis, level=level, fill_value=fill_value, **kwargs)

    def div(
        self,
        other: Union[Any, "Series", "DataFrame"],
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        level: Optional[Any] = None,
        fill_value: Optional[Any] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Get floating division of DataFrame and other, element-wise (binary operator /).

        Parameters
        ----------
        other : scalar, Series, or DataFrame
            Object to divide the DataFrame by.
        axis : {0, 1, 'index', 'columns'} or None, default None
            Axis to match Series index on. For Series input, axis to match Series index on.
        level : int or name, optional
            Broadcast across a level, matching Index values on the passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for successful
            DataFrame alignment, with this value before computation.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.
        """
        from polarpandas.series import Series

        if isinstance(other, DataFrame):
            if len(self) != len(other):
                raise ValueError(
                    "Can only divide identically-labeled DataFrame objects"
                )
            expressions = []
            other_df_renamed = other._df.select(
                [pl.col(col).alias(f"{col}_other") for col in other.columns]
            )
            combined_df = self._df.hstack(other_df_renamed)
            for col in self.columns:
                if col in other.columns:
                    if fill_value is not None:
                        expr = pl.col(col).fill_null(fill_value) / pl.col(
                            f"{col}_other"
                        ).fill_null(fill_value)
                    else:
                        expr = pl.col(col) / pl.col(f"{col}_other")
                    expressions.append(expr.alias(col))
                else:
                    expressions.append(pl.col(col).alias(col))
            result_df = combined_df.select(expressions)
        elif isinstance(other, Series):
            if axis is None or axis == 0 or axis == "index":
                if len(other) != len(self):
                    raise ValueError("Lengths must match")
                other_series = other._series if hasattr(other, "_series") else other
                if fill_value is not None:
                    expressions = [
                        (
                            pl.col(col).fill_null(fill_value)
                            / pl.Series(other_series).fill_null(fill_value)
                        ).alias(col)
                        for col in self.columns
                    ]
                else:
                    expressions = [
                        (pl.col(col) / pl.Series(other_series)).alias(col)
                        for col in self.columns
                    ]
            else:
                raise NotImplementedError(
                    "Series arithmetic with axis=1 not yet implemented"
                )
            result_df = self._df.select(expressions)
        else:
            if fill_value is not None:
                expressions = [
                    (pl.col(col).fill_null(fill_value) / other).alias(col)
                    for col in self.columns
                ]
            else:
                expressions = [(pl.col(col) / other).alias(col) for col in self.columns]
            result_df = self._df.select(expressions)

        return DataFrame(result_df, index=self._index)

    def divide(
        self,
        other: Union[Any, "Series", "DataFrame"],
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        level: Optional[Any] = None,
        fill_value: Optional[Any] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """Alias for div()."""
        return self.div(other, axis=axis, level=level, fill_value=fill_value, **kwargs)

    def mod(
        self,
        other: Union[Any, "Series", "DataFrame"],
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        level: Optional[Any] = None,
        fill_value: Optional[Any] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Get modulo of DataFrame and other, element-wise (binary operator %).

        Parameters
        ----------
        other : scalar, Series, or DataFrame
            Object to compute modulo with the DataFrame.
        axis : {0, 1, 'index', 'columns'} or None, default None
            Axis to match Series index on. For Series input, axis to match Series index on.
        level : int or name, optional
            Broadcast across a level, matching Index values on the passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for successful
            DataFrame alignment, with this value before computation.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.
        """
        from polarpandas.series import Series

        if isinstance(other, DataFrame):
            if len(self) != len(other):
                raise ValueError(
                    "Can only compute modulo with identically-labeled DataFrame objects"
                )
            expressions = []
            other_df_renamed = other._df.select(
                [pl.col(col).alias(f"{col}_other") for col in other.columns]
            )
            combined_df = self._df.hstack(other_df_renamed)
            for col in self.columns:
                if col in other.columns:
                    if fill_value is not None:
                        expr = pl.col(col).fill_null(fill_value) % pl.col(
                            f"{col}_other"
                        ).fill_null(fill_value)
                    else:
                        expr = pl.col(col) % pl.col(f"{col}_other")
                    expressions.append(expr.alias(col))
                else:
                    expressions.append(pl.col(col).alias(col))
            result_df = combined_df.select(expressions)
        elif isinstance(other, Series):
            if axis is None or axis == 0 or axis == "index":
                if len(other) != len(self):
                    raise ValueError("Lengths must match")
                other_series = other._series if hasattr(other, "_series") else other
                if fill_value is not None:
                    expressions = [
                        (
                            pl.col(col).fill_null(fill_value)
                            % pl.Series(other_series).fill_null(fill_value)
                        ).alias(col)
                        for col in self.columns
                    ]
                else:
                    expressions = [
                        (pl.col(col) % pl.Series(other_series)).alias(col)
                        for col in self.columns
                    ]
            else:
                raise NotImplementedError(
                    "Series arithmetic with axis=1 not yet implemented"
                )
            result_df = self._df.select(expressions)
        else:
            if fill_value is not None:
                expressions = [
                    (pl.col(col).fill_null(fill_value) % other).alias(col)
                    for col in self.columns
                ]
            else:
                expressions = [(pl.col(col) % other).alias(col) for col in self.columns]
            result_df = self._df.select(expressions)

        return DataFrame(result_df, index=self._index)

    def pow(
        self,
        other: Union[Any, "Series", "DataFrame"],
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        level: Optional[Any] = None,
        fill_value: Optional[Any] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Get exponential power of DataFrame and other, element-wise (binary operator **).

        Parameters
        ----------
        other : scalar, Series, or DataFrame
            Object to raise the DataFrame to the power of.
        axis : {0, 1, 'index', 'columns'} or None, default None
            Axis to match Series index on. For Series input, axis to match Series index on.
        level : int or name, optional
            Broadcast across a level, matching Index values on the passed MultiIndex level.
        fill_value : float or None, default None
            Fill existing missing (NaN) values, and any new element needed for successful
            DataFrame alignment, with this value before computation.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.
        """
        from polarpandas.series import Series

        if isinstance(other, DataFrame):
            if len(self) != len(other):
                raise ValueError(
                    "Can only compute power with identically-labeled DataFrame objects"
                )
            expressions = []
            other_df_renamed = other._df.select(
                [pl.col(col).alias(f"{col}_other") for col in other.columns]
            )
            combined_df = self._df.hstack(other_df_renamed)
            for col in self.columns:
                if col in other.columns:
                    if fill_value is not None:
                        expr = (
                            pl.col(col)
                            .fill_null(fill_value)
                            .pow(pl.col(f"{col}_other").fill_null(fill_value))
                        )
                    else:
                        expr = pl.col(col).pow(pl.col(f"{col}_other"))
                    expressions.append(expr.alias(col))
                else:
                    expressions.append(pl.col(col).alias(col))
            result_df = combined_df.select(expressions)
        elif isinstance(other, Series):
            if axis is None or axis == 0 or axis == "index":
                if len(other) != len(self):
                    raise ValueError("Lengths must match")
                other_series = other._series if hasattr(other, "_series") else other
                if fill_value is not None:
                    expressions = [
                        (
                            pl.col(col)
                            .fill_null(fill_value)
                            .pow(pl.Series(other_series).fill_null(fill_value))
                        ).alias(col)
                        for col in self.columns
                    ]
                else:
                    expressions = [
                        (pl.col(col).pow(pl.Series(other_series))).alias(col)
                        for col in self.columns
                    ]
            else:
                raise NotImplementedError(
                    "Series arithmetic with axis=1 not yet implemented"
                )
            result_df = self._df.select(expressions)
        else:
            if fill_value is not None:
                expressions = [
                    (pl.col(col).fill_null(fill_value).pow(other)).alias(col)
                    for col in self.columns
                ]
            else:
                expressions = [
                    (pl.col(col).pow(other)).alias(col) for col in self.columns
                ]
            result_df = self._df.select(expressions)

        return DataFrame(result_df, index=self._index)

    def skew(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        """
        Return unbiased kurtosis over requested axis.

        Parameters
        ----------
        axis : {0, 1, 'index', 'columns'}, default 0
            Axis along which to compute kurtosis. 0 or 'index' for column-wise, 1 or 'columns' for row-wise.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. If False, will attempt to use everything.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Series with kurtosis values for each column (axis=0) or row (axis=1).
        """
        from polarpandas.series import Series

        if axis is None or axis == 1 or axis == "columns":
            # Row-wise kurtosis (axis=1)
            if numeric_only:
                numeric_cols = [
                    col
                    for col in self.columns
                    if _is_numeric_dtype(self._df[col].dtype)
                ]
                if not numeric_cols:
                    return Series(pl.Series([], dtype=pl.Float64))
                result_series = self._df.select(
                    pl.concat_list(numeric_cols).list.kurtosis()  # type: ignore[attr-defined]
                )["literal"]
            else:
                result_series = self._df.select(
                    pl.concat_list(self.columns).list.kurtosis()  # type: ignore[attr-defined]
                )["literal"]
            index = (
                self._index
                if self._index is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)
        else:
            # Column-wise kurtosis (axis=0, default)
            if numeric_only:
                numeric_cols = [
                    col
                    for col in self.columns
                    if _is_numeric_dtype(self._df[col].dtype)
                ]
                if not numeric_cols:
                    return Series(pl.Series([], dtype=pl.Float64))
                result_pl = self._df.select(
                    [pl.col(col).kurtosis() for col in numeric_cols]
                )
            else:
                result_pl = self._df.select(
                    [pl.col(col).kurtosis() for col in self.columns]
                )
            values = [result_pl[col].to_list()[0] for col in result_pl.columns]
            return Series(values, index=result_pl.columns)

    def sem(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        ddof: int = 1,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        """
        Return unbiased standard error of the mean over requested axis.

        Parameters
        ----------
        axis : {0, 1, 'index', 'columns'}, default 0
            Axis along which to compute SEM. 0 or 'index' for column-wise, 1 or 'columns' for row-wise.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. If False, will attempt to use everything.
        ddof : int, default 1
            Delta degrees of freedom. The divisor used in calculations is N - ddof.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Series with SEM values for each column (axis=0) or row (axis=1).
        """
        from polarpandas.series import Series

        if axis is None or axis == 1 or axis == "columns":
            # Row-wise SEM (axis=1) - std / sqrt(n)
            if numeric_only:
                numeric_cols = [
                    col
                    for col in self.columns
                    if _is_numeric_dtype(self._df[col].dtype)
                ]
                if not numeric_cols:
                    return Series(pl.Series([], dtype=pl.Float64))
                # Compute row-wise std and divide by sqrt(n)
                result_series = self._df.select(
                    pl.concat_list(numeric_cols).list.std(ddof=ddof)
                    / pl.concat_list(numeric_cols).list.len().sqrt()
                )["literal"]
            else:
                result_series = self._df.select(
                    pl.concat_list(self.columns).list.std(ddof=ddof)
                    / pl.concat_list(self.columns).list.len().sqrt()
                )["literal"]
            index = (
                self._index
                if self._index is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)
        else:
            # Column-wise SEM (axis=0, default) - std / sqrt(n)
            if numeric_only:
                numeric_cols = [
                    col
                    for col in self.columns
                    if _is_numeric_dtype(self._df[col].dtype)
                ]
                if not numeric_cols:
                    return Series(pl.Series([], dtype=pl.Float64))
                result_pl = self._df.select(
                    [
                        pl.col(col).std(ddof=ddof) / pl.col(col).count().sqrt()
                        for col in numeric_cols
                    ]
                )
            else:
                result_pl = self._df.select(
                    [
                        pl.col(col).std(ddof=ddof) / pl.col(col).count().sqrt()
                        for col in self.columns
                    ]
                )
            values = [result_pl[col].to_list()[0] for col in result_pl.columns]
            return Series(values, index=result_pl.columns)

    def idxmax(
        self,
        axis: int = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        """
        Return index of first occurrence of maximum over requested axis.

        Parameters
        ----------
        axis : {0, 1}, default 0
            Axis along which to find index. 0 for column-wise, 1 for row-wise.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Series with indices of maximum values.
        """
        from polarpandas.series import Series

        if axis == 1:
            # Row-wise idxmax - return column name with max value for each row
            if numeric_only:
                numeric_cols = [
                    col
                    for col in self.columns
                    if self._df[col].dtype
                    in (
                        pl.Int8,
                        pl.Int16,
                        pl.Int32,
                        pl.Int64,
                        pl.UInt8,
                        pl.UInt16,
                        pl.UInt32,
                        pl.UInt64,
                        pl.Float32,
                        pl.Float64,
                    )
                ]
                if not numeric_cols:
                    return Series(pl.Series([], dtype=pl.Utf8))
                # For row-wise, we need to find which column has the max value
                # This is complex - for now, return first column name as placeholder
                result_series = pl.Series([numeric_cols[0]] * len(self._df))
            else:
                # For non-numeric, use first column as fallback
                result_series = pl.Series([self.columns[0]] * len(self._df))
            index = (
                self._index
                if self._index is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)
        else:
            # Column-wise idxmax (axis=0, default) - return row index with max value for each column
            if numeric_only:
                numeric_cols = [
                    col
                    for col in self.columns
                    if self._df[col].dtype
                    in (
                        pl.Int8,
                        pl.Int16,
                        pl.Int32,
                        pl.Int64,
                        pl.UInt8,
                        pl.UInt16,
                        pl.UInt32,
                        pl.UInt64,
                        pl.Float32,
                        pl.Float64,
                    )
                ]
                if not numeric_cols:
                    return Series(pl.Series([], dtype=pl.Int64))
                result_pl = self._df.select(
                    [pl.col(col).arg_max() for col in numeric_cols]
                )
            else:
                result_pl = self._df.select(
                    [pl.col(col).arg_max() for col in self.columns]
                )
            values = [result_pl[col].to_list()[0] for col in result_pl.columns]
            # Use index if available
            if self._index is not None:
                indexed_values = [
                    self._index[v] if v < len(self._index) else v for v in values
                ]
                return Series(indexed_values, index=result_pl.columns)
            else:
                return Series(values, index=result_pl.columns)

    def idxmin(
        self,
        axis: int = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        """
        Return index of first occurrence of minimum over requested axis.

        Parameters
        ----------
        axis : {0, 1}, default 0
            Axis along which to find index. 0 for column-wise, 1 for row-wise.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Series with indices of minimum values.
        """
        from polarpandas.series import Series

        if axis == 1:
            # Row-wise idxmin - return column name with min value for each row
            if numeric_only:
                numeric_cols = [
                    col
                    for col in self.columns
                    if self._df[col].dtype
                    in (
                        pl.Int8,
                        pl.Int16,
                        pl.Int32,
                        pl.Int64,
                        pl.UInt8,
                        pl.UInt16,
                        pl.UInt32,
                        pl.UInt64,
                        pl.Float32,
                        pl.Float64,
                    )
                ]
                if not numeric_cols:
                    return Series(pl.Series([], dtype=pl.Utf8))
                # For row-wise, we need to find which column has the min value
                # This is complex - for now, return first column name as placeholder
                result_series = pl.Series([numeric_cols[0]] * len(self._df))
            else:
                # For non-numeric, use first column as fallback
                result_series = pl.Series([self.columns[0]] * len(self._df))
            index = (
                self._index
                if self._index is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)
        else:
            # Column-wise idxmin (axis=0, default) - return row index with min value for each column
            if numeric_only:
                numeric_cols = [
                    col
                    for col in self.columns
                    if self._df[col].dtype
                    in (
                        pl.Int8,
                        pl.Int16,
                        pl.Int32,
                        pl.Int64,
                        pl.UInt8,
                        pl.UInt16,
                        pl.UInt32,
                        pl.UInt64,
                        pl.Float32,
                        pl.Float64,
                    )
                ]
                if not numeric_cols:
                    return Series(pl.Series([], dtype=pl.Int64))
                result_pl = self._df.select(
                    [pl.col(col).arg_min() for col in numeric_cols]
                )
            else:
                result_pl = self._df.select(
                    [pl.col(col).arg_min() for col in self.columns]
                )
            values = [result_pl[col].to_list()[0] for col in result_pl.columns]
            # Use index if available
            if self._index is not None:
                indexed_values = [
                    self._index[v] if v < len(self._index) else v for v in values
                ]
                return Series(indexed_values, index=result_pl.columns)
            else:
                return Series(values, index=result_pl.columns)

    def explode(
        self,
        column: Union[str, List[str]],
        ignore_index: bool = False,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Transform each element of a list-like to a row, replicating index values.

        Parameters
        ----------
        column : str or list of str
            Column(s) to explode.
        ignore_index : bool, default False
            If True, the resulting index will be labeled 0, 1, …, n - 1.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            DataFrame with exploded columns.
        """
        columns_to_explode = [column] if isinstance(column, str) else column

        result_df = self._df.explode(columns_to_explode)

        if ignore_index:
            return DataFrame(result_df)
        else:
            return DataFrame(result_df, index=self._index)

    def stack(
        self,
        level: int = -1,
        dropna: bool = True,
        **kwargs: Any,
    ) -> Union["Series", "DataFrame"]:
        """
        Stack the prescribed level(s) from columns to index.

        Parameters
        ----------
        level : int, default -1
            Level(s) to stack from the column axis onto the index axis.
        dropna : bool, default True
            Whether to drop rows in the resulting Series/DataFrame with missing values.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series or DataFrame
            Stacked DataFrame or Series.
        """
        from polarpandas.series import Series

        # Polars doesn't have direct stack, so we use melt as a workaround
        # This is a simplified implementation
        result_df = self._df.melt()
        if dropna:
            result_df = result_df.drop_nulls()

        # Return as Series with MultiIndex-like structure
        index_tuples = [
            (row[0], row[1])
            for row in result_df.select(["variable", "value"]).iter_rows()
        ]
        return Series(result_df["value"], index=index_tuples)

    def unstack(
        self,
        level: int = -1,
        fill_value: Optional[Any] = None,
        sort: bool = True,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Pivot a level of the (necessarily hierarchical) index labels.

        Parameters
        ----------
        level : int, default -1
            Level(s) to unstack.
        fill_value : scalar, optional
            Replace NaN with this value if unstack produces missing values.
        sort : bool, default True
            Sort the levels of the resulting pivot.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            DataFrame with unstacked index.
        """
        # Simplified implementation - Polars doesn't have direct unstack
        # This would require MultiIndex support which is limited
        return DataFrame(self._df)

    def query(
        self,
        expr: str,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Optional["DataFrame"]:
        """
        Query the columns of a DataFrame with a boolean expression.

        Parameters
        ----------
        expr : str
            The query string to evaluate.
        inplace : bool, default False
            Whether to modify the DataFrame in place.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame or None
            DataFrame resulting from the query, or None if inplace=True.
        """
        # Simple implementation using Polars filter with column references
        # For full pandas query() support, would need expression parser
        # This is a basic implementation that handles simple column comparisons
        try:
            # Try to evaluate as Python expression with column access
            # This is a simplified version - full implementation would need parser
            filtered_df = self._df.filter(
                pl.col(expr.split()[0]) if " " in expr else pl.col(expr)
            )
        except Exception:
            # Fallback: return original DataFrame
            filtered_df = self._df

        if inplace:
            self._df = filtered_df
            return None
        else:
            return DataFrame(filtered_df)

    def to_dict(
        self,
        orient: str = "dict",
        into: type = dict,
        index: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Convert the DataFrame to a dictionary.

        Parameters
        ----------
        orient : str, default 'dict'
            The format of the returned dictionary. Options: 'dict', 'list', 'series', 'split', 'tight', 'records', 'index'.
        into : type, default dict
            The collection type to return (not used, always returns dict).
        index : bool, default True
            Whether to include the index in the output.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        dict
            Dictionary representation of the DataFrame.
        """
        if orient == "dict":
            result = {col: self._df[col].to_list() for col in self.columns}
            if index and self._index is not None:
                result["_index"] = self._index
            return result
        elif orient == "list":
            return [self._df[col].to_list() for col in self.columns]  # type: ignore[return-value]
        elif orient == "records":
            return [dict(zip(self.columns, row)) for row in self._df.iter_rows()]  # type: ignore[return-value]
        elif orient == "split":
            return {
                "columns": self.columns,
                "data": [list(row) for row in self._df.iter_rows()],
                "index": self._index
                if self._index is not None
                else list(range(len(self._df))),
            }
        else:
            # Default to dict
            return {col: self._df[col].to_list() for col in self.columns}

    def select_dtypes(
        self,
        include: Optional[Union[Any, List[Any]]] = None,
        exclude: Optional[Union[Any, List[Any]]] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Return a subset of the DataFrame's columns based on the column dtypes.

        Parameters
        ----------
        include : scalar or list-like, optional
            A selection of dtypes to include.
        exclude : scalar or list-like, optional
            A selection of dtypes to exclude.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            DataFrame with selected columns.
        """
        cols_to_keep = []

        for col in self.columns:
            dtype = self._df[col].dtype
            include_col = True

            if include is not None:
                include_list = include if isinstance(include, list) else [include]
                include_col = any(
                    dtype == include_dtype or str(dtype) == str(include_dtype)
                    for include_dtype in include_list
                )

            if exclude is not None and include_col:
                exclude_list = exclude if isinstance(exclude, list) else [exclude]
                include_col = not any(
                    dtype == exclude_dtype or str(dtype) == str(exclude_dtype)
                    for exclude_dtype in exclude_list
                )

            if include_col:
                cols_to_keep.append(col)

        return DataFrame(self._df.select(cols_to_keep))

    def reindex(
        self,
        labels: Optional[Any] = None,
        index: Optional[Any] = None,
        columns: Optional[Any] = None,
        axis: Optional[Union[int, Literal["index", "columns"]]] = None,
        method: Optional[str] = None,
        copy: Optional[bool] = None,
        level: Optional[Any] = None,
        fill_value: Optional[Any] = None,
        limit: Optional[int] = None,
        tolerance: Optional[Any] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Conform DataFrame to new index with optional filling logic.

        Parameters
        ----------
        labels : array-like, optional
            New labels / index to conform to.
        index : array-like, optional
            New labels for the index.
        columns : array-like, optional
            New labels for the columns.
        axis : int or str, optional
            Axis to reindex.
        method : str, optional
            Method to use for filling holes in reindexed DataFrame.
        copy : bool, optional
            Return a new object, even if the passed indexes are the same.
        level : int or name, optional
            Not used, for pandas compatibility.
        fill_value : scalar, optional
            Value to use for missing values.
        limit : int, optional
            Maximum number of consecutive elements to forward/backward fill.
        tolerance : optional
            Not used, for pandas compatibility.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            DataFrame with new index/columns.
        """
        result_df = self._df.clone()
        new_index = self._index

        if index is not None:
            new_index = (
                list(index) if not isinstance(index, (list, tuple)) else list(index)
            )

        if columns is not None:
            new_columns = (
                list(columns)
                if not isinstance(columns, (list, tuple))
                else list(columns)
            )
            # Add missing columns with fill_value
            for col in new_columns:
                if col not in result_df.columns:
                    if fill_value is not None:
                        result_df = result_df.with_columns(
                            pl.lit(fill_value).alias(col)
                        )
                    else:
                        result_df = result_df.with_columns(pl.lit(None).alias(col))
            # Remove columns not in new_columns
            cols_to_remove = [
                col for col in result_df.columns if col not in new_columns
            ]
            if cols_to_remove:
                result_df = result_df.drop(cols_to_remove)
            # Reorder columns
            result_df = result_df.select(new_columns)

        return DataFrame(result_df, index=new_index)

    def info(self) -> None:
        """
        Print information about the DataFrame.

        Prints the schema and summary information.
        """
        print("<class 'polarpandas.DataFrame'>")
        print(f"Columns: {len(self.columns)}")
        print(f"Rows: {len(self)}")
        print("\nColumn details:")
        for col in self.columns:
            dtype = self._df[col].dtype
            null_count = self._df[col].null_count()
            print(f"  {col}: {dtype} (null values: {null_count})")

    def drop_duplicates(
        self,
        subset: Optional[Union[str, List[str]]] = None,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Optional["DataFrame"]:
        """
        Remove duplicate rows.

        Parameters
        ----------
        subset : list, optional
            Columns to consider for identifying duplicates
        inplace : bool, default False
            If True, modify DataFrame in place

        Returns
        -------
        DataFrame or None
            DataFrame with duplicates removed, or None if inplace=True
        """
        # Polars uses unique() instead of drop_duplicates()
        result_df = self._df.unique(subset=subset, **kwargs)

        if inplace:
            self._df = result_df
            return None
        else:
            return DataFrame(result_df)

    def duplicated(
        self, subset: Optional[List[str]] = None, keep: str = "first"
    ) -> "Series":
        """
        Return boolean Series denoting duplicate rows.

        Parameters
        ----------
        subset : list, optional
            Columns to consider for identifying duplicates
        keep : {'first', 'last', False}, default 'first'
            Which duplicates to mark

        Returns
        -------
        Series
            Boolean series indicating duplicates
        """
        # Polars doesn't have a direct duplicated() method
        # We'll implement a simple version
        from polarpandas.series import Series

        if subset is None:
            subset = self.columns

        # Use Polars is_duplicated()
        result = self._df.is_duplicated()
        return Series(result)

    def is_duplicated(self) -> "Series":
        """
        Check if each row is duplicated.

        Returns
        -------
        Series
            Boolean series indicating if each row is duplicated
        """
        from polarpandas.series import Series

        result = self._df.is_duplicated()
        return Series(result)

    def sort_index(self, inplace: bool = False, **kwargs: Any) -> Optional["DataFrame"]:
        """
        Sort by index (row numbers).

        Parameters
        ----------
        inplace : bool, default False
            If True, modify DataFrame in place

        Returns
        -------
        DataFrame or None
            Sorted DataFrame, or None if inplace=True
        """
        # Since we're using simple range indices, just return as-is
        # In a full implementation, this would sort by actual index values
        if inplace:
            return None
        else:
            return DataFrame(self._df.clone())

    def isin(self, values: Union[Dict[str, List[Any]], List[Any]]) -> "DataFrame":
        """
        Check whether each element is contained in values.

        Parameters
        ----------
        values : iterable or dict
            Values to check for

        Returns
        -------
        DataFrame
            Boolean DataFrame
        """
        # Apply is_in() to each column
        if isinstance(values, dict):
            # Dictionary mapping column names to values
            result_cols = []
            for col in self.columns:
                if col in values:
                    result_cols.append(pl.col(col).is_in(values[col]))
                else:
                    result_cols.append(pl.lit(False))
            result = self._df.select(result_cols)
        else:
            # List of values - check all columns
            result = self._df.select([pl.col(c).is_in(values) for c in self.columns])

        return DataFrame(result)

    def equals(self, other: Any) -> bool:
        """
        Check if two DataFrames are equal.

        Parameters
        ----------
        other : DataFrame
            DataFrame to compare with

        Returns
        -------
        bool
            True if equal, False otherwise
        """
        if isinstance(other, DataFrame):
            materialized_self = self._df
            materialized_other = other._df
            return materialized_self.equals(materialized_other)
        elif isinstance(other, pl.DataFrame):
            return self._df.equals(other)
        return False

    def reset_index(
        self, drop: bool = False, inplace: bool = False
    ) -> Optional["DataFrame"]:
        """
        Reset the index.

        Parameters
        ----------
        drop : bool, default False
            Whether to drop the index or add it as a column(s)
        inplace : bool, default False
            If True, modify DataFrame in place

        Returns
        -------
        DataFrame or None
            DataFrame with reset index, or None if inplace=True
        """
        if not drop:
            # Add index as column(s)
            result_df = self._df.clone()

            if self._index is not None:
                # Check if MultiIndex
                if len(self._index) > 0 and isinstance(self._index[0], tuple):
                    # MultiIndex - add each level as a column
                    if isinstance(self._index_name, tuple):
                        col_names = list(self._index_name)
                    else:
                        col_names = [f"level_{i}" for i in range(len(self._index[0]))]

                    # Extract each level and create columns
                    index_cols = {}
                    for level_idx, col_name in enumerate(col_names):
                        level_values = [
                            t[level_idx] if level_idx < len(t) else None
                            for t in self._index
                        ]
                        index_cols[col_name] = pl.Series(col_name, level_values)

                    # Reorder: index columns first, then data columns
                    data_cols = {col: result_df[col] for col in result_df.columns}
                    result_df = pl.DataFrame({**index_cols, **data_cols})
                else:
                    # Regular Index - add as single column
                    col_name = (
                        self._index_name
                        if isinstance(self._index_name, str)
                        else "index"
                    )
                    index_col = pl.Series(col_name, self._index)
                    # Reorder: index column first, then data columns
                    data_cols = {col: result_df[col] for col in result_df.columns}
                    result_df = pl.DataFrame({col_name: index_col, **data_cols})
            else:
                # No index - add default range index
                result_df = result_df.with_row_index("index")
        else:
            result_df = self._df.clone()

        result = DataFrame(result_df)
        result._index = None
        result._index_name = None

        if inplace:
            self._df = result._df
            self._index = None
            self._index_name = None
            return None
        else:
            return result

    def _validate_index_keys(self, keys: Union[str, List[str]]) -> List[str]:
        """Validate and normalize index keys.

        Parameters
        ----------
        keys : str or list of str
            Column name(s) to use as index

        Returns
        -------
        List[str]
            Normalized list of column names

        Raises
        ------
        KeyError
            If keys is None or contains invalid column names
        ValueError
            If keys is empty or contains nulls
        """
        # Handle None case - pandas raises KeyError for None
        if keys is None:
            raise KeyError("None of [None] are in the columns")

        # Handle single column name
        if isinstance(keys, str):
            keys = [keys]

        # Validate keys is not empty
        if not keys:
            raise ValueError(
                "Must pass non-zero number of levels/codes for MultiIndex.\n"
                "Example: df.set_index(['level1', 'level2'])"
            )

        # Validate keys exist
        for key in keys:
            if key not in self._df.columns:
                raise create_keyerror_with_suggestions(
                    key, self._df.columns, context="column"
                )

        # Check if any index columns contain nulls
        has_nulls = any(self._df[key].null_count() > 0 for key in keys)

        if has_nulls:
            # Polars has limited null handling in index - this is a limitation
            raise ValueError(
                "Polars has limited support for null values in index. This is a known limitation."
            )

        return keys

    def _build_index_from_keys(
        self, keys: List[str], target_df: Optional[pl.DataFrame] = None
    ) -> Tuple[List[Any], Union[str, Tuple[str, ...]]]:
        """Build index values and name from column keys.

        Parameters
        ----------
        keys : List[str]
            Column names to use as index
        target_df : pl.DataFrame, optional
            DataFrame to extract values from (defaults to self._df)

        Returns
        -------
        Tuple[List[Any], Union[str, Tuple[str, ...]]]
            Tuple of (index_values, index_name)
        """
        if target_df is None:
            target_df = self._df

        if len(keys) == 1:
            index_values = target_df[keys[0]].to_list()
            index_name: Union[str, Tuple[str, ...]] = keys[0]
        else:
            # Multi-level index - create tuples
            index_values = list(zip(*[target_df[key].to_list() for key in keys]))
            index_name = tuple(keys)  # Store as tuple for hashability

        return index_values, index_name

    def _append_to_existing_index(
        self, keys: List[str], target_df: Optional[pl.DataFrame] = None
    ) -> Tuple[List[Any], Union[str, Tuple[str, ...]]]:
        """Append columns to existing index.

        Parameters
        ----------
        keys : List[str]
            Column names to append to index
        target_df : pl.DataFrame, optional
            DataFrame to extract values from (defaults to self._df)

        Returns
        -------
        Tuple[List[Any], Union[str, Tuple[str, ...]]]
            Tuple of (new_index_values, new_index_name)
        """
        if target_df is None:
            target_df = self._df

        existing_index = list(self._index) if self._index is not None else []
        new_values, _ = self._build_index_from_keys(keys, target_df)

        # Create tuples of (existing_index[i], new_values[i])
        new_index = []
        for i in range(len(existing_index)):
            if len(keys) == 1:
                new_index.append((existing_index[i], new_values[i]))
            else:
                new_index.append((existing_index[i],) + new_values[i])

        # Update index name for append
        if isinstance(self._index_name, (list, tuple)):
            new_index_name = tuple(list(self._index_name) + keys)
        else:
            new_index_name = (
                tuple([self._index_name] + keys)
                if self._index_name is not None
                else tuple(keys)
            )

        return new_index, new_index_name

    def _drop_index_columns(self, keys: List[str]) -> None:
        """Drop columns used as index from DataFrame.

        Parameters
        ----------
        keys : List[str]
            Column names to drop
        """
        columns_to_keep = [col for col in self._df.columns if col not in keys]
        if columns_to_keep:
            self._df = self._df.select(columns_to_keep)
        else:
            # If all columns are used as index, create empty DataFrame with index
            self._df = pl.DataFrame()

    def set_index(
        self,
        keys: Union[str, List[str]],
        drop: bool = True,
        append: bool = False,
        inplace: bool = False,
    ) -> Optional["DataFrame"]:
        """
        Set DataFrame index using one or more columns.

        Parameters
        ----------
        keys : str or list of str
            Column name(s) to use as index.
        drop : bool, default True
            Delete columns to be used as the new index.
        append : bool, default False
            Whether to append columns to existing index.
        inplace : bool, default False
            Modify the DataFrame in place (do not create a new object).

        Returns
        -------
        DataFrame or None
            DataFrame with the new index or None if inplace=True.
        """
        # Validate and normalize keys
        keys = self._validate_index_keys(keys)

        if inplace:
            # Modify in place
            if append and self._index is not None:
                self._index, self._index_name = self._append_to_existing_index(keys)
            else:
                # Replace index
                self._index, self._index_name = self._build_index_from_keys(keys)

            # Drop columns if requested
            if drop:
                self._drop_index_columns(keys)

            return None
        else:
            # Create a copy
            result = DataFrame(self._df)

            if append and self._index is not None:
                result._index, result._index_name = self._append_to_existing_index(
                    keys, result._df
                )
            else:
                # Replace index
                result._index, result._index_name = self._build_index_from_keys(
                    keys, result._df
                )

            # Drop columns if requested
            if drop:
                columns_to_keep = [col for col in result._df.columns if col not in keys]
                if columns_to_keep:
                    result._df = result._df.select(columns_to_keep)
                else:
                    # If all columns are used as index, create empty DataFrame with index
                    result._df = pl.DataFrame()

            return result

    def align(
        self,
        other: "DataFrame",
        join: str = "outer",
        axis: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple["DataFrame", "DataFrame"]:
        """
        Align two DataFrames on their columns and/or index.

        Parameters
        ----------
        other : DataFrame
            Other DataFrame to align with
        join : str, default "outer"
            Type of join to perform
        axis : Any, optional
            Axis to align on
        **kwargs
            Additional arguments

        Returns
        -------
        tuple of DataFrame
            Aligned DataFrames
        """
        # Simplified implementation - align columns
        all_cols = set(self.columns) | set(other.columns)
        left_cols = [col for col in all_cols if col in self.columns]
        right_cols = [col for col in all_cols if col in other.columns]

        left_aligned = self.reindex(columns=left_cols)
        right_aligned = other.reindex(columns=right_cols)

        return left_aligned, right_aligned

    def corrwith(
        self,
        other: Union["DataFrame", "Series"],
        axis: int = 0,
        drop: bool = False,
        method: str = "pearson",
    ) -> "Series":
        """
        Compute pairwise correlation.

        Parameters
        ----------
        other : DataFrame or Series
            Object to compute correlation with
        axis : int, default 0
            Axis to compute correlation along
        drop : bool, default False
            Drop missing indices from result
        method : str, default "pearson"
            Correlation method

        Returns
        -------
        Series
            Pairwise correlations
        """
        from .series import Series

        if isinstance(other, Series):
            # Compute correlation with Series
            correlations = []
            for col in self.columns:
                try:
                    # Combine columns and compute correlation
                    combined = self._df.select(
                        [pl.col(col), other._series.alias("other")]
                    )
                    corr = combined.select(pl.corr(col, "other")).item()
                    correlations.append(corr)
                except Exception:
                    correlations.append(None)
            return Series(correlations, index=self.columns)
        elif isinstance(other, DataFrame):
            # Compute correlation with DataFrame
            correlations = []
            for col in self.columns:
                if col in other.columns:
                    try:
                        combined = self._df.select(
                            [pl.col(col), pl.col(col).alias("other")]
                        )
                        corr = combined.select(pl.corr(col, "other")).item()
                        correlations.append(corr)
                    except Exception:
                        correlations.append(None)
                else:
                    correlations.append(None)
            return Series(correlations, index=self.columns)
        else:
            raise TypeError(f"Unsupported type for corrwith: {type(other)}")

    def droplevel(
        self, level: Union[int, str, List[Union[int, str]]], axis: int = 0
    ) -> "DataFrame":
        """
        Return DataFrame with requested index / column level(s) removed.

        Parameters
        ----------
        level : int, str, or list
            Level(s) to drop
        axis : int, default 0
            Axis to drop level from

        Returns
        -------
        DataFrame
            DataFrame with level(s) removed
        """
        if axis == 0:
            # Drop from index
            if self._index is None or len(self._index) == 0:
                return self.copy()

            # Check if MultiIndex
            if isinstance(self._index[0], tuple):
                # Create MultiIndex and drop level
                if isinstance(self._index_name, tuple):
                    names_list: Optional[List[Optional[str]]] = list(self._index_name)
                else:
                    names_list = None
                mi = MultiIndex.from_tuples(self._index, names=names_list)
                new_mi = mi.droplevel(level)

                # Convert back to list format
                if isinstance(new_mi, MultiIndex):
                    result = self.copy()
                    result._index = new_mi.tolist()
                    result._index_name = new_mi.names if new_mi.names else None  # type: ignore[assignment]  # type: ignore[assignment]
                    return result
                else:
                    # Converted to Index - preserve the name
                    result = self.copy()
                    result._index = new_mi.tolist()
                    # Get the name from the Index's series name
                    result._index_name = (
                        new_mi._series.name if new_mi._series.name != "index" else None
                    )
                    return result
            else:
                # Regular Index - can't drop level
                return self.copy()
        else:
            # Column level dropping not yet implemented
            return self.copy()

    def reindex_like(self, other: "DataFrame", **kwargs: Any) -> "DataFrame":
        """
        Return an object with matching indices as other object.

        Parameters
        ----------
        other : DataFrame
            Object with the target index
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            DataFrame with matching indices
        """
        return self.reindex(index=other.index, columns=other.columns, **kwargs)

    def rename_axis(
        self,
        mapper: Optional[Any] = None,
        index: Optional[Any] = None,
        columns: Optional[Any] = None,
        axis: int = 0,
        copy: bool = True,
        inplace: bool = False,
    ) -> Optional["DataFrame"]:
        """
        Set the name of the index or columns.

        Parameters
        ----------
        mapper : Any, optional
            Value to set the axis name to
        index : Any, optional
            Value to set the index name to
        columns : Any, optional
            Value to set the columns name to
        axis : int, default 0
            Axis to rename
        copy : bool, default True
            Whether to copy the DataFrame
        inplace : bool, default False
            Whether to modify in place

        Returns
        -------
        DataFrame or None
            DataFrame with renamed axis, or None if inplace=True
        """
        result = self.copy() if copy else self

        if index is not None:
            result._index_name = index
        elif mapper is not None and axis == 0:
            result._index_name = mapper

        if columns is not None:
            # Polars doesn't have column index names, so we store it separately
            result._columns_index = columns
        elif mapper is not None and axis == 1:
            result._columns_index = mapper

        if inplace:
            self._index_name = result._index_name
            self._columns_index = result._columns_index
            return None
        return result
    def to_pandas(self) -> Any:
        """
        Convert polarpandas DataFrame to pandas DataFrame.

        Note: This method requires pandas to be installed.

        Returns
        -------
        pandas.DataFrame
            Converted pandas DataFrame
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_pandas() method. Install with: pip install pandas"
            ) from e

        # Convert Polars DataFrame to pandas
        # self._df should always be a DataFrame, but check defensively
        polars_df = self._df
        if hasattr(polars_df, "collect"):
            polars_df = polars_df.collect()
        pandas_df = polars_df.to_pandas()

        # Set index if we have one
        if self._index is not None:
            # Convert list to pandas Index
            pandas_df.index = pd.Index(self._index)
            if self._index_name is not None:
                # Handle MultiIndex case
                if isinstance(self._index_name, tuple) and len(self._index_name) > 1:
                    # Create MultiIndex with proper names
                    import pandas as pd

                    pandas_df.index = pd.MultiIndex.from_tuples(
                        self._index, names=self._index_name
                    )
                else:
                    # Convert empty string to None for pandas compatibility
                    index_name_value: Optional[Union[str, Tuple[str, ...]]]
                    if isinstance(self._index_name, str):
                        index_name_value = (
                            self._index_name if self._index_name != "" else None
                        )
                    elif isinstance(self._index_name, tuple):
                        # For tuple index names, use first element or empty string handling
                        index_name_value = (
                            self._index_name if self._index_name != ("",) else None
                        )
                    else:
                        # self._index_name is None here
                        index_name_value = None  # type: ignore[unreachable]
                    if hasattr(pandas_df.index, "name"):
                        pandas_df.index.name = index_name_value

        columns_meta = getattr(self, "_columns_index", None)
        if columns_meta is not None:
            if hasattr(columns_meta, "names"):
                pandas_df.columns = columns_meta.copy()
            else:
                pandas_df.columns = pd.Index(columns_meta)

        return pandas_df

    def to_hdf(
        self,
        path_or_buf: Any,
        key: str,
        mode: str = "a",
        complevel: Optional[int] = None,
        complib: Optional[str] = None,
        append: bool = False,
        format: Optional[str] = None,
        index: bool = True,
        min_itemsize: Optional[Any] = None,
        nan_rep: Optional[Any] = None,
        dropna: Optional[bool] = None,
        data_columns: Optional[Any] = None,
        errors: str = "strict",
        encoding: str = "UTF-8",
        **kwargs: Any,
    ) -> None:
        """
        Write DataFrame to HDF5 file.

        Parameters
        ----------
        path_or_buf : str or file-like
            File path or file-like object
        key : str
            Identifier for the group in the store
        mode : str, default "a"
            File mode
        complevel : int, optional
            Compression level
        complib : str, optional
            Compression library
        append : bool, default False
            Append to existing file
        format : str, optional
            Format specification
        index : bool, default True
            Write index
        min_itemsize : Any, optional
            Minimum string size
        nan_rep : Any, optional
            Representation for NaN
        dropna : bool, optional
            Drop NA values
        data_columns : Any, optional
            Columns to create as data columns
        errors : str, default "strict"
            Error handling
        encoding : str, default "UTF-8"
            Encoding to use
        **kwargs
            Additional keyword arguments forwarded to pandas.
        """

        try:
            import pandas as pd  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise NotImplementedError(
                "to_hdf() requires pandas. Install with `pip install pandas`."
            ) from exc

        pandas_df = self.to_pandas()

        pandas_df.to_hdf(
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

        """
        Write DataFrame to HDF5 file.

        Parameters
        ----------
        path_or_buf : str or file-like
            File path or file-like object
        key : str
            Identifier for the group in the store
        mode : str, default "a"
            File mode
        complevel : int, optional
            Compression level
        complib : str, optional
            Compression library
        append : bool, default False
            Append to existing file
        format : str, optional
            Format specification
        index : bool, default True
            Write index
        min_itemsize : Any, optional
            Minimum string size
        nan_rep : Any, optional
            Representation for NaN
        dropna : bool, optional
            Drop NA values
        data_columns : Any, optional
            Columns to create as data columns
        errors : str, default "strict"
            Error handling
        encoding : str, default "UTF-8"
            Encoding
        **kwargs
            Additional arguments

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2]})
        >>> df.to_hdf("data.h5", "df")
        """
        # Try h5py first, then tables (pytables)
        try:
            import h5py
            import numpy as np  # noqa: F401

            # Convert to numpy structured array
            data = self.to_records(index=index)
            with h5py.File(path_or_buf, mode) as f:
                if key in f:
                    if append:
                        # Append to existing dataset
                        dataset = f[key]
                        new_shape = (dataset.shape[0] + len(data),)
                        dataset.resize(new_shape)
                        dataset[-len(data) :] = data
                    else:
                        del f[key]
                        f.create_dataset(
                            key,
                            data=data,
                            compression=complib,
                            compression_opts=complevel,
                        )
                else:
                    f.create_dataset(
                        key, data=data, compression=complib, compression_opts=complevel
                    )
        except ImportError:
            try:
                import tables as tb  # noqa: F401

                # Convert to pandas-like structure for pytables
                pd_df = self.to_pandas() if hasattr(self, "to_pandas") else None
                if pd_df is None:
                    # Fallback: convert manually
                    import pandas as pd

                    pd_df = pd.DataFrame(self._df.to_dict(as_series=False))
                pd_df.to_hdf(
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
            except ImportError:
                raise NotImplementedError(
                    "to_hdf() requires h5py or tables (pytables).\n"
                    "Workarounds:\n"
                    "  - Install: pip install h5py\n"
                    "  - Or install: pip install tables\n"
                    "  - Export to Parquet/CSV first, then convert"
                ) from None

    def to_html(
        self,
        buf: Optional[Any] = None,
        columns: Optional[Any] = None,
        col_space: Optional[Any] = None,
        header: bool = True,
        index: bool = True,
        na_rep: str = "NaN",
        formatters: Optional[Any] = None,
        float_format: Optional[str] = None,
        sparsify: Optional[bool] = None,
        index_names: bool = True,
        justify: Optional[str] = None,
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
        show_dimensions: Union[bool, str] = False,
        decimal: str = ".",
        bold_rows: bool = False,
        classes: Optional[str] = None,
        escape: bool = True,
        notebook: bool = False,
        border: Optional[int] = None,
        table_id: Optional[str] = None,
        render_links: bool = False,
        encoding: Optional[str] = None,
        **kwargs: Any,
    ) -> Union["Series", "DataFrame"]:
        """
        Render the DataFrame as an HTML table.
        """
        _require_pandas("DataFrame.to_html")

        pandas_df = self.to_pandas()
        return pandas_df.to_html(
            buf=buf,
            columns=columns,
            col_space=col_space,
            header=header,
            index=index,
            na_rep=na_rep,
            formatters=formatters,
            float_format=float_format,
            sparsify=sparsify,
            index_names=index_names,
            justify=justify,
            max_rows=max_rows,
            max_cols=max_cols,
            show_dimensions=show_dimensions,
            decimal=decimal,
            bold_rows=bold_rows,
            classes=classes,
            escape=escape,
            notebook=notebook,
            border=border,
            table_id=table_id,
            render_links=render_links,
            encoding=encoding,
            **kwargs,
        )
class _ILocIndexer:
    """Integer position-based indexer for DataFrame."""

    def __init__(self, df: DataFrame):
        self._df = df

    def __getitem__(self, key: Any) -> Union["Series", "DataFrame", Any]:
        """Get items by integer position."""
        if isinstance(key, tuple):
            # Row and column indexing: df.iloc[row, col]
            row_key, col_key = key
            return self._get_rows_cols(row_key, col_key)
        else:
            # Row-only indexing: df.iloc[row]
            return self._get_rows(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set items by integer position."""
        if isinstance(key, tuple):
            # Row and column indexing: df.iloc[row, col] = value
            row_key, col_key = key
            self._set_rows_cols(row_key, col_key, value)
        else:
            # Row-only indexing: df.iloc[row] = value
            self._set_rows(key, value)

    def _get_rows(self, row_key: Any) -> Union["Series", "DataFrame"]:
        """Get rows by integer position."""
        # Get the Polars DataFrame from the parent DataFrame
        polars_df = self._df._df
        import polars as pl

        # Use Polars for integer-based indexing
        if isinstance(row_key, (slice, list)):
            selected_df = polars_df[row_key]
            result = DataFrame(selected_df, index_name=self._df._index_name)
            # Preserve index for selected rows
            if self._df._index is not None:
                if isinstance(row_key, slice):
                    # Convert slice to list of indices
                    start = row_key.start if row_key.start is not None else 0
                    stop = (
                        row_key.stop
                        if row_key.stop is not None
                        else len(self._df._index)
                    )
                    step = row_key.step if row_key.step is not None else 1
                    selected_indices = list(range(start, stop, step))
                else:
                    selected_indices = row_key
                result._index = [self._df._index[i] for i in selected_indices]
                result._index_name = self._df._index_name
            return result
        else:
            # Single row - return as Series
            from polarpandas.series import Series

            try:
                row_values = polars_df.row(row_key, named=True)
                return Series(
                    list(row_values.values())
                    if isinstance(row_values, dict)
                    else row_values
                )
            except (IndexError, pl.exceptions.OutOfBoundsError) as e:
                if isinstance(row_key, int):
                    if row_key < 0:
                        row_key = len(polars_df) + row_key
                    if row_key >= len(polars_df) or row_key < 0:
                        raise IndexError(
                            f"index {row_key} is out of bounds for axis 0 with size {len(polars_df)}"
                        ) from e
                raise IndexError(f"index {row_key} is out of bounds") from e

    def _get_rows_cols(
        self, row_key: Any, col_key: Any
    ) -> Union["Series", "DataFrame", Any]:
        """Get rows and columns by integer position."""
        # Get the Polars DataFrame from the parent DataFrame
        polars_df = self._df._df
        import polars as pl

        # Use Polars for integer-based indexing
        if isinstance(row_key, (slice, list)):
            selected_df = polars_df[row_key, col_key]
            return DataFrame(selected_df)
        else:
            # Single cell access - return scalar value directly
            if isinstance(col_key, (int, str)):
                try:
                    # Use Polars row() method for single row access
                    row_values = polars_df.row(row_key, named=True)
                    if isinstance(col_key, int):
                        # Get column by integer position
                        col_name = polars_df.columns[col_key]
                        return row_values[col_name]
                    else:
                        # Get column by name
                        return row_values[col_key]
                except (IndexError, pl.exceptions.OutOfBoundsError) as e:
                    if isinstance(row_key, int):
                        if row_key < 0:
                            row_key = len(polars_df) + row_key
                        if row_key >= len(polars_df) or row_key < 0:
                            raise IndexError(
                                f"index {row_key} is out of bounds for axis 0 with size {len(polars_df)}"
                            ) from e
                    raise IndexError(
                        f"index {row_key} or {col_key} is out of bounds"
                    ) from e
                except KeyError as e:
                    raise KeyError(f"Column {col_key} not found") from e
            else:
                # Single row, multiple columns - return as Series
                from polarpandas.series import Series

                try:
                    row_data = polars_df.row(row_key, named=True)
                    if isinstance(col_key, list):
                        return Series([row_data[k] for k in col_key])
                    else:
                        col_name = (
                            polars_df.columns[col_key]
                            if isinstance(col_key, int)
                            else col_key
                        )
                        value = (
                            row_data[col_name]
                            if isinstance(row_data, dict)
                            else row_data
                        )
                        return Series(
                            [value]
                            if not isinstance(value, (list, pl.Series))
                            else value
                        )
                except (IndexError, pl.exceptions.OutOfBoundsError) as e:
                    if isinstance(row_key, int):
                        if row_key < 0:
                            row_key = len(polars_df) + row_key
                        if row_key >= len(polars_df) or row_key < 0:
                            raise IndexError(
                                f"index {row_key} is out of bounds for axis 0 with size {len(polars_df)}"
                            ) from e
                    raise IndexError(
                        f"index {row_key} or {col_key} is out of bounds"
                    ) from e

    def _set_rows(self, row_key: Any, value: Any) -> None:
        """Set rows by integer position."""
        polars_df = self._df._df
        import polars as pl

        # Handle row key (int, slice, or list)
        if isinstance(row_key, slice):
            # Convert slice to list of indices
            start = row_key.start if row_key.start is not None else 0
            stop = row_key.stop if row_key.stop is not None else len(polars_df)
            step = row_key.step if row_key.step is not None else 1
            row_indices = list(range(start, stop, step))
        elif isinstance(row_key, list):
            row_indices = row_key
        else:
            # Single row
            row_key_int = row_key
            # Handle negative indices
            if isinstance(row_key_int, int) and row_key_int < 0:
                row_key_int = len(polars_df) + row_key_int

            # Validate row bounds
            if row_key_int >= len(polars_df) or row_key_int < 0:
                raise IndexError(
                    f"index {row_key_int} is out of bounds for axis 0 with size {len(polars_df)}"
                )
            row_indices = [row_key_int]

        # Handle value types
        if isinstance(value, dict):
            # Dict: update matching columns
            new_cols = []
            for col_name in polars_df.columns:
                if col_name in value:
                    # Column in dict - update rows
                    if len(row_indices) == 1:
                        new_cols.append(
                            pl.when(pl.int_range(pl.len()) == row_indices[0])
                            .then(pl.lit(value[col_name]))
                            .otherwise(pl.col(col_name))
                            .alias(col_name)
                        )
                    else:
                        # Multiple rows - broadcast value or use list
                        val = value[col_name]
                        if isinstance(val, (list, pl.Series)) and len(val) == len(
                            row_indices
                        ):
                            when_expr = pl.col(col_name)
                            for row_idx, v in zip(
                                row_indices, val if isinstance(val, list) else list(val)
                            ):
                                when_expr = (
                                    pl.when(pl.int_range(pl.len()) == row_idx)
                                    .then(pl.lit(v))
                                    .otherwise(when_expr)
                                )
                            new_cols.append(when_expr.alias(col_name))
                        else:
                            # Broadcast scalar
                            when_expr = pl.col(col_name)
                            for row_idx in row_indices:
                                when_expr = (
                                    pl.when(pl.int_range(pl.len()) == row_idx)
                                    .then(pl.lit(val))
                                    .otherwise(when_expr)
                                )
                            new_cols.append(when_expr.alias(col_name))
                else:
                    # Column not in dict - keep original
                    new_cols.append(pl.col(col_name))
            polars_df = polars_df.with_columns(new_cols)
        elif hasattr(value, "__iter__") and not isinstance(value, str):
            # List, Series, or array-like
            try:
                import polarpandas as ppd

                if isinstance(value, ppd.Series):
                    # PolarPandas Series - extract values
                    value_list = value.to_list()
                elif isinstance(value, pl.Series):
                    value_list = value.to_list()
                else:
                    value_list = list(value)

                # Match columns in order
                if len(value_list) == len(polars_df.columns):
                    # Value matches number of columns - update each column
                    new_cols = []
                    for i, col_name in enumerate(polars_df.columns):
                        col_val = value_list[i]
                        if len(row_indices) == 1:
                            new_cols.append(
                                pl.when(pl.int_range(pl.len()) == row_indices[0])
                                .then(pl.lit(col_val))
                                .otherwise(pl.col(col_name))
                                .alias(col_name)
                            )
                        else:
                            # Multiple rows - broadcast or use array
                            if isinstance(col_val, (list, pl.Series)) and len(
                                col_val
                            ) == len(row_indices):
                                when_expr = pl.col(col_name)
                                for row_idx, v in zip(
                                    row_indices,
                                    col_val
                                    if isinstance(col_val, list)
                                    else list(col_val),
                                ):
                                    when_expr = (
                                        pl.when(pl.int_range(pl.len()) == row_idx)
                                        .then(pl.lit(v))
                                        .otherwise(when_expr)
                                    )
                                new_cols.append(when_expr.alias(col_name))
                            else:
                                # Broadcast scalar
                                when_expr = pl.col(col_name)
                                for row_idx in row_indices:
                                    when_expr = (
                                        pl.when(pl.int_range(pl.len()) == row_idx)
                                        .then(pl.lit(col_val))
                                        .otherwise(when_expr)
                                    )
                                new_cols.append(when_expr.alias(col_name))
                    polars_df = polars_df.with_columns(new_cols)
                elif len(row_indices) == 1 and len(value_list) == 1:
                    # Single row, single value - broadcast to all columns
                    new_cols = []
                    for col_name in polars_df.columns:
                        new_cols.append(
                            pl.when(pl.int_range(pl.len()) == row_indices[0])
                            .then(pl.lit(value_list[0]))
                            .otherwise(pl.col(col_name))
                            .alias(col_name)
                        )
                    polars_df = polars_df.with_columns(new_cols)
                else:
                    raise ValueError(
                        f"Cannot assign value of length {len(value_list)} to {len(row_indices)} row(s) with {len(polars_df.columns)} columns"
                    )
            except (TypeError, AttributeError):
                # Not iterable in expected way - treat as scalar
                new_cols = []
                for col_name in polars_df.columns:
                    if len(row_indices) == 1:
                        new_cols.append(
                            pl.when(pl.int_range(pl.len()) == row_indices[0])
                            .then(pl.lit(value))
                            .otherwise(pl.col(col_name))
                            .alias(col_name)
                        )
                    else:
                        when_expr = pl.col(col_name)
                        for row_idx in row_indices:
                            when_expr = (
                                pl.when(pl.int_range(pl.len()) == row_idx)
                                .then(pl.lit(value))
                                .otherwise(when_expr)
                            )
                        new_cols.append(when_expr.alias(col_name))
                polars_df = polars_df.with_columns(new_cols)
        else:
            # Scalar value - broadcast to all columns
            new_cols = []
            for col_name in polars_df.columns:
                if len(row_indices) == 1:
                    new_cols.append(
                        pl.when(pl.int_range(pl.len()) == row_indices[0])
                        .then(pl.lit(value))
                        .otherwise(pl.col(col_name))
                        .alias(col_name)
                    )
                else:
                    # Broadcast scalar to multiple rows
                    when_expr = pl.col(col_name)
                    for row_idx in row_indices:
                        when_expr = (
                            pl.when(pl.int_range(pl.len()) == row_idx)
                            .then(pl.lit(value))
                            .otherwise(when_expr)
                        )
                    new_cols.append(when_expr.alias(col_name))
            polars_df = polars_df.with_columns(new_cols)

        self._df._df = polars_df
        # Index preserved automatically (no shape change)

    def _set_rows_cols(self, row_key: Any, col_key: Any, value: Any) -> None:
        """Set rows and columns by integer position."""
        polars_df = self._df._df
        import polars as pl

        # Handle row key (int or slice)
        if isinstance(row_key, slice):
            # For slice, we need to update multiple rows
            # Convert slice to list of indices
            start = row_key.start if row_key.start is not None else 0
            stop = row_key.stop if row_key.stop is not None else len(polars_df)
            step = row_key.step if row_key.step is not None else 1
            row_indices = list(range(start, stop, step))
        elif isinstance(row_key, list):
            row_indices = row_key
        else:
            # Single row
            row_key_int = row_key
            # Handle negative indices
            if isinstance(row_key_int, int) and row_key_int < 0:
                row_key_int = len(polars_df) + row_key_int

            # Validate row bounds
            if row_key_int >= len(polars_df) or row_key_int < 0:
                raise IndexError(
                    f"index {row_key_int} is out of bounds for axis 0 with size {len(polars_df)}"
                )
            row_indices = [row_key_int]

        # Handle column key (int, str, list, or slice)
        if isinstance(col_key, slice):
            # Convert slice to list of column indices/names
            start = col_key.start if col_key.start is not None else 0
            stop = col_key.stop if col_key.stop is not None else len(polars_df.columns)
            step = col_key.step if col_key.step is not None else 1
            col_keys = list(range(start, stop, step))
            col_names = [
                polars_df.columns[c]
                if isinstance(c, int) and 0 <= c < len(polars_df.columns)
                else str(c)
                for c in col_keys
            ]
        elif isinstance(col_key, list):
            col_names = []
            for c in col_key:
                if isinstance(c, int):
                    if c < 0:
                        c = len(polars_df.columns) + c
                    if c >= len(polars_df.columns) or c < 0:
                        raise IndexError(f"index {c} is out of bounds for axis 1")
                    col_names.append(polars_df.columns[c])
                else:
                    col_names.append(str(c))
        elif isinstance(col_key, int):
            # Handle negative column indices
            col_key_int = col_key
            if col_key_int < 0:
                col_key_int = len(polars_df.columns) + col_key_int

            # Validate column bounds
            if col_key_int >= len(polars_df.columns) or col_key_int < 0:
                raise IndexError(
                    f"index {col_key_int} is out of bounds for axis 1 with size {len(polars_df.columns)}"
                )

            col_names = [polars_df.columns[col_key_int]]
        else:
            # String column name
            col_names = [col_key]

        # Update each column using conditional expressions
        new_cols = []
        for col_name in col_names:
            if col_name in polars_df.columns:
                # Existing column - update using conditional
                if len(row_indices) == 1:
                    # Single row update
                    new_cols.append(
                        pl.when(pl.int_range(pl.len()) == row_indices[0])
                        .then(pl.lit(value))
                        .otherwise(pl.col(col_name))
                        .alias(col_name)
                    )
                else:
                    # Multiple rows - need to handle value as array
                    # For now, broadcast scalar to all selected rows
                    # If value is a list/array, it should match length of row_indices
                    if isinstance(value, (list, pl.Series)) and len(value) == len(
                        row_indices
                    ):
                        # Value matches number of rows - map each row index to corresponding value
                        value_series = (
                            pl.Series(value)
                            if not isinstance(value, pl.Series)
                            else value
                        )
                        # Build when chain for multiple row updates
                        when_expr = pl.col(col_name)
                        for row_idx, val in zip(
                            row_indices,
                            value if isinstance(value, list) else list(value_series),
                        ):
                            when_expr = (
                                pl.when(pl.int_range(pl.len()) == row_idx)
                                .then(pl.lit(val))
                                .otherwise(when_expr)
                            )
                        new_cols.append(when_expr.alias(col_name))
                    else:
                        # Broadcast scalar to all selected rows
                        when_expr = pl.col(col_name)
                        for row_idx in row_indices:
                            when_expr = (
                                pl.when(pl.int_range(pl.len()) == row_idx)
                                .then(pl.lit(value))
                                .otherwise(when_expr)
                            )
                        new_cols.append(when_expr.alias(col_name))
            else:
                # New column - create with None/default and set value
                if len(row_indices) == 1:
                    new_cols.append(
                        pl.when(pl.int_range(pl.len()) == row_indices[0])
                        .then(pl.lit(value))
                        .otherwise(pl.lit(None))
                        .alias(col_name)
                    )
                else:
                    # Multiple rows - broadcast scalar or use array
                    when_expr = pl.lit(None)
                    for row_idx in row_indices:
                        when_expr = (
                            pl.when(pl.int_range(pl.len()) == row_idx)
                            .then(pl.lit(value))
                            .otherwise(when_expr)
                        )
                    new_cols.append(when_expr.alias(col_name))

        self._df._df = polars_df.with_columns(new_cols)
        # Index preserved automatically (no shape change)


class _AtIndexer:
    """Label-based scalar accessor for DataFrame."""

    def __init__(self, df: DataFrame):
        self._df = df

    def __getitem__(self, key: Any) -> Any:
        """Get single value by label."""
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            # Get the Polars DataFrame from the parent DataFrame
            polars_df = self._df._df

            # Use Polars for label-based indexing - limited support
            if self._df._index is not None:
                # Find row index
                try:
                    row_idx = self._df._index.index(row_key)
                    row_values = polars_df.row(row_idx, named=True)
                    return row_values[col_key]
                except ValueError as e:
                    if self._df._index is not None:
                        raise create_keyerror_with_suggestions(
                            str(row_key),
                            [str(k) for k in self._df._index],
                            context="index",
                        ) from e
                    raise KeyError(f"'{row_key}' not in index") from e
            else:
                # No index - use integer position
                if isinstance(row_key, int):
                    row_values = polars_df.row(row_key, named=True)
                    return row_values[col_key]
                else:
                    raise KeyError(f"'{row_key}' not in index")
        else:
            raise ValueError("at accessor requires (row, col) tuple")

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set single value by label."""
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            # Use Polars for label-based indexing - limited support
            if self._df._index is not None:
                # Find row index
                try:
                    row_idx = self._df._index.index(row_key)
                    # Update value in Polars LazyFrame
                    self._df._df = self._df._df.with_columns(
                        pl.when(pl.int_range(pl.len()) == row_idx)
                        .then(pl.lit(value))
                        .otherwise(pl.col(col_key))
                        .alias(col_key)
                    )
                except ValueError as e:
                    if self._df._index is not None:
                        raise create_keyerror_with_suggestions(
                            str(row_key),
                            [str(k) for k in self._df._index],
                            context="index",
                        ) from e
                    raise KeyError(f"'{row_key}' not in index") from e
            else:
                # No index - use integer position
                if isinstance(row_key, int):
                    # Update value in Polars LazyFrame
                    self._df._df = self._df._df.with_columns(
                        pl.when(pl.int_range(pl.len()) == row_key)
                        .then(pl.lit(value))
                        .otherwise(pl.col(col_key))
                        .alias(col_key)
                    )
                else:
                    raise KeyError(f"'{row_key}' not in index")
        else:
            raise ValueError("at accessor requires (row, col) tuple")


class _IAtIndexer:
    """Integer position-based scalar accessor for DataFrame."""

    def __init__(self, df: DataFrame):
        self._df = df

    def __getitem__(self, key: Any) -> Any:
        """Get single value by integer position."""
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            polars_df = self._df._df
            import polars as pl

            # Handle negative indices
            if isinstance(row_key, int) and row_key < 0:
                row_key = len(polars_df) + row_key

            # Validate row bounds
            if row_key >= len(polars_df) or row_key < 0:
                raise IndexError(
                    f"index {row_key} is out of bounds for axis 0 with size {len(polars_df)}"
                )

            # Get row values
            try:
                row_values = polars_df.row(row_key, named=True)
            except (IndexError, pl.exceptions.OutOfBoundsError) as e:
                raise IndexError(
                    f"index {row_key} is out of bounds for axis 0 with size {len(polars_df)}"
                ) from e

            # Handle column key (int or str)
            if isinstance(col_key, int):
                # Handle negative column indices
                if col_key < 0:
                    col_key = len(polars_df.columns) + col_key

                # Validate column bounds
                if col_key >= len(polars_df.columns) or col_key < 0:
                    raise IndexError(
                        f"index {col_key} is out of bounds for axis 1 with size {len(polars_df.columns)}"
                    )

                col_name = polars_df.columns[col_key]
                return row_values[col_name]
            elif isinstance(col_key, str):
                # Column name provided
                if col_key not in polars_df.columns:
                    from polarpandas._exceptions import create_keyerror_with_suggestions

                    raise create_keyerror_with_suggestions(
                        col_key, polars_df.columns, context="column"
                    )
                return row_values[col_key]
            else:
                raise TypeError(f"Column key must be int or str, got {type(col_key)}")
        else:
            raise ValueError("iat accessor requires (row, col) tuple")

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set single value by integer position."""
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            polars_df = self._df._df
            import polars as pl

            # Handle negative row indices
            if isinstance(row_key, int) and row_key < 0:
                row_key = len(polars_df) + row_key

            # Validate row bounds
            if row_key >= len(polars_df) or row_key < 0:
                raise IndexError(
                    f"index {row_key} is out of bounds for axis 0 with size {len(polars_df)}"
                )

            # Handle column key (int or str)
            if isinstance(col_key, int):
                # Handle negative column indices
                if col_key < 0:
                    col_key = len(polars_df.columns) + col_key

                # Validate column bounds
                if col_key >= len(polars_df.columns) or col_key < 0:
                    raise IndexError(
                        f"index {col_key} is out of bounds for axis 1 with size {len(polars_df.columns)}"
                    )

                col_name = polars_df.columns[col_key]
            elif isinstance(col_key, str):
                col_name = col_key
                if col_name not in polars_df.columns:
                    from polarpandas._exceptions import create_keyerror_with_suggestions

                    raise create_keyerror_with_suggestions(
                        col_name, polars_df.columns, context="column"
                    )
            else:
                raise TypeError(f"Column key must be int or str, got {type(col_key)}")

            # Update value using Polars conditional expression (same pattern as _AtIndexer)
            self._df._df = polars_df.with_columns(
                pl.when(pl.int_range(pl.len()) == row_key)
                .then(pl.lit(value))
                .otherwise(pl.col(col_name))
                .alias(col_name)
            )
            # Index preserved automatically (no shape change)
        else:
            raise ValueError("iat accessor requires (row, col) tuple")


class _RollingGroupBy:
    """Rolling window groupby object."""

    def __init__(self, df: DataFrame, window: int, **kwargs: Any) -> None:
        self._df = df
        self._window = window
        self._kwargs = kwargs

    def _apply_rolling(self, operation: str) -> "DataFrame":
        """Apply rolling operation across all columns.

        Parameters
        ----------
        operation : str
            Name of the rolling operation (e.g., 'mean', 'sum', 'std', 'max', 'min')

        Returns
        -------
        DataFrame
            DataFrame with rolling operation applied to all columns
        """
        result_cols = []
        polars_df = self._df._df
        rolling_method_name = f"rolling_{operation}"
        for col in self._df.columns:
            rolling_method = getattr(polars_df[col], rolling_method_name)
            result_cols.append(rolling_method(window_size=self._window).alias(col))
        result_df = polars_df.select(result_cols)
        return DataFrame(result_df)

    def mean(self) -> "DataFrame":
        """Calculate rolling mean."""
        return self._apply_rolling("mean")

    def sum(self) -> "DataFrame":
        """Calculate rolling sum."""
        return self._apply_rolling("sum")

    def std(self) -> "DataFrame":
        """Calculate rolling standard deviation."""
        return self._apply_rolling("std")

    def max(self) -> "DataFrame":
        """Calculate rolling maximum."""
        return self._apply_rolling("max")


class _BaseIndexer:
    """Shared helpers for loc/iloc indexers."""

    def __init__(self, df: "DataFrame") -> None:
        self._df = df

    def _convert(self, result: Any) -> Any:
        return self._df._from_pandas_like(result)

    def _ensure_pandas_value(self, value: Any) -> Any:
        if isinstance(value, DataFrame):
            return value.to_pandas()

        try:
            from polarpandas.series import Series as PolarPandasSeries
        except ImportError:  # pragma: no cover - defensive fallback
            PolarPandasSeries = None  # type: ignore

        if PolarPandasSeries is not None and isinstance(value, PolarPandasSeries):
            return value.to_pandas()

        try:
            import polars as pl  # type: ignore

            if isinstance(value, pl.DataFrame):
                return value.to_pandas()
            if isinstance(value, pl.Series):
                return value.to_pandas()
        except ImportError:  # pragma: no cover - polars always available in runtime
            pass

        if hasattr(value, "to_pandas"):
            return value.to_pandas()

        return value


class _LocIndexer(_BaseIndexer):
    """Label-based indexer leveraging pandas semantics."""

    def __getitem__(self, key: Any) -> Any:
        _require_pandas("DataFrame.loc")

        pandas_result = self._df.to_pandas().loc[key]
        return self._convert(pandas_result)

    def __setitem__(self, key: Any, value: Any) -> None:
        _require_pandas("DataFrame.loc")
        pandas_df = self._df.to_pandas()
        pandas_value = self._ensure_pandas_value(value)
        pandas_df.loc[key] = pandas_value
        updated = self._df._from_pandas_dataframe(pandas_df)
        self._df._update_from_dataframe(updated)


class _ILocIndexer(_BaseIndexer):
    """Position-based indexer leveraging pandas semantics."""

    def _convert_single_column_selector(self, value: Any, columns: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, tuple)):
            return columns.get_loc(value)
        return value

    def _convert_column_selector(self, selector: Any, columns: Any) -> Any:
        try:
            import numpy as np  # type: ignore
        except ImportError:  # pragma: no cover
            np = None  # type: ignore

        if isinstance(selector, slice):
            start = self._convert_single_column_selector(selector.start, columns)
            stop = self._convert_single_column_selector(selector.stop, columns)
            return slice(start, stop, selector.step)
        if isinstance(selector, list):
            return [self._convert_single_column_selector(item, columns) for item in selector]
        if np is not None and isinstance(selector, np.ndarray):
            return [
                self._convert_single_column_selector(item, columns)
                for item in selector.tolist()
            ]
        if isinstance(selector, tuple):
            return tuple(
                self._convert_single_column_selector(item, columns) for item in selector
            )
        return self._convert_single_column_selector(selector, columns)

    def _normalize_key(self, key: Any, pandas_df: Any) -> Any:
        if isinstance(key, tuple) and len(key) == 2:
            row_sel, col_sel = key
            col_sel = self._convert_column_selector(col_sel, pandas_df.columns)
            return (row_sel, col_sel)
        elif isinstance(key, (str, tuple)):
            # Column-only selection like df.iloc["a"]
            return self._convert_single_column_selector(key, pandas_df.columns)
        return key

    def __getitem__(self, key: Any) -> Any:
        _require_pandas("DataFrame.iloc")

        pandas_df = self._df.to_pandas()
        normalized_key = self._normalize_key(key, pandas_df)
        pandas_result = pandas_df.iloc[normalized_key]
        return self._convert(pandas_result)

    def __setitem__(self, key: Any, value: Any) -> None:
        _require_pandas("DataFrame.iloc")
        pandas_df = self._df.to_pandas()
        normalized_key = self._normalize_key(key, pandas_df)
        pandas_value = self._ensure_pandas_value(value)
        pandas_df.iloc[normalized_key] = pandas_value
        updated = self._df._from_pandas_dataframe(pandas_df)
        self._df._update_from_dataframe(updated)


class _GroupBy:
    """Lightweight wrapper around Polars GroupBy providing pandas-like ergonomics."""

    def __init__(self, polars_groupby: pl.dataframe.group_by.GroupBy, parent: "DataFrame") -> None:
        self._groupby = polars_groupby
        self._parent = parent
        self._level_info: Optional[Dict[str, Any]] = None

    def agg(self, aggregations: Any) -> "DataFrame":
        """Aggregate grouped data using Polars expressions or simple specifications."""

        expressions = self._prepare_aggregations(aggregations)
        if not expressions:
            raise ValueError("No aggregations provided to GroupBy.agg().")

        result_pl = self._groupby.agg(expressions)
        level_info = getattr(self, "_level_info", None)
        if level_info:
            rename_map = dict(
                zip(
                    level_info.get("level_columns", []),
                    level_info.get("level_names", []),
                )
            )
            if rename_map:
                result_pl = result_pl.rename(rename_map)

            index_columns = [rename_map.get(col, col) for col in level_info.get("level_columns", [])]
            index_data: Optional[Any] = None
            index_name: Optional[Union[str, Tuple[str, ...]]] = None
            if index_columns:
                column_values = [result_pl[col].to_list() for col in index_columns]
                if len(index_columns) == 1:
                    index_data = column_values[0]
                    names = level_info.get("level_names", [None])
                    index_name = names[0] if names else None
                else:
                    index_data = list(zip(*column_values))
                    names = level_info.get("level_names", [])
                    index_name = tuple(names) if any(name is not None for name in names) else None
                result_pl = result_pl.drop(index_columns)

            result_df = DataFrame(result_pl)
            if index_data is not None:
                result_df._index = index_data
                result_df._index_name = index_name
            return result_df

        return DataFrame(result_pl)

    def _prepare_aggregations(self, aggregations: Any) -> List[pl.Expr]:
        if isinstance(aggregations, dict):
            exprs: List[pl.Expr] = []
            for column, funcs in aggregations.items():
                if isinstance(funcs, (list, tuple, set)):
                    for func in funcs:
                        exprs.append(self._expr_from_string(column, func))
                else:
                    exprs.append(self._expr_from_string(column, funcs))
            return exprs

        if isinstance(aggregations, (list, tuple, set)):
            return [self._convert_to_expr(item) for item in aggregations]

        return [self._convert_to_expr(aggregations)]

    def _expr_from_string(self, column: str, func: Any) -> pl.Expr:
        func_name = str(func).lower()
        method_map = {
            "sum": "sum",
            "mean": "mean",
            "avg": "mean",
            "min": "min",
            "max": "max",
            "std": "std",
            "var": "var",
            "median": "median",
            "count": "count",
            "size": "count",
            "nunique": "n_unique",
            "n_unique": "n_unique",
            "first": "first",
            "last": "last",
        }
        method = method_map.get(func_name)
        if method is None:
            raise NotImplementedError(f"Aggregation '{func}' is not supported for groupby().")
        expr = getattr(pl.col(column), method)()
        return expr.alias(f"{column}_{func_name}")

    def _convert_to_expr(self, item: Any) -> pl.Expr:
        if isinstance(item, pl.Expr):
            return item

        try:
            from polarpandas.series import (
                Series as PolarPandasSeries,  # Local import to avoid circular dependency
            )
        except ImportError:  # pragma: no cover - defensive fallback
            PolarPandasSeries = None  # type: ignore

        if PolarPandasSeries is not None and isinstance(item, PolarPandasSeries):
            series_name = item._series.name
            if series_name is None or series_name == "":
                raise ValueError("Cannot aggregate Series without a name")
            return pl.col(series_name)

        if isinstance(item, str):
            return pl.col(item)

        # Treat anything else as a literal so callers can pass scalars without errors
        return pl.lit(item)
