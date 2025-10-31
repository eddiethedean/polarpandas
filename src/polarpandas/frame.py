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
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import polars as pl

from polarpandas._exceptions import (
    convert_to_keyerror,
    create_keyerror_with_suggestions,
)
from polarpandas._index_manager import IndexManager
from polarpandas.index import Index
from polarpandas.utils import convert_schema_to_polars

if TYPE_CHECKING:
    from .lazyframe import LazyFrame
    from .series import Series


class DataFrame:
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

        return LazyFrame(self._df.lazy())

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
                    raise ValueError("No columns to parse from file") from e
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
        Read a JSON file into DataFrame.

        Parameters
        ----------
        path : str
            Path to JSON file
        dtype : dict, pl.Schema, or None, optional
            Schema specification for columns. Can be pandas-style dict or Polars schema.
            See DataFrame constructor for details.
        schema : dict, pl.Schema, or None, optional
            Direct Polars schema specification (alternative to dtype).
        **kwargs
            Additional arguments passed to Polars read_json()

        Returns
        -------
        DataFrame
            DataFrame loaded from JSON
        """
        # Map pandas-style parameters to Polars equivalents
        polars_kwargs: Dict[str, Any] = {}

        # Handle dtype/schema parameters
        dtype = kwargs.pop("dtype", None)
        schema = kwargs.pop("schema", None)

        # If both are provided, schema takes precedence
        schema_to_use = schema if schema is not None else dtype

        if schema_to_use is not None:
            polars_schema = convert_schema_to_polars(schema_to_use)
            if polars_schema is not None:
                polars_kwargs["schema"] = polars_schema

        # Use Polars JSON read - orient parameter support is limited
        # Remove pandas-specific parameters that Polars doesn't support
        polars_kwargs.update(
            {k: v for k, v in kwargs.items() if k not in ["orient", "lines"]}
        )

        try:
            df = pl.read_json(path, **polars_kwargs)
            return cls(df)
        except Exception as e:
            # If Polars JSON read fails, this is a limitation
            raise ValueError(
                f"Polars JSON read failed: {e}. Some JSON formats may not be supported."
            ) from e

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
            raise ValueError("Either 'mapping' or 'columns' must be provided")

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
            # Return the stored index
            return Index(self._index)
        else:
            # Create a simple RangeIndex-like object
            return Index(list(range(len(self._df))))

    @property
    def loc(self) -> "_LocIndexer":
        """Access a group of rows and columns by label(s)."""
        # For now, return a simple stub
        # Full implementation would return a LocIndexer object
        return _LocIndexer(self)

    @property
    def iloc(self) -> "_ILocIndexer":
        """Access a group of rows and columns by integer position(s)."""
        # For now, return a simple stub
        # Full implementation would return an ILocIndexer object
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

    def groupby(
        self, by: Union[str, List[str]], *args: Any, **kwargs: Any
    ) -> "_GroupBy":
        """
        Group DataFrame by one or more columns.

        Split the DataFrame into groups based on values in one or more columns.
        Returns a GroupBy object that can be used for aggregation operations.

        Parameters
        ----------
        by : str or list of str
            Column name(s) to group by. Rows with the same values in these
            columns will be grouped together.
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

        Notes
        -----
        - GroupBy operations in Polars use expressions (e.g., pl.col("x").mean())
          rather than string aggregation functions like pandas
        - The GroupBy object is lazy; aggregations are computed when called

        See Also
        --------
        _GroupBy : The GroupBy object returned by this method
        """
        # Polars uses group_by() instead of groupby()
        # Return a wrapper for the Polars GroupBy object
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
        if isinstance(other, DataFrame):
            other_df = other._df
        else:
            # This branch is technically unreachable due to type annotation
            # but kept for defensive programming
            other_df = other  # type: ignore[unreachable]

        # Handle both DataFrame and LazyFrame cases
        if hasattr(other_df, "_df"):
            # Convert polarpandas DataFrame to LazyFrame
            other_polars = other_df._df
            if isinstance(other_polars, pl.DataFrame):
                other_lazy = other_polars.lazy()
            elif isinstance(other_polars, pl.LazyFrame):
                other_lazy = other_polars
            else:
                # Handle PyDataFrame or other types
                # PyDataFrame is the underlying Polars DataFrame, convert it properly
                try:
                    # Try to get the DataFrame and convert to LazyFrame
                    if hasattr(other_polars, "collect"):
                        # It's a LazyFrame that needs collecting first, then convert
                        other_lazy = other_polars.collect().lazy()
                    else:
                        # Assume it's a PyDataFrame, wrap in DataFrame then lazy
                        other_lazy = pl.DataFrame._from_pydf(other_polars).lazy()
                except (AttributeError, TypeError):
                    # Fallback: try to convert via pandas
                    try:
                        import pandas as pd

                        other_lazy = pl.from_pandas(
                            pd.DataFrame(other_polars.to_dict())
                        ).lazy()
                    except ImportError as e:
                        raise ImportError(
                            "pandas is required for merging with unsupported DataFrame types. "
                            "Install with: pip install pandas"
                        ) from e
        elif isinstance(other_df, pl.DataFrame):
            # Convert Polars DataFrame to LazyFrame
            other_lazy = other_df.lazy()
        # Note: other_df should be pl.DataFrame after extraction from polarpandas DataFrame
        elif hasattr(other_df, "lazy"):  # type: ignore[unreachable]
            other_lazy = other_df.lazy()
        else:
            other_lazy = other_df

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
            Whether to drop the index or add it as a column
        inplace : bool, default False
            If True, modify DataFrame in place

        Returns
        -------
        DataFrame or None
            DataFrame with reset index, or None if inplace=True
        """
        # For simple range indices, this is mostly a no-op
        # In a full implementation, this would handle custom indices
        if not drop:
            # Add index as a column
            result_df = self._df.with_row_index("index")
        else:
            result_df = self._df.clone()

        if inplace:
            self._df = result_df
            return None
        else:
            return DataFrame(result_df)

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
            raise ValueError("Must pass non-zero number of levels/codes")

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

    def transpose(self) -> "DataFrame":
        """
        Transpose index and columns using pure Polars.

        Returns
        -------
        DataFrame
            Transposed DataFrame
        """
        # Handle empty DataFrame
        if len(self._df.columns) == 0:
            return DataFrame()

        # Use Polars transpose with column names from index if available
        column_names = self._index if self._index else None

        try:
            transposed = self._df.transpose(
                include_header=False, column_names=column_names
            )
            result = DataFrame(transposed)

            # Rename columns to match pandas (0, 1, 2, ...)
            num_cols = len(transposed.columns)
            new_columns = [str(i) for i in range(num_cols)]
            result._df = result._df.rename(dict(zip(result._df.columns, new_columns)))

            # Set index from original columns
            result._index = list(self._df.columns)
            result._index_name = None

            return result
        except Exception as e:
            # If Polars transpose fails, this is a limitation
            raise ValueError(
                f"Polars transpose failed: {e}. This may be due to mixed data types."
            ) from e

    @property
    def T(self) -> "DataFrame":
        """
        Transpose index and columns.

        Returns
        -------
        DataFrame
            Transposed DataFrame
        """
        return self.transpose()

    def to_csv(self, path: Optional[str] = None, **kwargs: Any) -> Optional[str]:
        """
        Write DataFrame to CSV file.

        Parameters
        ----------
        path : str, optional
            File path. If None, return string
        **kwargs
            Additional arguments passed to Polars write_csv()

        Returns
        -------
        str or None
            CSV string if path is None, otherwise None
        """
        # Map pandas-style parameters to Polars equivalents
        polars_kwargs = {}

        # Handle pandas-specific parameters
        index_param = kwargs.get("index", True)  # Default to True like pandas
        if "index" in kwargs:
            index_param = kwargs.pop("index")

        # Map pandas parameters to Polars
        if "sep" in kwargs:
            polars_kwargs["separator"] = kwargs.pop("sep")

        if "header" in kwargs:
            header = kwargs.pop("header")
            if isinstance(header, list):
                # Polars doesn't support custom header names, so we need to temporarily rename columns
                original_columns = self._df.columns
                if len(header) != len(original_columns):
                    raise ValueError(
                        f"Header length ({len(header)}) must match number of columns ({len(original_columns)})"
                    )

                # Create a temporary DataFrame with renamed columns
                temp_df = self._df.rename(dict(zip(original_columns, header)))

                # Write the temporary DataFrame
                if path is None:
                    return temp_df.write_csv(**polars_kwargs)  # type: ignore[no-any-return]
                else:
                    temp_df.write_csv(path, **polars_kwargs)
                    return None
            else:
                polars_kwargs["include_header"] = header

        # Pass through other parameters
        polars_kwargs.update(kwargs)

        # If index=False, use Polars write_csv directly
        if not index_param:
            if path is None:
                return self._df.write_csv(**polars_kwargs)  # type: ignore[no-any-return]
            else:
                self._df.write_csv(path, **polars_kwargs)
                return None

        # Handle index=True case - add index as first column
        else:  # index_param is True
            # Create a copy with index as a column
            df_to_write = self._df.clone()

            # Add index column if we have one
            if self._index is not None:
                index_name = (
                    self._index_name if self._index_name is not None else "index"
                )
                # Ensure index_name is a string (not tuple)
                if isinstance(index_name, tuple):
                    index_name = "_".join(str(n) for n in index_name)

                # Add index as first column
                df_to_write = df_to_write.with_columns(
                    pl.Series(index_name, self._index)
                ).select([index_name] + df_to_write.columns)
            else:
                # No stored index - use integer index
                df_to_write = df_to_write.with_row_index("index").select(
                    ["index"] + df_to_write.columns
                )

            # Write with index column included
            if path is None:
                return df_to_write.write_csv(**polars_kwargs)  # type: ignore[no-any-return]
            else:
                df_to_write.write_csv(path, **polars_kwargs)
                return None

    def to_parquet(self, path: str, **kwargs: Any) -> None:
        """
        Write DataFrame to Parquet file.

        Parameters
        ----------
        path : str
            File path
        **kwargs
            Additional arguments passed to Polars write_parquet()
        """
        self._df.write_parquet(path, **kwargs)

    def to_json(self, path: Optional[str] = None, **kwargs: Any) -> Optional[str]:
        """
        Write DataFrame to JSON.

        Parameters
        ----------
        path : str, optional
            File path. If None, return string
        **kwargs
            Additional arguments passed to Polars write_json()

        Returns
        -------
        str or None
            JSON string if path is None, otherwise None
        """
        # Use Polars JSON write - orient parameter support is limited
        # Remove pandas-specific parameters that Polars doesn't support
        polars_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["orient", "lines"]
        }

        try:
            if path is None:
                return self._df.write_json()
            else:
                self._df.write_json(path, **polars_kwargs)
                return None
        except Exception as e:
            # If Polars JSON write fails, this is a limitation
            raise ValueError(
                f"Polars JSON write failed: {e}. Some JSON formats may not be supported."
            ) from e

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
            pandas_df.index = self._index
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

        # Convert string column names that look like integers to RangeIndex
        try:
            # Check if all column names are string representations of consecutive integers starting from 0
            col_names = list(pandas_df.columns)
            if all(isinstance(name, str) and name.isdigit() for name in col_names):
                int_cols = [int(name) for name in col_names]
                if int_cols == list(range(len(int_cols))):
                    # Convert to RangeIndex
                    pandas_df.columns = pd.RangeIndex(
                        start=0, stop=len(int_cols), step=1
                    )
        except Exception:
            # If conversion fails, keep original column names
            pass

        return pandas_df

    def to_sql(self, name: str, con: Any, **kwargs: Any) -> None:
        """
        Write DataFrame to SQL database.

        Parameters
        ----------
        name : str
            Table name
        con : connection object
            Database connection
        **kwargs
            Additional arguments passed to Polars write_database()

        Examples
        --------
        >>> df.to_sql("table", connection)
        """
        self._df.write_database(name, con, **kwargs)

    def to_feather(self, path: str, **kwargs: Any) -> None:
        """
        Write DataFrame to Feather file.

        Parameters
        ----------
        path : str
            Path to Feather file
        **kwargs
            Additional arguments passed to Polars write_ipc()

        Examples
        --------
        >>> df.to_feather("data.feather")
        """
        self._df.write_ipc(path, **kwargs)

    def sample(
        self, n: Optional[int] = None, frac: Optional[float] = None, **kwargs: Any
    ) -> "DataFrame":
        """
        Return a random sample of items.

        Parameters
        ----------
        n : int, optional
            Number of items to return
        frac : float, optional
            Fraction of items to return
        **kwargs
            Additional arguments passed to Polars sample()

        Returns
        -------
        DataFrame
            Random sample
        """
        if frac is not None:
            n = int(len(self) * frac)

        return DataFrame(self._df.sample(n=n, **kwargs))

    def pivot(
        self,
        index: Optional[Union[str, List[str]]] = None,
        columns: Optional[Union[str, List[str]]] = None,
        values: Optional[Union[str, List[str]]] = None,
    ) -> "DataFrame":
        """
        Pivot table operation.

        Parameters
        ----------
        index : str or list
            Column(s) to use as index
        columns : str
            Column to use for columns
        values : str
            Column to use for values

        Returns
        -------
        DataFrame
            Pivoted DataFrame
        """
        # Polars uses pivot() but with different parameter names
        return DataFrame(self._df.pivot(on=columns, index=index, values=values))  # type: ignore[arg-type]

    def pivot_table(
        self,
        values: str,
        index: str,
        columns: str,
        aggfunc: str = "mean",
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Create a pivot table.

        Parameters
        ----------
        values : str
            Column to aggregate
        index : str
            Column to use as index
        columns : str
            Column to use as columns
        aggfunc : str, default "mean"
            Aggregation function
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            Pivot table

        Examples
        --------
        >>> df = ppd.DataFrame({
        ...     "A": ["foo", "foo", "bar", "bar"],
        ...     "B": ["one", "two", "one", "two"],
        ...     "C": [1, 2, 3, 4]
        ... })
        >>> result = df.pivot_table(values="C", index="A", columns="B")
        """
        # Use the existing pivot method
        return self.pivot(index=index, columns=columns, values=values)

    def get_dummies(self, **kwargs: Any) -> "DataFrame":
        """
        Convert categorical variables into dummy/indicator variables.

        Parameters
        ----------
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            DataFrame with dummy variables

        Examples
        --------
        >>> df = ppd.DataFrame({"category": ["A", "B", "A"]})
        >>> result = df.get_dummies()
        """
        # Use Polars to_dummies() method
        return DataFrame(self._df.to_dummies(**kwargs))

    def rolling(self, window: int, **kwargs: Any) -> "_RollingGroupBy":
        """
        Provide rolling window calculations.

        Parameters
        ----------
        window : int
            Size of the rolling window
        **kwargs
            Additional arguments

        Returns
        -------
        _RollingGroupBy
            Rolling window object
        """
        return _RollingGroupBy(self, window, **kwargs)

    def rolling_mean(self, window: int, **kwargs: Any) -> "DataFrame":
        """
        Calculate rolling mean.

        Parameters
        ----------
        window : int
            Size of the rolling window
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            DataFrame with rolling mean values
        """
        # Apply rolling mean to each column
        columns = self._df.columns
        rolling_exprs = [pl.col(col).rolling_mean(window, **kwargs) for col in columns]
        result_df = self._df.with_columns(rolling_exprs)
        return DataFrame(result_df)

    def rolling_sum(self, window: int, **kwargs: Any) -> "DataFrame":
        """
        Calculate rolling sum.

        Parameters
        ----------
        window : int
            Size of the rolling window
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            DataFrame with rolling sum values
        """
        # Use lazy operations to maintain lazy state
        # Apply rolling sum to each column
        columns = self._df.columns
        rolling_exprs = [pl.col(col).rolling_sum(window, **kwargs) for col in columns]
        result_df = self._df.with_columns(rolling_exprs)
        return DataFrame(result_df)

    def rolling_std(self, window: int, **kwargs: Any) -> "DataFrame":
        """
        Calculate rolling standard deviation.

        Parameters
        ----------
        window : int
            Size of the rolling window
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            DataFrame with rolling standard deviation values
        """
        # Use lazy operations to maintain lazy state
        # Apply rolling std to each column
        columns = self._df.columns
        rolling_exprs = [pl.col(col).rolling_std(window, **kwargs) for col in columns]
        result_df = self._df.with_columns(rolling_exprs)
        return DataFrame(result_df)

    def rolling_max(self, window: int, **kwargs: Any) -> "DataFrame":
        """
        Calculate rolling maximum.

        Parameters
        ----------
        window : int
            Size of the rolling window
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            DataFrame with rolling maximum values
        """
        # Use lazy operations to maintain lazy state
        # Apply rolling max to each column
        columns = self._df.columns
        rolling_exprs = [pl.col(col).rolling_max(window, **kwargs) for col in columns]
        result_df = self._df.with_columns(rolling_exprs)
        return DataFrame(result_df)

    def rolling_min(self, window: int, **kwargs: Any) -> "DataFrame":
        """
        Calculate rolling minimum.

        Parameters
        ----------
        window : int
            Size of the rolling window
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            DataFrame with rolling minimum values
        """
        # Use lazy operations to maintain lazy state
        # Apply rolling min to each column
        columns = self._df.columns
        rolling_exprs = [pl.col(col).rolling_min(window, **kwargs) for col in columns]
        result_df = self._df.with_columns(rolling_exprs)
        return DataFrame(result_df)

    def group_by(self, *by: Union[str, List[str]], **kwargs: Any) -> Any:
        """
        Group DataFrame by one or more columns.

        Parameters
        ----------
        *by : str or list of str
            Column names to group by
        **kwargs
            Additional arguments

        Returns
        -------
        GroupBy object
            Grouped DataFrame object
        """
        # Group by columns
        grouped = self._df.group_by(*by, **kwargs)
        # Return the grouped object directly - it will be wrapped when methods are called
        return grouped

    def sort(self, by: Union[str, List[str]], **kwargs: Any) -> "DataFrame":
        """
        Sort DataFrame by one or more columns.

        Parameters
        ----------
        by : str or list of str
            Column names to sort by
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            Sorted DataFrame
        """
        sorted_df = self._df.sort(by, **kwargs)
        return DataFrame(sorted_df)

    def apply(
        self, func: Callable[..., Any], axis: int = 0
    ) -> Union["Series", "DataFrame"]:
        """
        Apply a function along an axis.

        Parameters
        ----------
        func : function
            Function to apply
        axis : {0, 1}, default 0
            0 for columns, 1 for rows

        Returns
        -------
        Series or DataFrame
            Result of applying function
        """
        from polarpandas.series import Series

        if axis == 0:
            # Apply to each column
            results = {}
            for col in self.columns:
                result = func(self._df[col])
                results[col] = result
            return Series(list(results.values()), name="apply_result")
        else:
            # Apply to each row - more complex
            raise NotImplementedError("apply() with axis=1 not yet implemented")

    def applymap(self, func: Callable[..., Any]) -> "DataFrame":
        """
        Apply a function element-wise.

        Parameters
        ----------
        func : function
            Function to apply to each element

        Returns
        -------
        DataFrame
            DataFrame with function applied
        """
        # Apply function to each column
        result_cols = []
        for col in self.columns:
            result_cols.append(
                self._df[col].map_elements(func, return_dtype=pl.Float64).alias(col)
            )

        return DataFrame(self._df.select(result_cols))

    @staticmethod
    def concat(dfs: List[Any], axis: int = 0, **kwargs: Any) -> "DataFrame":
        """
        Concatenate DataFrames.

        Parameters
        ----------
        dfs : list of DataFrame
            DataFrames to concatenate
        axis : {0, 1}, default 0
            0 for vertical, 1 for horizontal

        Returns
        -------
        DataFrame
            Concatenated DataFrame
        """
        # Extract underlying Polars DataFrames
        pl_dfs = [df._df if isinstance(df, DataFrame) else df for df in dfs]

        if axis == 0:
            # Vertical concatenation
            result = pl.concat(pl_dfs, how="vertical", **kwargs)
        else:
            # Horizontal concatenation
            result = pl.concat(pl_dfs, how="horizontal", **kwargs)

        return DataFrame(result)

    def nlargest(
        self,
        n: int,
        columns: Union[str, List[str]],
        keep: Literal["first", "last", "all"] = "first",
    ) -> "DataFrame":
        """
        Return the first n rows ordered by columns in descending order.

        Parameters
        ----------
        n : int
            Number of rows to return
        columns : str or list of str
            Column name(s) to order by
        keep : {'first', 'last', 'all'}, default 'first'
            When there are duplicate values:
            - 'first' : keep the first occurrence
            - 'last' : keep the last occurrence
            - 'all' : keep all occurrences

        Returns
        -------
        DataFrame
            The n largest rows
        """
        # Handle empty DataFrame
        if self._df.height == 0:
            raise KeyError(
                f"Column '{columns[0] if isinstance(columns, str) else columns[0]}' not found"
            )

        # Use Polars for nlargest operation with index preservation
        if isinstance(columns, str):
            columns = [columns]

        # Store original indices before sorting
        if self._index is not None:
            # Add row count to track original positions
            temp_df = self._df.with_row_index("__temp_idx__")
            sorted_df = temp_df.sort(by=columns, descending=True).head(n)

            # Extract original indices
            original_indices = sorted_df["__temp_idx__"].to_list()
            result_indices = [self._index[i] for i in original_indices]

            # Remove temporary column and create result
            result_df = sorted_df.drop("__temp_idx__")
            result = DataFrame(result_df)
            result._index = result_indices
        else:
            # No stored index, but preserve original row positions
            temp_df = self._df.with_row_index("__temp_idx__")
            sorted_df = temp_df.sort(by=columns, descending=True).head(n)

            # Extract original row positions
            original_indices = sorted_df["__temp_idx__"].to_list()

            # Remove temporary column and create result
            result_df = sorted_df.drop("__temp_idx__")
            result = DataFrame(result_df)
            result._index = original_indices
        return result

    def nsmallest(
        self,
        n: int,
        columns: Union[str, List[str]],
        keep: Literal["first", "last", "all"] = "first",
    ) -> "DataFrame":
        """
        Return the first n rows ordered by columns in ascending order.

        Parameters
        ----------
        n : int
            Number of rows to return
        columns : str or list of str
            Column name(s) to order by
        keep : {'first', 'last', 'all'}, default 'first'
            When there are duplicate values:
            - 'first' : keep the first occurrence
            - 'last' : keep the last occurrence
            - 'all' : keep all occurrences

        Returns
        -------
        DataFrame
            The n smallest rows
        """
        # Handle empty DataFrame
        if self._df.height == 0:
            raise KeyError(
                f"Column '{columns[0] if isinstance(columns, str) else columns[0]}' not found"
            )

        # Use Polars for nsmallest operation with index preservation
        if isinstance(columns, str):
            columns = [columns]

        # Store original indices before sorting
        if self._index is not None:
            # Add row count to track original positions
            temp_df = self._df.with_row_index("__temp_idx__")
            sorted_df = temp_df.sort(by=columns, descending=False).head(n)

            # Extract original indices
            original_indices = sorted_df["__temp_idx__"].to_list()
            result_indices = [self._index[i] for i in original_indices]

            # Remove temporary column and create result
            result_df = sorted_df.drop("__temp_idx__")
            result = DataFrame(result_df)
            result._index = result_indices
        else:
            # No stored index, but preserve original row positions
            temp_df = self._df.with_row_index("__temp_idx__")
            sorted_df = temp_df.sort(by=columns, descending=False).head(n)

            # Extract original row positions
            original_indices = sorted_df["__temp_idx__"].to_list()

            # Remove temporary column and create result
            result_df = sorted_df.drop("__temp_idx__")
            result = DataFrame(result_df)
            result._index = original_indices
        return result

    def corr(self, method: str = "pearson", min_periods: int = 1) -> "DataFrame":
        """
        Compute pairwise correlation of columns.

        Parameters
        ----------
        method : {'pearson', 'kendall', 'spearman'}, default 'pearson'
            Correlation method
        min_periods : int, default 1
            Minimum number of observations required per pair of columns

        Returns
        -------
        DataFrame
            Correlation matrix
        """
        # Polars doesn't have a direct corr method, so we'll use a workaround
        # For now, return a simple implementation
        # This is a limitation - Polars doesn't have built-in correlation
        raise NotImplementedError(
            "Polars doesn't have built-in correlation. This is a known limitation."
        )

    def cov(self, min_periods: Optional[int] = None) -> "DataFrame":
        """
        Compute pairwise covariance of columns.

        Parameters
        ----------
        min_periods : int, optional
            Minimum number of observations required per pair of columns

        Returns
        -------
        DataFrame
            Covariance matrix
        """
        # This is a limitation - Polars doesn't have built-in covariance
        raise NotImplementedError(
            "Polars doesn't have built-in covariance. This is a known limitation."
        )

    def rank(
        self,
        axis: int = 0,
        method: str = "average",
        numeric_only: bool = False,
        na_option: str = "keep",
        ascending: bool = True,
        pct: bool = False,
    ) -> "DataFrame":
        """
        Compute numerical data ranks along axis.

        Parameters
        ----------
        axis : {0, 1}, default 0
            Axis to rank along
        method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
            How to rank the group of records
        numeric_only : bool, default False
            Include only numeric columns
        na_option : {'keep', 'top', 'bottom'}, default 'keep'
            How to rank NaN values
        ascending : bool, default True
            Whether or not the elements should be ranked in ascending order
        pct : bool, default False
            Whether to display the returned rankings in percentile form

        Returns
        -------
        DataFrame
            DataFrame with ranks
        """
        if axis == 1:
            raise NotImplementedError("rank with axis=1 not yet implemented")

        # Map pandas methods to Polars methods
        from typing import Literal, cast

        method_map = {
            "average": "average",
            "min": "min",
            "max": "max",
            "first": "ordinal",  # Polars uses 'ordinal' for first occurrence
            "dense": "dense",
        }
        polars_method_str = method_map.get(method, method)
        # Cast to Literal type expected by Polars
        polars_method: Literal[
            "average", "min", "max", "dense", "ordinal", "random"
        ] = cast(
            Literal["average", "min", "max", "dense", "ordinal", "random"],
            polars_method_str,
        )

        # Apply rank to each column
        result_cols = []
        for col in self._df.columns:
            if numeric_only and not self._df[col].dtype.is_numeric():
                # Skip non-numeric columns when numeric_only=True
                continue
            else:
                rank_expr = pl.col(col).rank(
                    method=polars_method,
                    descending=not ascending,
                )
                if pct:
                    rank_expr = rank_expr / pl.len()
                # Cast to float64 to match pandas dtype
                rank_expr = rank_expr.cast(pl.Float64)
                result_cols.append(rank_expr.alias(col))

        result_df = self._df.select(result_cols)
        return DataFrame(result_df)

    def diff(self, periods: int = 1) -> "DataFrame":
        """
        First discrete difference of element.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference

        Returns
        -------
        DataFrame
            DataFrame with differences
        """
        result_cols = []
        for col in self._df.columns:
            if self._df[col].dtype.is_numeric():
                result_cols.append(pl.col(col).diff(periods).alias(col))
            else:
                result_cols.append(pl.col(col))

        result_df = self._df.select(result_cols)
        return DataFrame(result_df)

    def pct_change(
        self,
        periods: int = 1,
        fill_method: str = "pad",
        limit: Optional[int] = None,
        freq: Optional[str] = None,
    ) -> "DataFrame":
        """
        Percentage change between the current and a prior element.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for forming percent change
        fill_method : str, default 'pad'
            How to handle NAs before computing percent changes
        limit : int, optional
            The number of consecutive NAs to fill before stopping
        freq : str, optional
            Increment to use from time series API

        Returns
        -------
        DataFrame
            DataFrame with percentage changes
        """
        result_cols = []
        for col in self._df.columns:
            if self._df[col].dtype.is_numeric():
                # Calculate percentage change
                pct_change = (pl.col(col) - pl.col(col).shift(periods)) / pl.col(
                    col
                ).shift(periods)
                result_cols.append(pct_change.alias(col))
            else:
                result_cols.append(pl.col(col))

        result_df = self._df.select(result_cols)
        return DataFrame(result_df)

    def cumsum(self, axis: Optional[int] = None, skipna: bool = True) -> "DataFrame":
        """
        Return cumulative sum over a DataFrame axis.

        Parameters
        ----------
        axis : {0, 1, None}, default None
            Axis along which the cumulative sum is computed
        skipna : bool, default True
            Exclude NA/null values

        Returns
        -------
        DataFrame
            DataFrame with cumulative sums
        """
        if axis == 1:
            raise NotImplementedError("cumsum with axis=1 not yet implemented")

        result_cols = []
        for col in self._df.columns:
            if self._df[col].dtype.is_numeric():
                result_cols.append(pl.col(col).cum_sum().alias(col))
            elif self._df[col].dtype == pl.Boolean:
                # Cast boolean cumsum to int64 to match pandas behavior
                result_cols.append(pl.col(col).cum_sum().cast(pl.Int64).alias(col))
            else:
                result_cols.append(pl.col(col))

        result_df = self._df.select(result_cols)
        return DataFrame(result_df)

    def cumprod(self, axis: Optional[int] = None, skipna: bool = True) -> "DataFrame":
        """
        Return cumulative product over a DataFrame axis.

        Parameters
        ----------
        axis : {0, 1, None}, default None
            Axis along which the cumulative product is computed
        skipna : bool, default True
            Exclude NA/null values

        Returns
        -------
        DataFrame
            DataFrame with cumulative products
        """
        if axis == 1:
            raise NotImplementedError("cumprod with axis=1 not yet implemented")

        result_cols = []
        for col in self._df.columns:
            if self._df[col].dtype.is_numeric():
                result_cols.append(pl.col(col).cum_prod().alias(col))
            else:
                result_cols.append(pl.col(col))

        result_df = self._df.select(result_cols)
        return DataFrame(result_df)

    def cummax(self, axis: Optional[int] = None, skipna: bool = True) -> "DataFrame":
        """
        Return cumulative maximum over a DataFrame axis.

        Parameters
        ----------
        axis : {0, 1, None}, default None
            Axis along which the cumulative maximum is computed
        skipna : bool, default True
            Exclude NA/null values

        Returns
        -------
        DataFrame
            DataFrame with cumulative maximums
        """
        if axis == 1:
            raise NotImplementedError("cummax with axis=1 not yet implemented")

        result_cols = []
        for col in self._df.columns:
            if self._df[col].dtype.is_numeric():
                result_cols.append(pl.col(col).cum_max().alias(col))
            else:
                result_cols.append(pl.col(col))

        result_df = self._df.select(result_cols)
        return DataFrame(result_df)

    def cummin(self, axis: Optional[int] = None, skipna: bool = True) -> "DataFrame":
        """
        Return cumulative minimum over a DataFrame axis.

        Parameters
        ----------
        axis : {0, 1, None}, default None
            Axis along which the cumulative minimum is computed
        skipna : bool, default True
            Exclude NA/null values

        Returns
        -------
        DataFrame
            DataFrame with cumulative minimums
        """
        if axis == 1:
            raise NotImplementedError("cummin with axis=1 not yet implemented")

        result_cols = []
        for col in self._df.columns:
            if self._df[col].dtype.is_numeric():
                result_cols.append(pl.col(col).cum_min().alias(col))
            else:
                result_cols.append(pl.col(col))

        result_df = self._df.select(result_cols)
        return DataFrame(result_df)


class _LocIndexer:
    """Label-based indexer for DataFrame."""

    def __init__(self, df: DataFrame):
        self._df = df

    def __getitem__(self, key: Any) -> Union["Series", "DataFrame", Any]:
        """Get items by label."""
        if isinstance(key, tuple):
            # Row and column indexing: df.loc[row, col]
            row_key, col_key = key
            return self._get_rows_cols(row_key, col_key)
        else:
            # Row-only indexing: df.loc[row]
            return self._get_rows(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set items by label."""
        if isinstance(key, tuple):
            # Row and column indexing: df.loc[row, col] = value
            row_key, col_key = key
            self._set_rows_cols(row_key, col_key, value)
        else:
            # Row-only indexing: df.loc[row] = value
            self._set_rows(key, value)

    def _get_rows(self, row_key: Any) -> Union["Series", "DataFrame"]:
        """Get rows by label."""
        # Get the Polars DataFrame from the parent DataFrame
        polars_df = self._df._df

        # Handle boolean indexing with pandas Series
        if hasattr(row_key, "dtype") and str(row_key.dtype) == "bool":
            # Convert pandas Series mask to Polars expression
            import polars as pl

            mask_values = row_key.tolist()
            mask_series = pl.Series("mask", mask_values)
            selected_df = polars_df.filter(mask_series)
            result = DataFrame(selected_df)
            # Preserve index for selected rows
            if self._df._index is not None:
                selected_indices = [i for i, val in enumerate(mask_values) if val]
                result._index = [self._df._index[i] for i in selected_indices]
            else:
                # No stored index, but we need to preserve the original row positions
                selected_indices = [i for i, val in enumerate(mask_values) if val]
                result._index = selected_indices
            return result

        # Use Polars for row selection - limited label-based support
        if self._df._index is not None:
            # Find row index by label
            try:
                if isinstance(row_key, slice):
                    # Handle slice with labels
                    start_idx = (
                        self._df._index.index(row_key.start)
                        if row_key.start is not None
                        else 0
                    )
                    stop_idx = (
                        self._df._index.index(row_key.stop)
                        if row_key.stop is not None
                        else len(self._df._index)
                    )
                    row_indices = list(range(start_idx, stop_idx))
                elif isinstance(row_key, list):
                    # Handle list of labels
                    row_indices = [self._df._index.index(label) for label in row_key]
                else:
                    # Single label
                    row_indices = [self._df._index.index(row_key)]

                # Select rows by integer indices
                if len(row_indices) == 1:
                    # Single row - return as Series
                    from polarpandas.series import Series

                    return Series(polars_df[row_indices[0]])  # type: ignore[arg-type]
                else:
                    # Multiple rows - return as DataFrame
                    selected_df = polars_df[row_indices]
                    result = DataFrame(selected_df)
                    # Preserve index for selected rows
                    result._index = [self._df._index[i] for i in row_indices]
                    return result
            except ValueError as e:
                raise KeyError(f"'{row_key}' not in index") from e
        else:
            # No index - treat as integer position
            if isinstance(row_key, slice):
                try:
                    selected_df = polars_df[row_key]
                    return DataFrame(selected_df)
                except IndexError as e:
                    # Convert Polars IndexError to pandas KeyError for compatibility
                    raise KeyError(f"index {row_key} is out of bounds") from e
            elif isinstance(row_key, list):
                try:
                    selected_df = polars_df[row_key]
                    return DataFrame(selected_df)
                except IndexError as e:
                    # Convert Polars IndexError to pandas KeyError for compatibility
                    raise KeyError(f"index {row_key} is out of bounds") from e
            else:
                # Single row - return as Series
                from polarpandas.series import Series

                try:
                    # Get single row as Series - use slice to get all columns
                    row_data = polars_df.slice(row_key, 1)
                    # Convert to Series by taking the first (and only) row
                    # Create a list of values in column order
                    values = [row_data[col][0] for col in row_data.columns]
                    return Series(values, index=row_data.columns, strict=False)
                except IndexError as e:
                    # Convert Polars IndexError to pandas KeyError for compatibility
                    raise KeyError(f"index {row_key} is out of bounds") from e

    def _get_rows_cols(
        self, row_key: Any, col_key: Any
    ) -> Union["Series", "DataFrame", Any]:
        """Get rows and columns by label."""
        # Get the Polars DataFrame from the parent DataFrame
        polars_df = self._df._df
        import polars as pl

        # Handle boolean indexing with pandas Series
        if hasattr(row_key, "dtype") and str(row_key.dtype) == "bool":
            # Convert pandas Series mask to Polars expression
            mask_values = row_key.tolist()
            mask_series = pl.Series("mask", mask_values)
            selected_df = polars_df.filter(mask_series)

            # Select columns if specified
            if col_key is not None:
                if isinstance(col_key, str):
                    # Single column - return as Series
                    from polarpandas.series import Series

                    return Series(selected_df[col_key])
                else:
                    # Multiple columns - return as DataFrame
                    selected_df = selected_df[col_key]

            result = DataFrame(selected_df)
            # Preserve index for selected rows
            if self._df._index is not None:
                selected_indices = [i for i, val in enumerate(mask_values) if val]
                result._index = [self._df._index[i] for i in selected_indices]
            else:
                # No stored index, but we need to preserve the original row positions
                selected_indices = [i for i, val in enumerate(mask_values) if val]
                result._index = selected_indices
            return result

        # Use Polars for row/column selection - limited label-based support
        if self._df._index is not None:
            # Find row index by label
            try:
                if isinstance(row_key, slice):
                    # Handle slice with labels
                    start_idx = (
                        self._df._index.index(row_key.start)
                        if row_key.start is not None
                        else 0
                    )
                    stop_idx = (
                        self._df._index.index(row_key.stop)
                        if row_key.stop is not None
                        else len(self._df._index)
                    )
                    row_indices = list(range(start_idx, stop_idx))
                elif isinstance(row_key, list):
                    # Handle list of labels
                    row_indices = [self._df._index.index(label) for label in row_key]
                else:
                    # Single label
                    row_indices = [self._df._index.index(row_key)]

                # Select rows and columns
                if len(row_indices) == 1 and isinstance(col_key, str):
                    # Single cell access - return scalar value directly
                    row_values = polars_df.row(row_indices[0], named=True)
                    return row_values[col_key]
                elif len(row_indices) == 1:
                    # Single row, multiple columns - return as Series
                    from polarpandas.series import Series

                    return Series(polars_df[row_indices[0], col_key])
                else:
                    # Multiple rows - return as DataFrame
                    selected_df = polars_df[row_indices, col_key]
                    result = DataFrame(selected_df)
                    # Preserve index for selected rows
                    result._index = [self._df._index[i] for i in row_indices]
                    return result
            except ValueError as e:
                raise KeyError(f"'{row_key}' not in index") from e
        else:
            # No index - treat as integer position
            if isinstance(row_key, slice):
                selected_df = polars_df[row_key, col_key]
                return DataFrame(selected_df)
            elif isinstance(row_key, list):
                selected_df = polars_df[row_key, col_key]
                return DataFrame(selected_df)
            else:
                # Single cell access - return scalar value directly
                if isinstance(col_key, str):
                    try:
                        # Use Polars row() method for single row access
                        row_values = polars_df.row(row_key, named=True)
                        return row_values[col_key]
                    except (IndexError, pl.exceptions.OutOfBoundsError) as e:
                        raise KeyError(f"Key {row_key} or {col_key} not found") from e
                else:
                    # Single row, multiple columns - return as Series
                    from polarpandas.series import Series

                    try:
                        row_data = polars_df.row(row_key, named=True)
                        if isinstance(col_key, list):
                            return Series([row_data[k] for k in col_key])
                        else:
                            return Series([row_data[col_key]])
                    except (IndexError, pl.exceptions.OutOfBoundsError) as e:
                        raise KeyError(f"Key {row_key} or {col_key} not found") from e

    def _set_rows(self, row_key: Any, value: Any) -> None:
        """Set rows by label."""
        polars_df = self._df._df
        import polars as pl

        from polarpandas._exceptions import create_keyerror_with_suggestions

        # Convert label row_key to integer position(s)
        if self._df._index is not None:
            # We have an index - convert labels to positions
            # Handle pandas fallback: if integer not in index and index is not integer-based, use position
            is_integer_index = (
                all(
                    isinstance(k, (int, type(None)))
                    for k in self._df._index[:10]
                    if k is not None
                )
                if self._df._index
                else False
            )

            if isinstance(row_key, slice):
                # Slice of labels - convert to list of positions
                start = row_key.start
                stop = row_key.stop
                step = row_key.step if row_key.step is not None else 1

                # Find start position
                if start is None:
                    start_pos = 0
                else:
                    try:
                        start_pos = self._df._index.index(start)
                    except ValueError as err:
                        # Pandas fallback for integer slices
                        if isinstance(start, int) and not is_integer_index:
                            start_pos = start if start >= 0 else len(polars_df) + start
                        else:
                            raise create_keyerror_with_suggestions(
                                str(start),
                                [str(k) for k in self._df._index],
                                context="index",
                            ) from err

                # Find stop position
                if stop is None:
                    stop_pos = len(self._df._index)
                else:
                    try:
                        stop_pos = (
                            self._df._index.index(stop) + 1
                        )  # +1 because slice is exclusive
                    except ValueError as err:
                        # Pandas fallback for integer slices
                        if isinstance(stop, int) and not is_integer_index:
                            stop_pos = (
                                stop if stop >= 0 else len(polars_df) + stop
                            ) + 1
                        else:
                            raise create_keyerror_with_suggestions(
                                str(stop),
                                [str(k) for k in self._df._index],
                                context="index",
                            ) from err

                row_indices = list(range(start_pos, stop_pos, step))
            elif isinstance(row_key, list):
                # List of labels - convert each to position
                row_indices = []
                for label in row_key:
                    try:
                        row_indices.append(self._df._index.index(label))
                    except ValueError as err:
                        # Pandas fallback: if integer not found and index is not integer-based, use as position
                        if isinstance(label, int) and not is_integer_index:
                            label_pos = label if label >= 0 else len(polars_df) + label
                            if 0 <= label_pos < len(polars_df):
                                row_indices.append(label_pos)
                            else:
                                raise IndexError(
                                    f"index {label_pos} is out of bounds"
                                ) from err
                        else:
                            raise create_keyerror_with_suggestions(
                                str(label),
                                [str(k) for k in self._df._index],
                                context="index",
                            ) from err
            else:
                # Single label - convert to position
                try:
                    row_idx = self._df._index.index(row_key)
                    row_indices = [row_idx]
                except ValueError as err:
                    # Pandas fallback: if integer not found and index is not integer-based, create new row
                    if isinstance(row_key, int) and not is_integer_index:
                        # Pandas creates a new row with this label when assigning
                        # Add the new label to the index and create a new row
                        self._df._index.append(row_key)
                        # For full row assignment (_set_rows), preserve dtypes - values will be assigned below
                        # No float casting needed since we're assigning actual values, not leaving NaN
                        new_row_data_no_cast: Dict[str, Any] = {}
                        for col in polars_df.columns:
                            dtype = polars_df[col].dtype
                            # Use appropriate defaults that match dtype (preserve int types)
                            if dtype in (
                                pl.Int8,
                                pl.Int16,
                                pl.Int32,
                                pl.Int64,
                                pl.UInt8,
                                pl.UInt16,
                                pl.UInt32,
                                pl.UInt64,
                            ):
                                new_row_data_no_cast[col] = 0
                            elif dtype in (pl.Float32, pl.Float64):
                                new_row_data_no_cast[col] = float("nan")
                            elif dtype == pl.Boolean:
                                new_row_data_no_cast[col] = False
                            else:
                                new_row_data_no_cast[col] = None

                        new_row_df = pl.DataFrame([new_row_data_no_cast])
                        polars_df = pl.concat([polars_df, new_row_df])
                        self._df._df = polars_df
                        row_indices = [len(polars_df) - 1]
                    else:
                        raise create_keyerror_with_suggestions(
                            str(row_key),
                            [str(k) for k in self._df._index],
                            context="index",
                        ) from err
        else:
            # No index - treat row_key as integer position
            # For loc, if index doesn't exist, pandas creates new row with that label
            if isinstance(row_key, slice):
                start = row_key.start if row_key.start is not None else 0
                stop = row_key.stop if row_key.stop is not None else len(polars_df)
                step = row_key.step if row_key.step is not None else 1
                row_indices = list(range(start, stop, step))
            elif isinstance(row_key, list):
                row_indices = row_key
            else:
                row_key_int = row_key
                if isinstance(row_key_int, int) and row_key_int < 0:
                    row_key_int = len(polars_df) + row_key_int
                # Pandas loc creates new row if index doesn't exist (unlike iloc which raises)
                if row_key_int >= len(polars_df) or row_key_int < 0:
                    # Create new row with this integer as the label
                    if self._df._index is None:
                        # Initialize index with current row positions
                        self._df._index = list(range(len(polars_df)))
                    # Add the new label to the index (pandas creates one new row)
                    self._df._index.append(row_key_int)
                    # For full row assignment (_set_rows), preserve dtypes - values will be assigned below
                    new_row_data: Dict[str, Any] = {}
                    for col in polars_df.columns:
                        dtype = polars_df[col].dtype
                        # Use appropriate defaults that match dtype (preserve int types)
                        if dtype in (
                            pl.Int8,
                            pl.Int16,
                            pl.Int32,
                            pl.Int64,
                            pl.UInt8,
                            pl.UInt16,
                            pl.UInt32,
                            pl.UInt64,
                        ):
                            new_row_data[col] = 0
                        elif dtype in (pl.Float32, pl.Float64):
                            new_row_data[col] = float("nan")
                        elif dtype == pl.Boolean:
                            new_row_data[col] = False
                        else:
                            new_row_data[col] = None

                    new_row_df = pl.DataFrame([new_row_data])
                    polars_df = pl.concat([polars_df, new_row_df])
                    self._df._df = polars_df
                    row_indices = [len(polars_df) - 1]
                else:
                    row_indices = [row_key_int]

        # Handle value types (same pattern as _ILocIndexer._set_rows)
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
        """Set rows and columns by label."""
        polars_df = self._df._df
        import polars as pl

        from polarpandas._exceptions import create_keyerror_with_suggestions

        # Handle boolean indexing with pandas Series
        if hasattr(row_key, "dtype") and str(row_key.dtype) == "bool":
            # Convert pandas Series mask to Polars Series for efficient boolean indexing
            if hasattr(row_key, "tolist"):
                mask_values = row_key.tolist()
            else:
                mask_values = list(row_key)

            # Validate mask length
            if len(mask_values) != len(polars_df):
                raise ValueError(
                    f"Length of values ({len(mask_values)}) does not match length of index ({len(polars_df)})"
                )

            # Handle column key first
            if isinstance(col_key, slice):
                start = col_key.start if col_key.start is not None else 0
                stop = (
                    col_key.stop if col_key.stop is not None else len(polars_df.columns)
                )
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
                        if c not in polars_df.columns:
                            from polarpandas._exceptions import (
                                create_keyerror_with_suggestions,
                            )

                            raise create_keyerror_with_suggestions(
                                c, polars_df.columns, context="column"
                            )
                        col_names.append(str(c))
            elif isinstance(col_key, int):
                col_key_int = col_key
                if col_key_int < 0:
                    col_key_int = len(polars_df.columns) + col_key_int
                if col_key_int >= len(polars_df.columns) or col_key_int < 0:
                    raise IndexError(
                        f"index {col_key_int} is out of bounds for axis 1 with size {len(polars_df.columns)}"
                    )
                col_names = [polars_df.columns[col_key_int]]
            else:
                # String column name
                if col_key not in polars_df.columns:
                    from polarpandas._exceptions import create_keyerror_with_suggestions

                    raise create_keyerror_with_suggestions(
                        col_key, polars_df.columns, context="column"
                    )
                col_names = [col_key]

            # Convert mask to Polars Series for efficient operations
            mask_series = pl.Series("mask", mask_values)

            # Use Polars native boolean indexing instead of nested when chains
            # This is much more efficient than building row_indices list
            new_cols = []
            for col_name in col_names:
                if col_name in polars_df.columns:
                    # Use Polars when() with boolean mask directly
                    new_cols.append(
                        pl.when(mask_series)
                        .then(pl.lit(value))
                        .otherwise(pl.col(col_name))
                        .alias(col_name)
                    )
                else:
                    # New column
                    new_cols.append(
                        pl.when(mask_series)
                        .then(pl.lit(value))
                        .otherwise(pl.lit(None))
                        .alias(col_name)
                    )

            self._df._df = polars_df.with_columns(new_cols)
            # Index preserved automatically (no shape change)
            return
        elif self._df._index is not None:
            # We have an index - convert labels to positions
            # Handle pandas fallback: if integer not in index and index is not integer-based, use position
            is_integer_index = (
                all(
                    isinstance(k, (int, type(None)))
                    for k in self._df._index[:10]
                    if k is not None
                )
                if self._df._index
                else False
            )

            if isinstance(row_key, slice):
                # Slice of labels - convert to list of positions
                start = row_key.start
                stop = row_key.stop
                step = row_key.step if row_key.step is not None else 1

                # Find start position
                if start is None:
                    start_pos = 0
                else:
                    try:
                        start_pos = self._df._index.index(start)
                    except ValueError as err:
                        # Pandas fallback for integer slices
                        if isinstance(start, int) and not is_integer_index:
                            start_pos = start if start >= 0 else len(polars_df) + start
                        else:
                            raise create_keyerror_with_suggestions(
                                str(start),
                                [str(k) for k in self._df._index],
                                context="index",
                            ) from err

                # Find stop position
                if stop is None:
                    stop_pos = len(self._df._index)
                else:
                    try:
                        stop_pos = (
                            self._df._index.index(stop) + 1
                        )  # +1 because slice is exclusive
                    except ValueError as err:
                        # Pandas fallback for integer slices
                        if isinstance(stop, int) and not is_integer_index:
                            stop_pos = (
                                stop if stop >= 0 else len(polars_df) + stop
                            ) + 1
                        else:
                            raise create_keyerror_with_suggestions(
                                str(stop),
                                [str(k) for k in self._df._index],
                                context="index",
                            ) from err

                row_indices = list(range(start_pos, stop_pos, step))
            elif isinstance(row_key, list):
                # List of labels - convert each to position
                row_indices = []
                for label in row_key:
                    try:
                        row_indices.append(self._df._index.index(label))
                    except ValueError as err:
                        # Pandas fallback: if integer not found and index is not integer-based, use as position
                        if isinstance(label, int) and not is_integer_index:
                            label_pos = label if label >= 0 else len(polars_df) + label
                            if 0 <= label_pos < len(polars_df):
                                row_indices.append(label_pos)
                            else:
                                raise IndexError(
                                    f"index {label_pos} is out of bounds"
                                ) from err
                        else:
                            raise create_keyerror_with_suggestions(
                                str(label),
                                [str(k) for k in self._df._index],
                                context="index",
                            ) from err
            else:
                # Single label - convert to position
                try:
                    row_idx = self._df._index.index(row_key)
                    row_indices = [row_idx]
                except ValueError as err:
                    # Pandas fallback: if integer not found and index is not integer-based, create new row
                    if isinstance(row_key, int) and not is_integer_index:
                        # Pandas creates a new row with this label when assigning
                        # Add the new label to the index and create a new row
                        self._df._index.append(row_key)
                        # Create a new row with NaN for all columns (pandas behavior)
                        # First, cast integer columns to float to allow NaN (matching pandas behavior)
                        cast_exprs = []
                        new_row_data_cast: Dict[str, Any] = {}
                        for col in polars_df.columns:
                            dtype = polars_df[col].dtype
                            if dtype in (
                                pl.Int8,
                                pl.Int16,
                                pl.Int32,
                                pl.Int64,
                                pl.UInt8,
                                pl.UInt16,
                                pl.UInt32,
                                pl.UInt64,
                            ):
                                cast_exprs.append(
                                    pl.col(col).cast(pl.Float64).alias(col)
                                )
                                new_row_data_cast[col] = float("nan")
                            elif dtype in (pl.Float32, pl.Float64):
                                new_row_data_cast[col] = float("nan")
                            else:
                                # String and other types use None
                                new_row_data_cast[col] = None

                        if cast_exprs:
                            polars_df = polars_df.with_columns(cast_exprs)
                        new_row_df = pl.DataFrame([new_row_data_cast])
                        polars_df = pl.concat([polars_df, new_row_df])
                        self._df._df = polars_df
                        row_indices = [len(polars_df) - 1]
                    else:
                        raise create_keyerror_with_suggestions(
                            str(row_key),
                            [str(k) for k in self._df._index],
                            context="index",
                        ) from err
        else:
            # No index - treat row_key as integer position
            # For loc, if index doesn't exist, pandas creates new row with that label
            if isinstance(row_key, slice):
                start = row_key.start if row_key.start is not None else 0
                stop = row_key.stop if row_key.stop is not None else len(polars_df)
                step = row_key.step if row_key.step is not None else 1
                row_indices = list(range(start, stop, step))
            elif isinstance(row_key, list):
                row_indices = row_key
            else:
                row_key_int = row_key
                if isinstance(row_key_int, int) and row_key_int < 0:
                    row_key_int = len(polars_df) + row_key_int
                # Pandas loc creates new row if index doesn't exist (unlike iloc which raises)
                if row_key_int >= len(polars_df) or row_key_int < 0:
                    # Create new row with this integer as the label
                    if self._df._index is None:
                        # Initialize index with current row positions
                        self._df._index = list(range(len(polars_df)))
                    # Add the new label to the index (pandas creates one new row)
                    self._df._index.append(row_key_int)
                    # Create the new row
                    cast_exprs = []
                    new_row_data_no_index: Dict[str, Any] = {}
                    for col in polars_df.columns:
                        dtype = polars_df[col].dtype
                        if dtype in (
                            pl.Int8,
                            pl.Int16,
                            pl.Int32,
                            pl.Int64,
                            pl.UInt8,
                            pl.UInt16,
                            pl.UInt32,
                            pl.UInt64,
                        ):
                            cast_exprs.append(pl.col(col).cast(pl.Float64).alias(col))
                            new_row_data_no_index[col] = float("nan")
                        elif dtype in (pl.Float32, pl.Float64):
                            new_row_data_no_index[col] = float("nan")
                        else:
                            new_row_data_no_index[col] = None

                    if cast_exprs:
                        polars_df = polars_df.with_columns(cast_exprs)

                    new_row_df = pl.DataFrame([new_row_data_no_index])
                    polars_df = pl.concat([polars_df, new_row_df])
                    self._df._df = polars_df
                    row_indices = [len(polars_df) - 1]
                else:
                    row_indices = [row_key_int]

        # Handle column key (same as _ILocIndexer)
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
                    if c not in polars_df.columns:
                        raise create_keyerror_with_suggestions(
                            c, polars_df.columns, context="column"
                        )
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
            if col_key not in polars_df.columns:
                raise create_keyerror_with_suggestions(
                    col_key, polars_df.columns, context="column"
                )
            col_names = [col_key]

        # Update each column using conditional expressions (same pattern as _ILocIndexer)
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
        if isinstance(row_key, slice):
            selected_df = polars_df[row_key]
            return DataFrame(selected_df)
        elif isinstance(row_key, list):
            selected_df = polars_df[row_key]
            return DataFrame(selected_df)
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
        if isinstance(row_key, slice):
            selected_df = polars_df[row_key, col_key]
            return DataFrame(selected_df)
        elif isinstance(row_key, list):
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

    def min(self) -> "DataFrame":
        """Calculate rolling minimum."""
        return self._apply_rolling("min")


class _GroupBy:
    """GroupBy object for grouped operations."""

    def __init__(self, polars_groupby: Any, parent_df: DataFrame) -> None:
        """
        Initialize GroupBy wrapper.

        Parameters
        ----------
        polars_groupby : polars GroupBy object
            The underlying Polars GroupBy object
        parent_df : DataFrame
            Parent DataFrame being grouped
        """
        self._gb = polars_groupby
        self._parent = parent_df

    def agg(self, *args: Any, **kwargs: Any) -> "DataFrame":
        """
        Aggregate using one or more operations.

        Returns
        -------
        DataFrame
            Aggregated DataFrame
        """
        result = self._gb.agg(*args, **kwargs)
        return DataFrame(result)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying Polars GroupBy object."""
        attr = getattr(self._gb, name)
        # If it's a method that returns a DataFrame, wrap it
        if callable(attr):

            def wrapper(*args: Any, **kwargs: Any) -> Union["DataFrame", Any]:
                result = attr(*args, **kwargs)
                if hasattr(result, "columns"):  # It's a DataFrame-like object
                    return DataFrame(result)
                return result

            return wrapper
        return attr
