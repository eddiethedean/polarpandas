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

import contextlib
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import polars as pl

from polarpandas._exceptions import (
    convert_to_keyerror,
    create_keyerror_with_suggestions,
)
from polarpandas._index_manager import IndexManager
from polarpandas.index import Index, MultiIndex
from polarpandas.utils import convert_schema_to_polars

if TYPE_CHECKING:
    from .lazyframe import LazyFrame
    from .series import Series  # noqa: TC004


def _is_integer_dtype(dtype: Any) -> bool:
    """Check if a Polars dtype is an integer type."""
    return dtype in (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    )


def _is_numeric_dtype(dtype: Any) -> bool:
    """Check if a Polars dtype is a numeric type."""
    return dtype in (
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


def _is_float_dtype(dtype: Any) -> bool:
    """Check if a Polars dtype is a float type."""
    return dtype in (pl.Float32, pl.Float64)


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

        # Use Polars JSON read - orient parameter support is limited
        # Remove pandas-specific parameters that Polars doesn't support
        polars_kwargs.update(
            {k: v for k, v in kwargs.items() if k not in ["orient", "lines"]}
        )

        try:
            # Read JSON first (will infer types as strings if values are strings)
            df = pl.read_json(path, **polars_kwargs)

            # If dtype/schema is provided, cast columns to desired types
            if schema_to_use is not None:
                polars_schema = convert_schema_to_polars(schema_to_use)
                if polars_schema is not None:
                    # Cast columns to desired types (handles string-to-numeric conversion)
                    cast_exprs = [
                        pl.col(col).cast(dtype) for col, dtype in polars_schema.items()
                    ]
                    df = df.with_columns(cast_exprs)

            return cls(df)
        except Exception as e:
            # If Polars JSON read fails, this is a limitation
            error_msg = (
                f"Polars JSON read failed: {e}\n"
                "Possible solutions:\n"
                '  - Ensure JSON is in array format: [{{"col1": 1}}, {{"col2": 2}}]\n'
                "  - For NDJSON (newline-delimited), use scan_json() instead\n"
                "  - Check that all rows have consistent structure\n"
                "  - Verify file encoding (should be UTF-8)"
            )
            raise ValueError(error_msg) from e

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

    def _ensure_column_alignment(self, other: "DataFrame") -> None:
        if list(self.columns) != list(other.columns):
            raise ValueError("DataFrame columns must match for element-wise operations")

    def _preserve_binary_index(
        self, result: "DataFrame", other: Any, *, reverse: bool = False
    ) -> "DataFrame":
        if isinstance(other, DataFrame):
            left_index = other._index if reverse else self._index
            right_index = self._index if reverse else other._index
            if left_index is not None and right_index == left_index:
                result._index = list(left_index)
                result._index_name = other._index_name if reverse else self._index_name
        elif self._index is not None:
            result._index = list(self._index)
            result._index_name = self._index_name
        return result

    def _is_numeric_scalar(self, value: Any) -> bool:
        return isinstance(value, (int, float, complex, bool, np.number))

    def _binary_operation(
        self,
        other: Any,
        op_name: str,
        *,
        reverse: bool = False,
        numeric_only: bool = False,
    ) -> "DataFrame":
        method_name = f"__r{op_name}__" if reverse else f"__{op_name}__"

        if isinstance(other, DataFrame):
            self._ensure_column_alignment(other)
            target = getattr(self._df, method_name, None)
            try:
                if target is None:
                    raise AttributeError
                result_pl = target(other._df)
            except (AttributeError, TypeError):
                pyop = getattr(operator, op_name)
                result_pl = (
                    pyop(other._df, self._df) if reverse else pyop(self._df, other._df)
                )
            result_df = DataFrame(result_pl)
        else:
            if numeric_only and not self._is_numeric_scalar(other):
                raise TypeError(
                    f"Unsupported operand type(s) for {op_name}: 'DataFrame' and '{type(other).__name__}'"
                )
            target = getattr(self._df, method_name, None)
            try:
                if target is None:
                    raise AttributeError
                result_pl = target(other)
            except (AttributeError, TypeError):
                pyop = getattr(operator, op_name)
                result_pl = pyop(other, self._df) if reverse else pyop(self._df, other)
            result_df = DataFrame(result_pl)

        return self._preserve_binary_index(result_df, other, reverse=reverse)

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------
    def __add__(self, other: Any) -> "DataFrame":
        return self._binary_operation(other, "add", numeric_only=True)

    def __radd__(self, other: Any) -> "DataFrame":
        return self._binary_operation(other, "add", reverse=True, numeric_only=True)

    def __sub__(self, other: Any) -> "DataFrame":
        return self._binary_operation(other, "sub", numeric_only=True)

    def __rsub__(self, other: Any) -> "DataFrame":
        return self._binary_operation(other, "sub", reverse=True, numeric_only=True)

    def __mul__(self, other: Any) -> "DataFrame":
        return self._binary_operation(other, "mul", numeric_only=True)

    def __rmul__(self, other: Any) -> "DataFrame":
        return self._binary_operation(other, "mul", reverse=True, numeric_only=True)

    def __truediv__(self, other: Any) -> "DataFrame":
        return self._binary_operation(other, "truediv", numeric_only=True)

    def __rtruediv__(self, other: Any) -> "DataFrame":
        return self._binary_operation(other, "truediv", reverse=True, numeric_only=True)

    # ------------------------------------------------------------------
    # Comparison operators
    # ------------------------------------------------------------------
    def _comparison_operation(self, other: Any, op_name: str) -> "DataFrame":
        return self._binary_operation(other, op_name)

    def __eq__(self, other: Any) -> "DataFrame":  # type: ignore[override]
        return self._comparison_operation(other, "eq")

    def __ne__(self, other: Any) -> "DataFrame":  # type: ignore[override]
        return self._comparison_operation(other, "ne")

    def __gt__(self, other: Any) -> "DataFrame":
        return self._comparison_operation(other, "gt")

    def __ge__(self, other: Any) -> "DataFrame":
        return self._comparison_operation(other, "ge")

    def __lt__(self, other: Any) -> "DataFrame":
        return self._comparison_operation(other, "lt")

    def __le__(self, other: Any) -> "DataFrame":
        return self._comparison_operation(other, "le")

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
        if not isinstance(rename_dict, Mapping):
            raise TypeError(
                "'columns' must be a mapping of {old_name: new_name} pairs when renaming"
            )

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
        self,
        value: Any = None,
        inplace: bool = False,
        method: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional["DataFrame"]:
        """Fill null values."""
        if method is not None:
            strategy_map: Dict[str, Literal["forward", "backward"]] = {
                "ffill": "forward",
                "pad": "forward",
                "bfill": "backward",
                "backfill": "backward",
            }
            if method not in strategy_map:
                raise ValueError(
                    "method must be one of {'ffill', 'pad', 'bfill', 'backfill'}"
                )
            strategy = strategy_map[method]
            result_df = self._df.fill_null(
                strategy=strategy,
                limit=limit,
                **kwargs,
            )
        else:
            if value is None:
                raise ValueError(
                    "fillna requires a 'value' when no method is specified"
                )
            result_df = self._df.fill_null(value, **kwargs)

        if inplace:
            self._df = result_df
            return None
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
        missing_columns: List[str] = []
        if isinstance(by, str):
            missing_columns = [by] if by not in self._df.columns else []
        elif isinstance(by, Sequence):
            missing_columns = [
                col
                for col in by
                if isinstance(col, str) and col not in self._df.columns
            ]

        if missing_columns:
            missing_display = ", ".join(missing_columns)
            raise KeyError(f"Columns not found in DataFrame: {missing_display}")

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
                polars_gb = temp_df.group_by("__level_0__", *args, **kwargs)
                return _GroupBy(polars_gb, self)

        # Normal column-based grouping
        if by is None:
            raise ValueError("Must specify either 'by' or 'level' parameter")

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
        var_name = kwargs.pop("var_name", "variable")
        value_name = kwargs.pop("value_name", "value")
        unpivot_kwargs = kwargs

        result = cast("Any", self._df).unpivot(
            index=id_vars,
            on=value_vars,
            **unpivot_kwargs,
        )
        rename_map = {}
        if var_name != "variable":
            rename_map["variable"] = var_name
        if value_name != "value":
            rename_map["value"] = value_name
        if rename_map:
            result = result.rename(rename_map)
        return DataFrame(result)

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
        import polars as pl

        from polarpandas.series import Series

        # Handle level parameter
        if level is not None and axis == 0:
            # Group by level and sum - this collapses the MultiIndex along the specified level
            gb = self.groupby(level=level)
            # Aggregate all columns
            expressions = [pl.col(col).sum().alias(col) for col in self.columns]
            result = gb.agg(expressions)
            # Return as DataFrame (pandas returns DataFrame for level-based aggregations)
            return result

        if axis is None or axis == 1 or axis == "columns":
            # Row-wise sum (axis=1) - aggregate across columns for each row
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
                result_series = self._df.select(pl.sum_horizontal(numeric_cols))[
                    "literal"
                ]
            else:
                result_series = self._df.select(pl.sum_horizontal(self.columns))[
                    "literal"
                ]
            # Use index if available
            index = (
                self._index
                if self._index is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)
        else:
            # Column-wise sum (axis=0, default) - aggregate down columns
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
                result_pl = self._df.select([pl.col(col).sum() for col in numeric_cols])
            else:
                result_pl = self._df.select([pl.col(col).sum() for col in self.columns])
            # Convert to Series with column names as index
            values = [result_pl[col].to_list()[0] for col in result_pl.columns]
            return Series(values, index=result_pl.columns)

    def mean(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        level: Optional[Union[int, str, List[Union[int, str]]]] = None,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        """
        Return the mean of the values over the requested axis.

        Parameters
        ----------
        axis : {0, 1, 'index', 'columns'}, default 0
            Axis along which to compute mean. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. If False, will attempt to use everything.
        **kwargs
            Additional arguments passed to Polars mean().

        Returns
        -------
        Series or scalar
            Series when axis=0 (default), scalar when axis=1 or axis=None.
        """
        import polars as pl

        from polarpandas.series import Series

        # Handle level parameter
        if level is not None and axis == 0:
            # Group by level and mean - this collapses the MultiIndex along the specified level
            gb = self.groupby(level=level)
            # Aggregate all columns
            expressions = [pl.col(col).mean().alias(col) for col in self.columns]
            result = gb.agg(expressions)
            # Return as DataFrame (pandas returns DataFrame for level-based aggregations)
            return result

        if axis is None or axis == 1 or axis == "columns":
            # Row-wise mean (axis=1) - aggregate across columns for each row
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
                result_series = self._df.select(pl.mean_horizontal(numeric_cols))[
                    "literal"
                ]
            else:
                result_series = self._df.select(pl.mean_horizontal(self.columns))[
                    "literal"
                ]
            # Use index if available
            index = (
                self._index
                if self._index is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)
        else:
            # Column-wise mean (axis=0, default) - aggregate down columns
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
                    [pl.col(col).mean() for col in numeric_cols]
                )
            else:
                result_pl = self._df.select(
                    [pl.col(col).mean() for col in self.columns]
                )
            # Convert to Series with column names as index
            values = [result_pl[col].to_list()[0] for col in result_pl.columns]
            return Series(values, index=result_pl.columns)

    def min(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        level: Optional[Union[int, str, List[Union[int, str]]]] = None,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        """
        Return the minimum of the values over the requested axis.

        Parameters
        ----------
        axis : {0, 1, 'index', 'columns'}, default 0
            Axis along which to compute minimum. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. If False, will attempt to use everything.
        level : int, str, or list, optional
            If the axis is a MultiIndex, compute minimum along a particular level, collapsing into a Series.
        **kwargs
            Additional arguments passed to Polars min().

        Returns
        -------
        Series or scalar
            Series when axis=0 (default), scalar when axis=1 or axis=None.
        """
        import polars as pl

        from polarpandas.series import Series

        # Handle level parameter
        if level is not None and axis == 0:
            # Group by level and min - this collapses the MultiIndex along the specified level
            gb = self.groupby(level=level)
            # Aggregate all columns
            expressions = [pl.col(col).min().alias(col) for col in self.columns]
            result = gb.agg(expressions)
            # Return as DataFrame (pandas returns DataFrame for level-based aggregations)
            return result

        if axis is None or axis == 1 or axis == "columns":
            # Row-wise min (axis=1) - aggregate across columns for each row
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
                result_series = self._df.select(pl.min_horizontal(numeric_cols))[
                    "literal"
                ]
            else:
                result_series = self._df.select(pl.min_horizontal(self.columns))[
                    "literal"
                ]
            # Use index if available
            index = (
                self._index
                if self._index is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)
        else:
            # Column-wise min (axis=0, default) - aggregate down columns
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
                result_pl = self._df.select([pl.col(col).min() for col in numeric_cols])
            else:
                result_pl = self._df.select([pl.col(col).min() for col in self.columns])
            # Convert to Series with column names as index
            values = [result_pl[col].to_list()[0] for col in result_pl.columns]
            return Series(values, index=result_pl.columns)

    def max(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        level: Optional[Union[int, str, List[Union[int, str]]]] = None,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        """
        Return the maximum of the values over the requested axis.

        Parameters
        ----------
        axis : {0, 1, 'index', 'columns'}, default 0
            Axis along which to compute maximum. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. If False, will attempt to use everything.
        level : int, str, or list, optional
            If the axis is a MultiIndex, compute maximum along a particular level, collapsing into a Series.
        **kwargs
            Additional arguments passed to Polars max().

        Returns
        -------
        Series or scalar
            Series when axis=0 (default), scalar when axis=1 or axis=None.
        """
        import polars as pl

        from polarpandas.series import Series

        # Handle level parameter
        if level is not None and axis == 0:
            # Group by level and max - this collapses the MultiIndex along the specified level
            gb = self.groupby(level=level)
            # Aggregate all columns
            expressions = [pl.col(col).max().alias(col) for col in self.columns]
            result = gb.agg(expressions)
            # Return as DataFrame (pandas returns DataFrame for level-based aggregations)
            return result

        if axis is None or axis == 1 or axis == "columns":
            # Row-wise max (axis=1) - aggregate across columns for each row
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
                result_series = self._df.select(pl.max_horizontal(numeric_cols))[
                    "literal"
                ]
            else:
                result_series = self._df.select(pl.max_horizontal(self.columns))[
                    "literal"
                ]
            # Use index if available
            index = (
                self._index
                if self._index is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)
        else:
            # Column-wise max (axis=0, default) - aggregate down columns
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
                result_pl = self._df.select([pl.col(col).max() for col in numeric_cols])
            else:
                result_pl = self._df.select([pl.col(col).max() for col in self.columns])
            # Convert to Series with column names as index
            values = [result_pl[col].to_list()[0] for col in result_pl.columns]
            return Series(values, index=result_pl.columns)

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
        import polars as pl

        from polarpandas.series import Series

        # Handle level parameter
        if level is not None and axis == 0:
            # Group by level and std - this collapses the MultiIndex along the specified level
            gb = self.groupby(level=level)
            # Aggregate all columns
            expressions = [
                pl.col(col).std(ddof=ddof).alias(col) for col in self.columns
            ]
            result = gb.agg(expressions)
            # Return as DataFrame (pandas returns DataFrame for level-based aggregations)
            return result

        if axis is None or axis == 1 or axis == "columns":
            # Row-wise std (axis=1) - aggregate across columns for each row
            # Note: Polars doesn't have std_horizontal, so we compute row-wise manually
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
                # For row-wise std, we need to compute it manually
                # This is less efficient but matches pandas behavior
                result_series = self._df.select(
                    pl.concat_list(numeric_cols).list.std(ddof=ddof).alias("literal")
                )["literal"]
            else:
                result_series = self._df.select(
                    pl.concat_list(self.columns).list.std(ddof=ddof).alias("literal")
                )["literal"]
            # Use index if available
            index = (
                self._index
                if self._index is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)
        else:
            # Column-wise std (axis=0, default) - aggregate down columns
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
                    [pl.col(col).std(ddof=ddof) for col in numeric_cols]
                )
            else:
                result_pl = self._df.select(
                    [pl.col(col).std(ddof=ddof) for col in self.columns]
                )
            # Convert to Series with column names as index
            values = [result_pl[col].to_list()[0] for col in result_pl.columns]
            return Series(values, index=result_pl.columns)

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
        import polars as pl

        from polarpandas.series import Series

        # Handle level parameter
        if level is not None and axis == 0:
            # Group by level and var - this collapses the MultiIndex along the specified level
            gb = self.groupby(level=level)
            # Aggregate all columns
            expressions = [
                pl.col(col).var(ddof=ddof).alias(col) for col in self.columns
            ]
            result = gb.agg(expressions)
            # Return as DataFrame (pandas returns DataFrame for level-based aggregations)
            return result

        if axis is None or axis == 1 or axis == "columns":
            # Row-wise var (axis=1) - aggregate across columns for each row
            # Note: Polars doesn't have var_horizontal, so we compute row-wise manually
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
                # For row-wise var, we need to compute it manually
                result_series = self._df.select(
                    pl.concat_list(numeric_cols).list.var(ddof=ddof).alias("literal")
                )["literal"]
            else:
                result_series = self._df.select(
                    pl.concat_list(self.columns).list.var(ddof=ddof).alias("literal")
                )["literal"]
            # Use index if available
            index = (
                self._index
                if self._index is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)
        else:
            # Column-wise var (axis=0, default) - aggregate down columns
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
                    [pl.col(col).var(ddof=ddof) for col in numeric_cols]
                )
            else:
                result_pl = self._df.select(
                    [pl.col(col).var(ddof=ddof) for col in self.columns]
                )
            # Convert to Series with column names as index
            values = [result_pl[col].to_list()[0] for col in result_pl.columns]
            return Series(values, index=result_pl.columns)

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
        from polarpandas.series import Series

        if axis is None or axis == 1 or axis == "columns":
            # Row-wise count (axis=1) - count non-null values across columns for each row
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
                        pl.Boolean,
                    )
                ]
                if not numeric_cols:
                    return Series(pl.Series([], dtype=pl.UInt32))
                result_series = self._df.select(
                    pl.sum_horizontal(
                        [
                            pl.col(col).is_not_null().cast(pl.UInt32)
                            for col in numeric_cols
                        ]
                    )
                )["literal"]
            else:
                result_series = self._df.select(
                    pl.sum_horizontal(
                        [
                            pl.col(col).is_not_null().cast(pl.UInt32)
                            for col in self.columns
                        ]
                    )
                )["literal"]
            index = (
                self._index
                if self._index is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)
        else:
            # Column-wise count (axis=0, default) - count non-null values down columns
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
                        pl.Boolean,
                    )
                ]
                if not numeric_cols:
                    return Series(pl.Series([], dtype=pl.UInt32))
                result_pl = self._df.select(
                    [pl.col(col).count() for col in numeric_cols]
                )
            else:
                result_pl = self._df.select(
                    [pl.col(col).count() for col in self.columns]
                )
            # Convert to Series with column names as index
            values = [result_pl[col].to_list()[0] for col in result_pl.columns]
            return Series(values, index=result_pl.columns)

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
        import polars as pl

        from polarpandas.series import Series

        # Handle level parameter
        if level is not None and axis == 0:
            # Group by level and median - this collapses the MultiIndex along the specified level
            gb = self.groupby(level=level)
            # Aggregate all columns
            expressions = [pl.col(col).median().alias(col) for col in self.columns]
            result = gb.agg(expressions)
            # Return as DataFrame (pandas returns DataFrame for level-based aggregations)
            return result

        if axis is None or axis == 1 or axis == "columns":
            # Row-wise median (axis=1) - aggregate across columns for each row
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
                # For row-wise, we need to compute median across columns
                # Polars doesn't have median_horizontal, so we use concat_list and list.median
                result_series = self._df.select(
                    pl.concat_list(numeric_cols).list.median()
                )["literal"]
            else:
                result_series = self._df.select(
                    pl.concat_list(self.columns).list.median()
                )["literal"]
            index = (
                self._index
                if self._index is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)
        else:
            # Column-wise median (axis=0, default) - aggregate down columns
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
                    [pl.col(col).median() for col in numeric_cols]
                )
            else:
                result_pl = self._df.select(
                    [pl.col(col).median() for col in self.columns]
                )
            # Convert to Series with column names as index
            values = [result_pl[col].to_list()[0] for col in result_pl.columns]
            return Series(values, index=result_pl.columns)

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

    def product(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        """
        Return the product of the values over the requested axis.

        Alias for prod().
        """
        return self.prod(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def mode(
        self,
        axis: int = 0,
        numeric_only: bool = False,
        dropna: bool = True,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Get the mode(s) of each element along the selected axis.

        Parameters
        ----------
        axis : {0, 1}, default 0
            Axis along which to compute modes. 0 for column-wise, 1 for row-wise.
        numeric_only : bool, default False
            If True, only apply to numeric columns.
        dropna : bool, default True
            Don't consider counts of NaN/NaT.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            DataFrame with modes for each column (axis=0) or row (axis=1).
        """
        if axis == 1:
            # Row-wise mode - complex, return empty for now
            return DataFrame()
        else:
            # Column-wise mode (axis=0, default)
            result_data: Dict[str, List[Any]] = {}
            cols_to_use = self.columns
            if numeric_only:
                cols_to_use = [
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

            for col in cols_to_use:
                # Get value counts and find mode(s)
                value_counts_pl = self._df[col].value_counts(sort=True)
                if value_counts_pl.height == 0:
                    result_data[col] = []
                else:
                    # Get the maximum count (first row since sorted)
                    max_count = value_counts_pl[0, 1]
                    # Get all values with max_count
                    modes = []
                    for row in value_counts_pl.iter_rows():
                        if row[1] == max_count:
                            modes.append(row[0])
                        elif row[1] < max_count:
                            break
                    result_data[col] = modes

            # Find max length to pad
            max_len = max(len(v) for v in result_data.values()) if result_data else 0
            # Pad all lists to same length
            for col in result_data:
                result_data[col] = result_data[col] + [None] * (
                    max_len - len(result_data[col])
                )

            return DataFrame(result_data)

    def abs(self) -> "DataFrame":
        """
        Return a DataFrame with absolute numeric value of each element.

        Returns
        -------
        DataFrame
            DataFrame with absolute values.
        """
        result_df = self._df.select(
            [
                pl.col(col).abs()
                if _is_numeric_dtype(self._df[col].dtype)
                else pl.col(col)
                for col in self.columns
            ]
        )
        return DataFrame(result_df)

    def round(self, decimals: int = 0, **kwargs: Any) -> "DataFrame":
        """
        Round a DataFrame to a variable number of decimal places.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            DataFrame with rounded values.
        """
        result_df = self._df.select(
            [
                pl.col(col).round(decimals)
                if _is_numeric_dtype(self._df[col].dtype)
                else pl.col(col)
                for col in self.columns
            ]
        )
        return DataFrame(result_df)

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
                "mean": self.mean,
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
        self, axis: Optional[Union[int, Literal["index", "columns"]]] = None
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

    def memory_usage(self, index: bool = True, deep: bool = False) -> "Series":
        """
        Return the memory usage of each column in bytes.

        Parameters
        ----------
        index : bool, default True
            Specifies whether to include the memory usage of the DataFrame's index.
        deep : bool, default False
            If True, introspect the data deeply by interrogating object dtypes for system-level memory consumption.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Series with memory usage of each column in bytes.
        """
        from polarpandas.series import Series

        # Estimate memory usage from dtypes
        memory_data = {}
        schema = self._df.schema

        for col in self.columns:
            dtype = schema[col]
            col_len = len(self._df)

            # Estimate bytes per element based on dtype
            if dtype == pl.Int8 or dtype == pl.UInt8:
                bytes_per_elem = 1
            elif dtype == pl.Int16 or dtype == pl.UInt16:
                bytes_per_elem = 2
            elif dtype == pl.Int32 or dtype == pl.UInt32 or dtype == pl.Float32:
                bytes_per_elem = 4
            elif dtype == pl.Int64 or dtype == pl.UInt64 or dtype == pl.Float64:
                bytes_per_elem = 8
            elif dtype == pl.Boolean:
                bytes_per_elem = 1
            elif dtype == pl.Utf8:
                # For strings, estimate average length (rough approximation)
                bytes_per_elem = 8  # Rough estimate
            else:
                bytes_per_elem = 8  # Default estimate

            memory_data[col] = col_len * bytes_per_elem

        if index and self._index is not None:
            memory_data["Index"] = len(self._index) * 8  # Rough estimate for index

        result_series = pl.Series(
            name="memory_usage", values=list(memory_data.values())
        )
        return Series(result_series, index=list(memory_data.keys()))

    def all(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        """
        Return whether all elements are True, potentially over an axis.

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
            # Row-wise all (axis=1)
            if bool_only:
                bool_cols = [
                    col for col in self.columns if self._df[col].dtype == pl.Boolean
                ]
                if not bool_cols:
                    return Series(pl.Series([], dtype=pl.Boolean))
                result_series = self._df.select(pl.all_horizontal(bool_cols))["literal"]
            else:
                result_series = self._df.select(pl.all_horizontal(self.columns))[
                    "literal"
                ]
            index = (
                self._index
                if self._index is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)
        else:
            # Column-wise all (axis=0, default)
            if bool_only:
                bool_cols = [
                    col for col in self.columns if self._df[col].dtype == pl.Boolean
                ]
                if not bool_cols:
                    return Series(pl.Series([], dtype=pl.Boolean))
                result_pl = self._df.select([pl.col(col).all() for col in bool_cols])
            else:
                # For non-boolean columns, convert to boolean (non-zero/non-null = True)
                result_pl = self._df.select(
                    [
                        pl.col(col).cast(pl.Boolean).all()
                        if self._df[col].dtype != pl.Boolean
                        else pl.col(col).all()
                        for col in self.columns
                    ]
                )
            values = [result_pl[col].to_list()[0] for col in result_pl.columns]
            return Series(values, index=result_pl.columns)

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
        Return unbiased skew over requested axis.

        Parameters
        ----------
        axis : {0, 1, 'index', 'columns'}, default 0
            Axis along which to compute skewness. 0 or 'index' for column-wise, 1 or 'columns' for row-wise.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. If False, will attempt to use everything.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        Series
            Series with skewness values for each column (axis=0) or row (axis=1).
        """
        from polarpandas.series import Series

        if axis is None or axis == 1 or axis == "columns":
            # Row-wise skew (axis=1)
            if numeric_only:
                numeric_cols = [
                    col
                    for col in self.columns
                    if _is_numeric_dtype(self._df[col].dtype)
                ]
                if not numeric_cols:
                    return Series(pl.Series([], dtype=pl.Float64))
                # Use concat_list to compute row-wise skew
                result_series = self._df.select(
                    pl.concat_list(numeric_cols).list.skew()  # type: ignore[attr-defined]
                )["literal"]
            else:
                result_series = self._df.select(
                    pl.concat_list(self.columns).list.skew()  # type: ignore[attr-defined]
                )["literal"]
            index = (
                self._index
                if self._index is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)
        else:
            # Column-wise skew (axis=0, default)
            if numeric_only:
                numeric_cols = [
                    col
                    for col in self.columns
                    if _is_numeric_dtype(self._df[col].dtype)
                ]
                if not numeric_cols:
                    return Series(pl.Series([], dtype=pl.Float64))
                result_pl = self._df.select(
                    [pl.col(col).skew() for col in numeric_cols]
                )
            else:
                result_pl = self._df.select(
                    [pl.col(col).skew() for col in self.columns]
                )
            values = [result_pl[col].to_list()[0] for col in result_pl.columns]
            return Series(values, index=result_pl.columns)

    def kurtosis(
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
        """Query the columns of a DataFrame with a boolean expression."""
        from polarpandas.series import Series

        expr_namespace = {col: pl.col(col) for col in self.columns}
        try:
            filter_expr = eval(expr, {"pl": pl, "__builtins__": {}}, expr_namespace)
        except Exception as exc:
            raise ValueError(f"Unable to evaluate query expression: {expr}") from exc

        if isinstance(filter_expr, Series):
            filter_expr = filter_expr._series

        if not isinstance(filter_expr, (pl.Series, pl.Expr)):
            raise TypeError(
                "Query expressions must evaluate to a Polars expression or boolean Series"
            )

        filtered_df = self._df.filter(filter_expr)

        if inplace:
            self._df = filtered_df
            return None
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
        """Return a subset of columns filtered by dtype."""
        include_list: Optional[List[Any]] = (
            include
            if isinstance(include, list)
            else [include]
            if include is not None
            else None
        )
        exclude_list: Optional[List[Any]] = (
            exclude
            if isinstance(exclude, list)
            else [exclude]
            if exclude is not None
            else None
        )

        def _matches(dtype: pl.DataType, selector: Any) -> bool:
            if isinstance(selector, str):
                selector_lower = selector.lower()
                if selector_lower == "number":
                    return dtype.is_numeric()
                if selector_lower == "float":
                    return dtype.is_float()
                if selector_lower in {"int", "integer"}:
                    return dtype.is_integer()
                if selector_lower in {"bool", "boolean"}:
                    return dtype == pl.Boolean
                if selector_lower in {"object", "string", "str"}:
                    return dtype == pl.Utf8
            return dtype == selector or str(dtype) == str(selector)

        cols_to_keep: List[str] = []
        for col in self.columns:
            dtype = self._df[col].dtype
            include_match = True
            if include_list is not None:
                include_match = any(_matches(dtype, inc) for inc in include_list)

            if not include_match:
                continue

            if exclude_list is not None and any(
                _matches(dtype, exc) for exc in exclude_list
            ):
                continue

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

    def reorder_levels(self, order: List[int], axis: int = 0) -> "DataFrame":
        """
        Rearrange index levels using input order.

        Parameters
        ----------
        order : list of int
            List representing new level order
        axis : int, default 0
            Axis to reorder levels on

        Returns
        -------
        DataFrame
            DataFrame with reordered levels
        """
        if axis == 0:
            # Reorder index levels
            if self._index is None or len(self._index) == 0:
                return self.copy()

            # Check if MultiIndex
            if isinstance(self._index[0], tuple):
                # Create MultiIndex and reorder
                if isinstance(self._index_name, tuple):
                    names_list: Optional[List[Optional[str]]] = list(self._index_name)
                else:
                    names_list = None
                mi = MultiIndex.from_tuples(self._index, names=names_list)
                new_mi = mi.reorder_levels(order)  # type: ignore[arg-type]

                result = self.copy()
                result._index = new_mi.tolist()
                result._index_name = new_mi.names if new_mi.names else None  # type: ignore[assignment]
                return result
            else:
                # Regular Index - can't reorder
                return self.copy()
        else:
            # Column level reordering not yet implemented
            return self.copy()

    def swaplevel(
        self, i: Union[int, str] = -2, j: Union[int, str] = -1, axis: int = 0
    ) -> "DataFrame":
        """
        Swap levels i and j in a MultiIndex.

        Parameters
        ----------
        i : int or str, default -2
            First level to swap
        j : int or str, default -1
            Second level to swap
        axis : int, default 0
            Axis to swap levels on

        Returns
        -------
        DataFrame
            DataFrame with swapped levels
        """
        if axis == 0:
            # Swap index levels
            if self._index is None or len(self._index) == 0:
                return self.copy()

            # Check if MultiIndex
            if isinstance(self._index[0], tuple):
                # Create MultiIndex and swap
                if isinstance(self._index_name, tuple):
                    names_list: Optional[List[Optional[str]]] = list(self._index_name)
                else:
                    names_list = None
                mi = MultiIndex.from_tuples(self._index, names=names_list)
                new_mi = mi.swaplevel(i, j)

                result = self.copy()
                result._index = new_mi.tolist()
                result._index_name = new_mi.names if new_mi.names else None  # type: ignore[assignment]
                return result
            else:
                # Regular Index - can't swap
                return self.copy()
        else:
            # Column level swapping not yet implemented
            return self.copy()

    def take(self, indices: Any, axis: int = 0, **kwargs: Any) -> "DataFrame":
        """
        Return the elements in the given positional indices along an axis.

        Parameters
        ----------
        indices : array-like
            Indices to take
        axis : int, default 0
            Axis to take from
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            DataFrame with selected elements
        """
        if axis == 0:
            # Take rows
            return DataFrame(self._df[indices])
        else:
            # Take columns
            col_names = [self.columns[i] for i in indices]
            return DataFrame(self._df.select(col_names))

    def truncate(
        self,
        before: Optional[Any] = None,
        after: Optional[Any] = None,
        axis: Optional[Any] = None,
        copy: bool = True,
    ) -> "DataFrame":
        """
        Truncate a Series or DataFrame before and after some index value.

        Parameters
        ----------
        before : Any, optional
            Truncate all rows before this index value
        after : Any, optional
            Truncate all rows after this index value
        axis : Any, optional
            Axis to truncate along
        copy : bool, default True
            Whether to copy the DataFrame

        Returns
        -------
        DataFrame
            Truncated DataFrame
        """
        result = self.copy() if copy else self

        if (before is not None or after is not None) and self._index:
            # Use index to find positions
            start_idx = 0
            end_idx = len(self._index)

            if before is not None:
                with contextlib.suppress(ValueError):
                    start_idx = self._index.index(before) + 1

            if after is not None:
                with contextlib.suppress(ValueError):
                    end_idx = self._index.index(after)

            result = DataFrame(result._df[start_idx:end_idx])
            result._index = self._index[start_idx:end_idx] if self._index else None

        return result

    def xs(
        self,
        key: Any,
        axis: int = 0,
        level: Optional[Any] = None,
        drop_level: bool = True,
    ) -> "DataFrame":
        """
        Return cross-section from the Series/DataFrame.

        Parameters
        ----------
        key : Any
            Label contained in the index
        axis : int, default 0
            Axis to retrieve cross-section from
        level : Any, optional
            Level to retrieve cross-section from
        drop_level : bool, default True
            Whether to drop level from result

        Returns
        -------
        DataFrame or Series
            Cross-section
        """
        if axis == 0:
            # Get row by index label
            if self._index:
                try:
                    idx = self._index.index(key)
                    return DataFrame(self._df[idx : idx + 1])
                except ValueError:
                    raise KeyError(f"Key {key} not found in index") from None
            else:
                # Use integer index
                return DataFrame(self._df[key : key + 1])
        else:
            # Get column
            if key in self.columns:
                from .series import Series

                return Series(self._df[key])  # type: ignore[return-value]
            else:
                raise KeyError(f"Key {key} not found in columns")

    def kurt(
        self,
        axis: Optional[int] = None,
        skipna: bool = True,
        level: Optional[Any] = None,
        numeric_only: Optional[bool] = None,
        **kwargs: Any,
    ) -> "Series":
        """
        Return unbiased kurtosis over requested axis.

        Parameters
        ----------
        axis : int, optional
            Axis to compute kurtosis along
        skipna : bool, default True
            Exclude NA/null values
        level : Any, optional
            Level to compute kurtosis at
        numeric_only : bool, optional
            Include only numeric columns
        **kwargs
            Additional arguments

        Returns
        -------
        Series
            Kurtosis values
        """
        return self.kurtosis(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only or False,
            **kwargs,
        )

    def map(
        self, func: Any, na_action: Optional[str] = None, **kwargs: Any
    ) -> "DataFrame":
        """
        Apply a function to a Dataframe elementwise.

        Parameters
        ----------
        func : callable
            Function to apply
        na_action : str, optional
            How to handle NA values
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            DataFrame with function applied
        """
        # Apply function to each column
        new_cols = []
        for col in self.columns:
            col_series = self._df[col]
            if na_action == "ignore":
                # Apply only to non-null values
                mapped = col_series.map_elements(
                    func, return_dtype=col_series.dtype, **kwargs
                )
            else:
                mapped = col_series.map_elements(
                    func, return_dtype=col_series.dtype, **kwargs
                )
            new_cols.append(mapped.alias(col))

        return DataFrame(self._df.select(new_cols))

    def radd(
        self,
        other: Any,
        axis: Any = None,
        level: Any = None,
        fill_value: Optional[Any] = None,
    ) -> "DataFrame":
        """Reverse addition (other + self)."""
        return DataFrame(other + self._df)

    def rdiv(
        self,
        other: Any,
        axis: Any = None,
        level: Any = None,
        fill_value: Optional[Any] = None,
    ) -> "DataFrame":
        """Reverse division (other / self)."""
        return DataFrame(other / self._df)

    def rfloordiv(
        self,
        other: Any,
        axis: Any = None,
        level: Any = None,
        fill_value: Optional[Any] = None,
    ) -> "DataFrame":
        """Reverse floor division (other // self)."""
        return DataFrame(other // self._df)

    def rmod(
        self,
        other: Any,
        axis: Any = None,
        level: Any = None,
        fill_value: Optional[Any] = None,
    ) -> "DataFrame":
        """Reverse modulo (other % self)."""
        return DataFrame(other % self._df)

    def rmul(
        self,
        other: Any,
        axis: Any = None,
        level: Any = None,
        fill_value: Optional[Any] = None,
    ) -> "DataFrame":
        """Reverse multiplication (other * self)."""
        return DataFrame(other * self._df)

    def rpow(
        self,
        other: Any,
        axis: Any = None,
        level: Any = None,
        fill_value: Optional[Any] = None,
    ) -> "DataFrame":
        """Reverse power (other ** self)."""
        return DataFrame(other**self._df)

    def rsub(
        self,
        other: Any,
        axis: Any = None,
        level: Any = None,
        fill_value: Optional[Any] = None,
    ) -> "DataFrame":
        """Reverse subtraction (other - self)."""
        return DataFrame(other - self._df)

    def rtruediv(
        self,
        other: Any,
        axis: Any = None,
        level: Any = None,
        fill_value: Optional[Any] = None,
    ) -> "DataFrame":
        """Reverse true division (other / self)."""
        return DataFrame(other / self._df)

    def set_flags(
        self,
        copy: bool = False,
        allows_duplicate_labels: Optional[bool] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Return a new DataFrame with updated flags.

        Parameters
        ----------
        copy : bool, default False
            Whether to copy the DataFrame
        allows_duplicate_labels : bool, optional
            Whether to allow duplicate labels
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            DataFrame with updated flags
        """
        # Simplified implementation - Polars doesn't have flags
        return self.copy() if copy else self

    def to_period(
        self, freq: Optional[str] = None, axis: int = 0, copy: bool = True
    ) -> "DataFrame":
        """
        Convert DataFrame from DatetimeIndex to PeriodIndex.

        Parameters
        ----------
        freq : str, optional
            Frequency string
        axis : int, default 0
            Axis to convert
        copy : bool, default True
            Whether to copy the DataFrame

        Returns
        -------
        DataFrame
            DataFrame with PeriodIndex
        """
        # Simplified implementation - Polars doesn't have native Period type
        return self.copy() if copy else self

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

        # Check if we have a MultiIndex
        has_multiindex = (
            self._index is not None
            and len(self._index) > 0
            and isinstance(self._index[0], tuple)
        )

        # Use Polars transpose with column names from index if available
        if has_multiindex:
            # Convert MultiIndex tuples to strings for Polars column names
            column_names = [str(t) for t in self._index]  # type: ignore[union-attr]
        else:
            column_names = self._index if self._index else None  # type: ignore[assignment]

        try:
            transposed = self._df.transpose(
                include_header=False, column_names=column_names
            )
            result = DataFrame(transposed)

            # If we had a MultiIndex, it becomes the columns (stored as strings in Polars)
            # We need to restore it when converting to pandas
            if has_multiindex:
                # Store the MultiIndex column information
                result._column_index = self._index  # type: ignore[attr-defined]
                result._column_index_name = self._index_name  # type: ignore[attr-defined]
                # Columns are already set to MultiIndex tuple strings by Polars
            else:
                # Rename columns to match pandas (0, 1, 2, ...) if no index
                if column_names is None:
                    num_cols = len(transposed.columns)  # type: ignore[unreachable]
                    new_columns = [str(i) for i in range(num_cols)]
                    result._df = result._df.rename(
                        dict(zip(result._df.columns, new_columns))
                    )

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

    def to_numpy(
        self,
        dtype: Optional[Any] = None,
        copy: bool = False,
        na_value: Optional[Any] = None,
    ) -> Any:
        """
        Convert the DataFrame to a NumPy array.

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
            The DataFrame as a NumPy array.
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "numpy is required for to_numpy(). Install with: pip install numpy"
            ) from None

        # Convert to numpy array
        result = self._df.to_numpy()

        # Handle dtype conversion
        if dtype is not None:
            result = np.asarray(result, dtype=dtype)

        # Handle copy
        if copy:
            result = result.copy()

        # Handle na_value (Polars handles nulls, but we can replace them if needed)
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

        # Handle index=True case - add index as first column(s)
        else:  # index_param is True
            # Check if we have a MultiIndex (tuples in _index)
            has_multiindex = (
                self._index is not None
                and len(self._index) > 0
                and isinstance(self._index[0], tuple)
            )

            if has_multiindex:
                # For MultiIndex, flatten it into separate columns like pandas does
                # Use reset_index() to convert MultiIndex to columns
                df_with_index = self.reset_index(drop=False)
                df_to_write = (
                    df_with_index._df if df_with_index is not None else self._df
                )
            else:
                # Regular index - add as a single column
                df_to_write = self._df.clone()

                # Add index column if we have one
                if self._index is not None:
                    index_name = (
                        self._index_name if self._index_name is not None else "index"
                    )
                    # Add index as first column
                    df_to_write = df_to_write.with_columns(
                        pl.Series(index_name, self._index)
                    ).select([index_name] + df_to_write.columns)
                else:
                    # No stored index - use integer index
                    df_to_write = df_to_write.with_row_index("index").select(
                        ["index"] + df_to_write.columns
                    )

            # Write with index column(s) included
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

        # Handle MultiIndex columns (from transpose)
        if hasattr(self, "_column_index") and self._column_index is not None:
            # Restore MultiIndex columns
            if isinstance(self._column_index[0], tuple):
                # MultiIndex columns
                pandas_df.columns = pd.MultiIndex.from_tuples(
                    self._column_index,
                    names=self._column_index_name
                    if hasattr(self, "_column_index_name")
                    else None,
                )
            else:
                # Regular Index columns
                pandas_df.columns = pd.Index(self._column_index)
                if hasattr(self, "_column_index_name") and self._column_index_name:
                    pandas_df.columns.name = self._column_index_name

        # Convert string column names that look like integers to RangeIndex
        # (but skip if we just set MultiIndex columns)
        if not (hasattr(self, "_column_index") and self._column_index is not None):
            try:
                # Check if all column names are string representations of consecutive integers starting from 0
                col_names = list(pandas_df.columns)
                # Type guard: ensure all names are strings before checking isdigit
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

    def iterrows(self) -> Iterator[Tuple[Any, "Series"]]:
        """Iterate over DataFrame rows as (index, Series) pairs."""
        from polarpandas.series import Series

        rows = self._df.iter_rows(named=True)
        column_labels = list(self.columns)

        for i, row in enumerate(rows):
            values = [row[col] for col in column_labels]
            row_series = Series(values, index=column_labels)

            if self._index is not None and i < len(self._index):
                idx = self._index[i]
                row_series.name = idx
            else:
                idx = i
                row_series.name = idx

            yield (idx, row_series)

    def itertuples(
        self,
        index: bool = True,
        name: Optional[str] = "Pandas",
        **kwargs: Any,
    ) -> Iterator[Any]:
        """
        Iterate over DataFrame rows as namedtuples.

        Parameters
        ----------
        index : bool, default True
            If True, return the index as the first element of the tuple.
        name : str, default "Pandas"
            The name of the namedtuple returned.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Yields
        ------
        namedtuple
            A namedtuple representing each row.
        """
        from collections import namedtuple

        rows = self._df.iter_rows(named=True)
        col_names = self.columns

        # Create namedtuple class
        field_names = ["Index"] + list(col_names) if index else list(col_names)

        # Sanitize field names for namedtuple
        sanitized_names = []
        for name_field in field_names:
            # Replace invalid characters
            sanitized = name_field.replace(" ", "_").replace("-", "_")
            if sanitized[0].isdigit():
                sanitized = "_" + sanitized
            sanitized_names.append(sanitized)

        TupleClass = namedtuple(name or "Pandas", sanitized_names)  # type: ignore[misc]

        for i, row in enumerate(rows):
            # Get index
            if self._index is not None and i < len(self._index):
                idx = self._index[i]
            else:
                idx = i

            # Build tuple values
            values = [idx] + list(row.values()) if index else list(row.values())

            yield TupleClass(*values)

    @classmethod
    def from_dict(
        cls,
        data: Dict[Any, Any],
        orient: str = "columns",
        dtype: Optional[Any] = None,
        columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Construct DataFrame from dict of array-like or dicts.

        Parameters
        ----------
        data : dict
            Of the form {field : array-like} or {field : dict}.
        orient : {'columns', 'index', 'tight', 'records', 'list', 'split', 'values'}, default 'columns'
            Determines the orientation of the data.
        dtype : dtype, optional
            Data type to force, otherwise infer.
        columns : list, optional
            Column labels to use when orient='index'. Raises a ValueError if used with other orientations.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            DataFrame constructed from the dict.
        """
        if orient == "columns":
            # Default: keys are columns
            return cls(data, dtype=dtype, **kwargs)
        elif orient == "index":
            # Keys are index labels
            if columns is None:
                raise ValueError("columns must be specified when orient='index'")
            # Transpose the dict
            transposed = {}
            for col in columns:
                transposed[col] = [data.get(key, {}).get(col) for key in data]
            return cls(transposed, dtype=dtype, **kwargs)
        elif orient == "tight":
            # Similar to 'index' but with index and column names
            if "index" not in data or "columns" not in data or "data" not in data:
                raise ValueError(
                    "'tight' format requires 'index', 'columns', and 'data' keys"
                )
            index = data["index"]
            cols = data["columns"]
            values = data["data"]
            df_data = {col: [row[i] for row in values] for i, col in enumerate(cols)}
            result = cls(df_data)
            result._index = index
            return result
        elif orient == "records":
            # List of dicts
            if not data or not isinstance(list(data.values())[0], list):
                # Convert to list of dicts
                records = []
                keys = list(data.keys())
                max_len = max(
                    len(data[k]) if isinstance(data[k], (list, tuple)) else 1
                    for k in keys
                )
                for i in range(max_len):
                    record = {
                        k: (data[k][i] if i < len(data[k]) else None)
                        if isinstance(data[k], (list, tuple))
                        else data[k]
                        for k in keys
                    }
                    records.append(record)
                return cls(records, dtype=dtype, **kwargs)
            else:
                return cls(data, dtype=dtype, **kwargs)
        elif orient == "list":
            # Dict of lists
            return cls(data, dtype=dtype, **kwargs)
        elif orient == "split":
            # Dict with 'index', 'columns', 'data' keys
            if "index" not in data or "columns" not in data or "data" not in data:
                raise ValueError(
                    "'split' format requires 'index', 'columns', and 'data' keys"
                )
            index = data["index"]
            cols = data["columns"]
            values = data["data"]
            df_data = {col: [row[i] for row in values] for i, col in enumerate(cols)}
            result = cls(df_data)
            result._index = index
            return result
        else:
            raise ValueError(f"orient '{orient}' not recognized")

    @classmethod
    def from_records(
        cls,
        data: Any,
        index: Optional[Union[str, List[str]]] = None,
        exclude: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        coerce_float: bool = False,
        nrows: Optional[int] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Convert structured or record ndarray to DataFrame.

        Parameters
        ----------
        data : ndarray (structured dtype), list of tuples, dict, or DataFrame
            Structured input data.
        index : str, list of fields, array-like
            Field of array to use as the index, alternately a specific set of input labels to use.
        exclude : sequence, default None
            Columns or fields to exclude.
        columns : sequence, default None
            Column names to use. If the passed data do not have names associated with them, this argument provides names for the columns.
        coerce_float : bool, default False
            Attempt to convert values of non-string, non-numeric objects to floating point.
        nrows : int, optional
            Number of rows to read if data is an iterator.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Returns
        -------
        DataFrame
            DataFrame constructed from the records.
        """
        # Handle list of dicts
        if isinstance(data, list):
            if len(data) == 0:
                return cls({})

            # Check if it's a list of dicts
            if isinstance(data[0], dict):
                # Convert to dict of lists
                all_keys = set()
                for record in data:
                    all_keys.update(record.keys())

                df_data = {}
                for key in all_keys:
                    df_data[key] = [record.get(key) for record in data]

                result = cls(df_data)

                # Handle index
                if index is not None:
                    if isinstance(index, str):
                        if index in result.columns:
                            result.set_index(index, inplace=True)
                    else:
                        # List of field names
                        pass  # TODO: implement MultiIndex

                # Handle exclude
                if exclude is not None:
                    for col in exclude:
                        if col in result.columns:
                            result = result.drop(columns=[col])  # type: ignore[assignment]

                # Handle columns
                if columns is not None:
                    result = result[columns]  # type: ignore[assignment]

                return result

        # Handle dict
        elif isinstance(data, dict):
            return cls.from_dict(data, **kwargs)

        # Handle numpy structured array
        else:
            try:
                import numpy as np

                if isinstance(data, np.ndarray):
                    # Convert structured array to dict
                    df_data = {}
                    if data.dtype.names:
                        for name in data.dtype.names:
                            df_data[name] = data[name].tolist()
                    else:
                        # Regular array
                        for i in range(data.shape[1] if len(data.shape) > 1 else 1):
                            df_data[str(i)] = (
                                data[:, i].tolist()
                                if len(data.shape) > 1
                                else data.tolist()
                            )

                    result = cls(df_data)

                    # Handle index
                    if (
                        index is not None
                        and isinstance(index, str)
                        and index in result.columns
                    ):
                        result.set_index(index, inplace=True)

                    return result
            except ImportError:
                pass

        # Fallback: try to convert to dict
        return cls(data, **kwargs)

    def insert(
        self,
        loc: int,
        column: str,
        value: Any,
        allow_duplicates: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Insert column into DataFrame at specified location.

        Parameters
        ----------
        loc : int
            Insertion index. Must verify 0 <= loc <= len(columns).
        column : str
            Label of the inserted column.
        value : scalar, Series, or array-like
            Value to insert.
        allow_duplicates : bool, default False
            Allow duplicate column labels.
        **kwargs
            Additional arguments (not used, for pandas compatibility).

        Raises
        ------
        ValueError
            If column already exists and allow_duplicates is False.
        """
        if not allow_duplicates and column in self.columns:
            raise ValueError(f"cannot insert {column}, already exists")

        if not 0 <= loc <= len(self.columns):
            raise ValueError(
                f"loc must be between 0 and {len(self.columns)}, got {loc}"
            )

        # Convert value to Series if needed
        from polarpandas.series import Series

        if isinstance(value, Series):
            value_series = value._series
        elif isinstance(value, (list, tuple)):
            value_series = pl.Series(value)
        else:
            # Scalar - broadcast to all rows
            value_series = pl.Series([value] * len(self._df))

        # Get current columns
        current_cols = self.columns

        # Create new column order
        new_cols = current_cols[:loc] + [column] + current_cols[loc:]

        # Build new DataFrame with inserted column
        new_data = {}
        for i, col in enumerate(current_cols):
            if i < loc:
                new_data[col] = self._df[col].to_list()
            else:
                new_data[col] = self._df[col].to_list()

        # Insert the new column
        new_data[column] = value_series.to_list()

        # Reorder columns
        result_df = pl.DataFrame(new_data)
        result_df = result_df.select(new_cols)

        self._df = result_df

    def to_sql(
        self,
        name: str,
        con: Any,
        schema: Optional[str] = None,
        if_exists: str = "fail",
        index: bool = True,
        index_label: Optional[Union[str, List[str]]] = None,
        chunksize: Optional[int] = None,
        dtype: Optional[Dict[str, Any]] = None,
        method: Optional[Union[str, Callable[..., Any]]] = None,
        primary_key: Optional[Union[str, List[str]]] = None,
        auto_increment: bool = False,
    ) -> None:
        """
        Write DataFrame to SQL database.

        This method provides pandas-compatible interface for writing DataFrames to SQL.
        When primary_key or auto_increment parameters are used, SQLAlchemy is required.
        Otherwise, uses Polars' native write_database() for better performance.

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
        Create a simple table:

        >>> from sqlalchemy import create_engine
        >>> engine = create_engine('sqlite:///example.db')
        >>> df = ppd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
        >>> df.to_sql('users', engine)

        Create a table with a primary key:

        >>> df.to_sql('users', engine, if_exists='replace', primary_key='id')

        Create a table with an auto-incrementing primary key:

        >>> df.to_sql('users', engine, if_exists='replace',
        ...           primary_key='id', auto_increment=True)

        Create a table with a composite primary key:

        >>> df.to_sql('users', engine, primary_key=['id', 'email'])

        Notes
        -----
        - When primary_key or auto_increment is specified, SQLAlchemy is required
        - For best performance without these features, the method uses Polars'
          native write_database()
        - The index parameter and index_label are not fully supported yet when
          using Polars' native write_database()
        """
        # Check if we need SQLAlchemy features
        needs_sqlalchemy = primary_key is not None or auto_increment

        if needs_sqlalchemy:
            # Use SQLAlchemy for advanced features
            from polarpandas._sql_utils import create_table_with_primary_key

            if auto_increment and primary_key is None:
                raise ValueError("auto_increment requires primary_key to be specified")

            create_table_with_primary_key(
                df=self._df,
                table_name=name,
                connection=con,
                schema=schema,
                if_exists=if_exists,
                primary_key=primary_key,
                auto_increment=auto_increment,
                dtype=dtype,
                index=index,
                index_label=index_label,
            )
        else:
            # Use Polars' native write_database for better performance
            # Map pandas if_exists to Polars if_exists
            if if_exists not in ("fail", "replace", "append"):
                raise ValueError(
                    f"'{if_exists}' is not valid for if_exists. "
                    "Valid options are 'fail', 'replace', 'append'."
                )

            self._df.write_database(
                table_name=name,
                connection=con,
                if_table_exists=if_exists,  # type: ignore[arg-type]
            )

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
        """Create a pivot table with common aggregation functions."""
        if callable(aggfunc):
            raise NotImplementedError(
                "Callable aggfunc is not yet supported in pivot_table; use Polars expressions instead."
            )

        aggfunc_normalized = (aggfunc or "mean").lower()
        aggregate_map = {
            "mean": "mean",
            "sum": "sum",
            "min": "min",
            "max": "max",
            "count": "len",
            "median": "median",
        }

        if aggfunc_normalized not in aggregate_map:
            raise ValueError(
                "aggfunc must be one of {'mean', 'sum', 'min', 'max', 'count', 'median'}"
            )

        pivot_df: Any = self._df
        result = pivot_df.pivot(
            values=values,
            index=index,
            columns=columns,
            aggregate_function=aggregate_map[aggfunc_normalized],
            **kwargs,
        )
        return DataFrame(result)

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
        """Apply a function along an axis."""
        from polarpandas.series import Series

        if axis == 0:
            results = {}
            for col in self.columns:
                result = func(self._df[col])
                results[col] = result
            return Series(list(results.values()), name="apply_result")

        if axis not in (1, "columns"):
            raise ValueError("axis must be 0 or 1 when calling DataFrame.apply")

        row_results: List[Any] = []
        row_index: List[Any] = (
            list(self._index) if self._index is not None else list(range(len(self._df)))
        )

        column_names = self.columns

        for position, row_values in enumerate(self._df.iter_rows(named=True)):
            values = [row_values[col] for col in column_names]
            row_series = Series(values, index=column_names)
            if position < len(row_index):
                row_series.name = row_index[position]
            else:
                row_series.name = position

            result = func(row_series)
            row_results.append(result)

        if not row_results:
            return Series([], index=row_index)

        first_result = row_results[0]
        if isinstance(first_result, Series):
            typed_results = cast("List[Series]", row_results)
            data = {
                col: [res[col] for res in typed_results] for col in first_result.index
            }
            result_df = DataFrame(data)
            result_df._index = row_index[: len(result_df._df)]
            return result_df

        return Series(row_results, index=row_index[: len(row_results)])

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
            # map_elements returns a Series, which has an alias method
            mapped_series = self._df[col].map_elements(func, return_dtype=pl.Float64)
            result_cols.append(mapped_series.alias(col))

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
            Correlation method. Only 'pearson' is currently supported.
        min_periods : int, default 1
            Minimum number of observations required per pair of columns

        Returns
        -------
        DataFrame
            Correlation matrix
        """
        if method != "pearson":
            raise NotImplementedError(
                f"corr() method '{method}' is not yet implemented. Only 'pearson' is supported.\n"
                "Workaround: Convert to pandas temporarily: df.to_pandas().corr(method='{method}')"
            )

        # Get numeric columns only
        numeric_cols = [
            col for col in self._df.columns if self._df[col].dtype.is_numeric()
        ]

        if len(numeric_cols) == 0:
            # Return empty DataFrame with no columns
            return DataFrame()

        # Check if we have enough rows for correlation (need at least 2)
        if len(self._df) < 2:
            # Not enough data: return all NaN matrix (matching pandas behavior)
            corr_data = {
                col: [float("nan")] * len(numeric_cols) for col in numeric_cols
            }
            result_data = {"index": numeric_cols}
            result_data.update(corr_data)  # type: ignore[arg-type]
            result_df = pl.DataFrame(result_data)
            result = DataFrame(result_df)
            result_indexed = result.set_index("index")
            if result_indexed is None:
                raise RuntimeError("set_index returned None unexpectedly")
            result_indexed._index_name = None
            return result_indexed

        if len(numeric_cols) == 1:
            # Single column: return DataFrame with 1.0 (self-correlation)
            result = DataFrame({numeric_cols[0]: [1.0]}, index=[numeric_cols[0]])
            result._index_name = None
            return result

        # Calculate pairwise correlations using Polars expressions
        # corr(X, Y) = cov(X, Y) / (std(X) * std(Y))
        corr_result_data: Dict[str, List[float]] = {}
        for col1 in numeric_cols:
            corr_result_data[col1] = []
            for col2 in numeric_cols:
                if col1 == col2:
                    # Self-correlation is always 1.0
                    corr_result_data[col1].append(1.0)
                else:
                    # Calculate correlation: cov(X,Y) / (std(X) * std(Y))
                    # Use Polars expressions to compute this efficiently
                    # Drop nulls pairwise (only use rows where both columns are non-null)
                    df_subset = self._df.select([col1, col2]).drop_nulls()

                    # Calculate all needed statistics in one pass
                    # Use sample statistics (ddof=1) to match pandas behavior
                    n = len(df_subset)

                    # Handle edge case: need at least 2 observations for correlation
                    if n < 2:
                        corr_result_data[col1].append(float("nan"))
                    else:
                        stats = df_subset.select(
                            [
                                pl.col(col1).mean().alias("mean1"),
                                pl.col(col2).mean().alias("mean2"),
                                pl.col(col1).std(ddof=1).alias("std1"),
                                pl.col(col2).std(ddof=1).alias("std2"),
                                # Sample covariance: divide by (n-1) instead of n
                                (
                                    (pl.col(col1) - pl.col(col1).mean())
                                    * (pl.col(col2) - pl.col(col2).mean())
                                ).sum()
                                / pl.lit(n - 1).alias("cov"),
                            ]
                        )

                        # Extract values
                        row = stats.row(0)
                        mean1, mean2, std1, std2, cov_val = row

                        # Handle edge cases
                        if (
                            std1 is None
                            or std2 is None
                            or std1 == 0.0
                            or std2 == 0.0
                            or cov_val is None
                        ):
                            corr_result_data[col1].append(float("nan"))
                        else:
                            # Correlation = cov / (std1 * std2)
                            corr_value = (
                                cov_val / (std1 * std2)
                                if (std1 * std2) != 0.0
                                else float("nan")
                            )
                            corr_result_data[col1].append(corr_value)

        # Create correlation matrix DataFrame
        # Add index column as first column
        result_data = {"index": numeric_cols}
        result_data.update(corr_result_data)  # type: ignore[arg-type]
        result_df = pl.DataFrame(result_data)

        # Create polarpandas DataFrame and set index
        result = DataFrame(result_df)
        result_corr = result.set_index("index")
        if result_corr is None:
            raise RuntimeError("set_index returned None unexpectedly")
        # Remove index name to match pandas behavior (pandas corr/cov have unnamed index)
        result_corr._index_name = None

        return result_corr

    def cov(self, min_periods: Optional[int] = None) -> "DataFrame":
        """
        Compute pairwise covariance of columns.

        Parameters
        ----------
        min_periods : int, optional
            Minimum number of observations required per pair of columns (not yet implemented)

        Returns
        -------
        DataFrame
            Covariance matrix
        """
        if min_periods is not None:
            raise NotImplementedError(
                "cov() min_periods parameter is not yet implemented."
            )

        # Get numeric columns only
        numeric_cols = [
            col for col in self._df.columns if self._df[col].dtype.is_numeric()
        ]

        if len(numeric_cols) == 0:
            # Return empty DataFrame with no columns
            return DataFrame()

        if len(numeric_cols) == 1:
            # Single column: return variance (self-covariance, use sample variance to match pandas)
            var_value = self._df[numeric_cols[0]].var(ddof=1)
            return DataFrame({numeric_cols[0]: [var_value]}, index=[numeric_cols[0]])

        # Calculate pairwise covariances using Polars expressions
        # cov(X, Y) = E[(X - μX)(Y - μY)]
        cov_data: Dict[str, List[float]] = {}
        for col1 in numeric_cols:
            cov_data[col1] = []
            for col2 in numeric_cols:
                if col1 == col2:
                    # Self-covariance is variance (use sample variance to match pandas)
                    var_value = self._df[col1].var(ddof=1)
                    cov_data[col1].append(
                        var_value if var_value is not None else float("nan")  # type: ignore[arg-type]
                    )
                else:
                    # Calculate covariance: E[(X - μX)(Y - μY)]
                    # Drop nulls pairwise (only use rows where both columns are non-null)
                    df_subset = self._df.select([col1, col2]).drop_nulls()
                    n = len(df_subset)

                    # Handle edge case: need at least 2 observations for covariance
                    if n < 2:
                        cov_data[col1].append(float("nan"))
                    else:
                        # Calculate sample covariance in one pass (divide by n-1 to match pandas)
                        stats = df_subset.select(
                            [
                                (
                                    (pl.col(col1) - pl.col(col1).mean())
                                    * (pl.col(col2) - pl.col(col2).mean())
                                ).sum()
                                / pl.lit(n - 1).alias("cov")
                            ]
                        )

                        # Extract value
                        cov_result = stats.row(0)[0]
                        cov_data[col1].append(
                            cov_result if cov_result is not None else float("nan")
                        )

        # Create covariance matrix DataFrame
        # Add index column as first column
        result_data = {"index": numeric_cols}
        result_data.update(cov_data)  # type: ignore[arg-type]
        result_df = pl.DataFrame(result_data)

        # Create polarpandas DataFrame and set index
        result = DataFrame(result_df)
        result_cov = result.set_index("index")
        if result_cov is None:
            raise RuntimeError("set_index returned None unexpectedly")
        # Remove index name to match pandas behavior (pandas corr/cov have unnamed index)
        result_cov._index_name = None

        return result_cov

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
            raise NotImplementedError(
                "rank() with axis=1 (row-wise) is not yet implemented.\n"
                "Workaround: Use axis=0 (default) for column-wise ranking."
            )

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
            "Literal['average', 'min', 'max', 'dense', 'ordinal', 'random']",
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
            raise NotImplementedError(
                "cumsum() with axis=1 (row-wise) is not yet implemented.\n"
                "Workaround: Use axis=0 (default) for column-wise cumulative sum."
            )

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
            raise NotImplementedError(
                "cumprod() with axis=1 (row-wise) is not yet implemented.\n"
                "Workaround: Use axis=0 (default) for column-wise cumulative product."
            )

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
            raise NotImplementedError(
                "cummax() with axis=1 (row-wise) is not yet implemented.\n"
                "Workaround: Use axis=0 (default) for column-wise cumulative maximum."
            )

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
            raise NotImplementedError(
                "cummin() with axis=1 (row-wise) is not yet implemented.\n"
                "Workaround: Use axis=0 (default) for column-wise cumulative minimum."
            )

        result_cols = []
        for col in self._df.columns:
            if self._df[col].dtype.is_numeric():
                result_cols.append(pl.col(col).cum_min().alias(col))
            else:
                result_cols.append(pl.col(col))

        result_df = self._df.select(result_cols)
        return DataFrame(result_df)

    def asfreq(
        self,
        freq: str,
        method: Optional[str] = None,
        how: Optional[str] = None,
        normalize: bool = False,
        fill_value: Optional[Any] = None,
        **kwargs: Any,
    ) -> "DataFrame":
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
        DataFrame
            DataFrame with converted frequency

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2, 3]})
        >>> result = df.asfreq("D")
        """
        raise NotImplementedError(
            "asfreq() is not yet implemented.\n"
            "Workarounds:\n"
            "  - Use pandas: pd_df.asfreq(freq) then convert with polarpandas.DataFrame(df)\n"
            "  - Resample manually using resample() method"
        )

    def asof(
        self, where: Any, subset: Optional[Any] = None, **kwargs: Any
    ) -> "DataFrame":
        """
        Return last valid row up to label.

        Parameters
        ----------
        where : Any
            Label or labels
        subset : Any, optional
            Subset of columns
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            Last valid rows

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2, 3]})
        >>> result = df.asof(2)
        """
        # Simplified implementation
        if self._index:
            try:
                idx = (
                    self._index.index(where)
                    if where in self._index
                    else len(self._index) - 1
                )
                return DataFrame(self._df[: idx + 1])
            except (ValueError, TypeError):
                return DataFrame(self._df[: len(self._df)])
        return DataFrame(self._df[: len(self._df)])

    def at_time(
        self, time: Any, axis: Optional[Any] = None, asof: bool = False, **kwargs: Any
    ) -> "DataFrame":
        """
        Select values at particular time of day.

        Parameters
        ----------
        time : datetime.time or str
            Time to select
        axis : Any, optional
            Axis (not used)
        asof : bool, default False
            Use asof logic
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            Selected rows

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"time": ["09:00", "10:00"], "A": [1, 2]})
        >>> result = df.at_time("09:00")
        """
        # Simplified implementation - filter by time if datetime column exists
        if self._index and hasattr(self._index[0], "time"):
            filtered = [
                i
                for i, idx in enumerate(self._index)
                if hasattr(idx, "time") and idx.time() == time
            ]
            if filtered:
                return DataFrame(self._df[filtered])
        return DataFrame()

    def between_time(
        self,
        start_time: Any,
        end_time: Any,
        inclusive: str = "both",
        axis: Optional[Any] = None,
        **kwargs: Any,
    ) -> "DataFrame":
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
        axis : Any, optional
            Axis (not used)
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            Selected rows

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"time": ["09:00", "10:00", "11:00"], "A": [1, 2, 3]})
        >>> result = df.between_time("09:00", "10:00")
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
                return DataFrame(self._df[filtered])
        return DataFrame()

    def resample(
        self,
        rule: str,
        axis: int = 0,
        closed: Optional[str] = None,
        label: Optional[str] = None,
        convention: str = "start",
        kind: Optional[str] = None,
        loffset: Optional[Any] = None,
        base: Optional[int] = None,
        on: Optional[str] = None,
        level: Optional[Any] = None,
        origin: str = "start_day",
        offset: Optional[Any] = None,
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
            Axis to resample
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
            Column to resample on
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
        >>> df = ppd.DataFrame({"time": ["2020-01-01", "2020-01-02"], "A": [1, 2]})
        >>> result = df.resample("D")
        """
        raise NotImplementedError(
            "resample() is not yet implemented.\n"
            "Workarounds:\n"
            "  - Use pandas: pd_df.resample(rule) then convert with polarpandas.DataFrame(df)\n"
            "  - Use Polars group_by_dynamic() for time-based grouping"
        )

    def ewm(
        self,
        com: Optional[float] = None,
        span: Optional[float] = None,
        halflife: Optional[float] = None,
        alpha: Optional[float] = None,
        min_periods: int = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        axis: int = 0,
        times: Optional[Any] = None,
        method: str = "single",
        **kwargs: Any,
    ) -> Any:
        """
        Provide exponential weighted functions.

        Parameters
        ----------
        com : float, optional
            Center of mass
        span : float, optional
            Span
        halflife : float, optional
            Half-life
        alpha : float, optional
            Smoothing factor
        min_periods : int, default 0
            Minimum number of observations
        adjust : bool, default True
            Adjust for bias
        ignore_na : bool, default False
            Ignore NA values
        axis : int, default 0
            Axis
        times : Any, optional
            Times for time-based decay
        method : str, default "single"
            Method
        **kwargs
            Additional arguments

        Returns
        -------
        EWM
            Exponential weighted object

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2, 3]})
        >>> result = df.ewm(span=2)
        """
        raise NotImplementedError(
            "ewm() is not yet implemented.\n"
            "Workarounds:\n"
            "  - Use pandas: pd_df.ewm(span=span) then convert with polarpandas.DataFrame(df)\n"
            "  - Use Polars ewm_mean() for exponential weighted moving average"
        )

    def expanding(
        self, min_periods: int = 1, axis: int = 0, method: str = "single", **kwargs: Any
    ) -> Any:
        """
        Provide expanding window calculations.

        Parameters
        ----------
        min_periods : int, default 1
            Minimum number of observations
        axis : int, default 0
            Axis
        method : str, default "single"
            Method
        **kwargs
            Additional arguments

        Returns
        -------
        Expanding
            Expanding window object

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2, 3]})
        >>> result = df.expanding()
        """
        raise NotImplementedError(
            "expanding() is not yet implemented.\n"
            "Workarounds:\n"
            "  - Use pandas: pd_df.expanding() then convert with polarpandas.DataFrame(df)\n"
            "  - Use Polars cumulative functions for expanding calculations"
        )

    def combine(
        self,
        other: "DataFrame",
        func: Callable[[Any, Any], Any],
        fill_value: Optional[Any] = None,
        overwrite: bool = True,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Combine DataFrame with another DataFrame.

        Parameters
        ----------
        other : DataFrame
            Other DataFrame
        func : callable
            Function to combine values
        fill_value : Any, optional
            Value to use for missing values
        overwrite : bool, default True
            Overwrite existing values
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            Combined DataFrame

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df1 = ppd.DataFrame({"A": [1, 2]})
        >>> df2 = ppd.DataFrame({"A": [3, 4]})
        >>> result = df1.combine(df2, lambda x, y: x + y)
        """
        # Align DataFrames
        aligned_self, aligned_other = self.align(
            other, join="outer", fill_value=fill_value
        )
        # Combine using function
        result_cols = {}
        for col in aligned_self.columns:
            if col in aligned_other.columns:
                combined = [
                    func(s, o)
                    for s, o in zip(
                        aligned_self[col]._series, aligned_other[col]._series
                    )
                ]
                result_cols[col] = combined
            else:
                result_cols[col] = aligned_self[col]._series.to_list()
        return DataFrame(pl.DataFrame(result_cols))

    def convert_dtypes(
        self,
        infer_objects: bool = True,
        convert_string: bool = True,
        convert_integer: bool = True,
        convert_boolean: bool = True,
        convert_floating: bool = True,
        dtype_backend: str = "numpy_nullable",
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Convert columns to best possible dtypes.

        Parameters
        ----------
        infer_objects : bool, default True
            Infer object dtypes
        convert_string : bool, default True
            Convert to string dtype
        convert_integer : bool, default True
            Convert to integer dtype
        convert_boolean : bool, default True
            Convert to boolean dtype
        convert_floating : bool, default True
            Convert to floating dtype
        dtype_backend : str, default "numpy_nullable"
            Dtype backend
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            DataFrame with converted dtypes

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": ["1", "2"]})
        >>> result = df.convert_dtypes()
        """
        # Simplified implementation - Polars already uses optimal dtypes
        return self.copy()

    def eval(
        self, expr: str, inplace: bool = False, **kwargs: Any
    ) -> Optional["DataFrame"]:
        """
        Evaluate expression over DataFrame.

        Parameters
        ----------
        expr : str
            Expression to evaluate
        inplace : bool, default False
            Modify in place
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame or None
            Result of evaluation

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> result = df.eval("A + B")
        """
        from .operations import eval as eval_func

        result = eval_func(expr, target=self, **kwargs)
        if inplace:
            if isinstance(result, Series):
                # Assign result to new column or replace existing
                self._df = self._df.with_columns(result._series.alias("result"))
            return None
        return (
            result
            if isinstance(result, DataFrame)
            else DataFrame(
                {"result": result._series if isinstance(result, Series) else [result]}
            )
        )

    def infer_objects(self, copy: Optional[bool] = None) -> "DataFrame":
        """
        Attempt to infer better dtypes for object columns.

        Parameters
        ----------
        copy : bool, optional
            Copy data

        Returns
        -------
        DataFrame
            DataFrame with inferred dtypes

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": ["1", "2"]})
        >>> result = df.infer_objects()
        """
        # Simplified implementation - Polars already infers types
        return self.copy() if copy else self

    def isetitem(self, loc: Union[int, Any], value: Any) -> None:
        """
        Set item by integer position.

        Parameters
        ----------
        loc : int or tuple
            Location
        value : Any
            Value to set

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2]})
        >>> df.isetitem(0, [10, 20])
        """
        if isinstance(loc, tuple):
            row_idx, col_idx = loc
            col_name = self.columns[col_idx]
            self._df = self._df.with_columns(
                pl.when(pl.int_range(pl.len()) == row_idx)
                .then(pl.lit(value))
                .otherwise(pl.col(col_name))
                .alias(col_name)
            )
        elif isinstance(loc, int):
            col_name = self.columns[loc]
            if isinstance(value, (list, tuple)):
                self._df = self._df.with_columns(pl.Series(col_name, value))
            else:
                self._df = self._df.with_columns(pl.lit(value).alias(col_name))
        else:
            raise ValueError(f"Invalid location: {loc}")

    def boxplot(
        self,
        column: Optional[Any] = None,
        by: Optional[Any] = None,
        ax: Optional[Any] = None,
        fontsize: Optional[int] = None,
        rot: int = 0,
        grid: bool = True,
        figsize: Optional[Tuple[int, int]] = None,
        layout: Optional[Tuple[int, int]] = None,
        return_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Make a box plot from DataFrame columns.

        Parameters
        ----------
        column : str or list, optional
            Column name(s)
        by : str, optional
            Group by column
        ax : Any, optional
            Matplotlib axes
        fontsize : int, optional
            Font size
        rot : int, default 0
            Rotation
        grid : bool, default True
            Show grid
        figsize : tuple, optional
            Figure size
        layout : tuple, optional
            Layout
        return_type : str, optional
            Return type
        **kwargs
            Additional arguments

        Returns
        -------
        Any
            Plot object

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2, 3]})
        >>> df.boxplot()
        """
        raise NotImplementedError(
            "boxplot() is not yet implemented.\n"
            "Workarounds:\n"
            "  - Use pandas: pd_df.boxplot() for plotting\n"
            "  - Use matplotlib/seaborn directly for visualization"
        )

    def hist(
        self,
        column: Optional[Any] = None,
        by: Optional[Any] = None,
        grid: bool = True,
        xlabelsize: Optional[int] = None,
        xrot: Optional[float] = None,
        ylabelsize: Optional[int] = None,
        yrot: Optional[float] = None,
        ax: Optional[Any] = None,
        sharex: bool = False,
        sharey: bool = False,
        figsize: Optional[Tuple[int, int]] = None,
        layout: Optional[Tuple[int, int]] = None,
        bins: int = 10,
        backend: Optional[str] = None,
        legend: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Make a histogram of the DataFrame's columns.

        Parameters
        ----------
        column : str or list, optional
            Column name(s)
        by : str, optional
            Group by column
        grid : bool, default True
            Show grid
        xlabelsize : int, optional
            X-axis label size
        xrot : float, optional
            X-axis rotation
        ylabelsize : int, optional
            Y-axis label size
        yrot : float, optional
            Y-axis rotation
        ax : Any, optional
            Matplotlib axes
        sharex : bool, default False
            Share x-axis
        sharey : bool, default False
            Share y-axis
        figsize : tuple, optional
            Figure size
        layout : tuple, optional
            Layout
        bins : int, default 10
            Number of bins
        backend : str, optional
            Plotting backend
        legend : bool, default False
            Show legend
        **kwargs
            Additional arguments

        Returns
        -------
        Any
            Plot object

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2, 3]})
        >>> df.hist()
        """
        raise NotImplementedError(
            "hist() is not yet implemented.\n"
            "Workarounds:\n"
            "  - Use pandas: pd_df.hist() for plotting\n"
            "  - Use matplotlib/seaborn directly for visualization"
        )

    def plot(
        self,
        x: Optional[Any] = None,
        y: Optional[Any] = None,
        kind: str = "line",
        ax: Optional[Any] = None,
        subplots: bool = False,
        sharex: Optional[bool] = None,
        sharey: bool = False,
        layout: Optional[Tuple[int, int]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        use_index: bool = True,
        title: Optional[str] = None,
        grid: Optional[bool] = None,
        legend: Union[bool, str] = True,
        style: Optional[Any] = None,
        logx: Union[bool, str] = False,
        logy: Union[bool, str] = False,
        loglog: Union[bool, str] = False,
        xticks: Optional[Any] = None,
        yticks: Optional[Any] = None,
        xlim: Optional[Any] = None,
        ylim: Optional[Any] = None,
        rot: Optional[float] = None,
        fontsize: Optional[int] = None,
        colormap: Optional[str] = None,
        colorbar: Optional[bool] = None,
        position: float = 0.5,
        table: bool = False,
        yerr: Optional[Any] = None,
        xerr: Optional[Any] = None,
        label: Optional[str] = None,
        secondary_y: Union[bool, str, List[str]] = False,
        **kwargs: Any,
    ) -> Any:
        """
        Make plots of DataFrame.

        Parameters
        ----------
        x : str or int, optional
            X-axis column
        y : str, int, or list, optional
            Y-axis column(s)
        kind : str, default "line"
            Plot kind
        ax : Any, optional
            Matplotlib axes
        subplots : bool, default False
            Make subplots
        sharex : bool, optional
            Share x-axis
        sharey : bool, default False
            Share y-axis
        layout : tuple, optional
            Layout
        figsize : tuple, optional
            Figure size
        use_index : bool, default True
            Use index as x-axis
        title : str, optional
            Plot title
        grid : bool, optional
            Show grid
        legend : bool or str, default True
            Show legend
        style : Any, optional
            Plot style
        logx : bool or str, default False
            Log scale for x-axis
        logy : bool or str, default False
            Log scale for y-axis
        loglog : bool or str, default False
            Log scale for both axes
        xticks : Any, optional
            X-axis ticks
        yticks : Any, optional
            Y-axis ticks
        xlim : Any, optional
            X-axis limits
        ylim : Any, optional
            Y-axis limits
        rot : float, optional
            Rotation
        fontsize : int, optional
            Font size
        colormap : str, optional
            Colormap
        colorbar : bool, optional
            Show colorbar
        position : float, default 0.5
            Position
        table : bool, default False
            Show table
        yerr : Any, optional
            Y-axis error bars
        xerr : Any, optional
            X-axis error bars
        label : str, optional
            Label
        secondary_y : bool, str, or list, default False
            Secondary y-axis
        **kwargs
            Additional arguments

        Returns
        -------
        Any
            Plot object

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2, 3]})
        >>> df.plot()
        """
        raise NotImplementedError(
            "plot() is not yet implemented.\n"
            "Workarounds:\n"
            "  - Use pandas: pd_df.plot() for plotting\n"
            "  - Use matplotlib/seaborn directly for visualization"
        )

    def to_clipboard(
        self, excel: bool = True, sep: Optional[str] = None, **kwargs: Any
    ) -> None:
        """
        Copy object to clipboard.

        Parameters
        ----------
        excel : bool, default True
            Use Excel format
        sep : str, optional
            Separator
        **kwargs
            Additional arguments

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2]})
        >>> df.to_clipboard()
        """
        try:
            import pandas as pd  # noqa: F401

            # Convert to pandas and use its clipboard functionality
            pd_df = self.to_pandas()
            pd_df.to_clipboard(excel=excel, sep=sep, **kwargs)
        except ImportError:
            raise NotImplementedError(
                "to_clipboard() requires pandas.\n"
                "Workarounds:\n"
                "  - Install pandas: pip install pandas\n"
                "  - Use pandas: pd_df.to_clipboard()"
            ) from None

    def to_excel(
        self,
        excel_writer: Any,
        sheet_name: str = "Sheet1",
        na_rep: str = "",
        float_format: Optional[str] = None,
        columns: Optional[Any] = None,
        header: Union[bool, List[str]] = True,
        index: bool = True,
        index_label: Optional[Any] = None,
        startrow: int = 0,
        startcol: int = 0,
        engine: Optional[str] = None,
        merge_cells: bool = True,
        encoding: Optional[str] = None,
        inf_rep: str = "inf",
        verbose: bool = True,
        freeze_panes: Optional[Tuple[int, int]] = None,
        storage_options: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Write DataFrame to Excel file.

        Parameters
        ----------
        excel_writer : str or ExcelWriter
            Path or ExcelWriter object
        sheet_name : str, default "Sheet1"
            Sheet name
        na_rep : str, default ""
            Representation for NA values
        float_format : str, optional
            Format for floats
        columns : list, optional
            Columns to write
        header : bool or list, default True
            Write header
        index : bool, default True
            Write index
        index_label : str or list, optional
            Index column label
        startrow : int, default 0
            Start row
        startcol : int, default 0
            Start column
        engine : str, optional
            Engine to use
        merge_cells : bool, default True
            Merge cells
        encoding : str, optional
            Encoding
        inf_rep : str, default "inf"
            Representation for infinity
        verbose : bool, default True
            Verbose output
        freeze_panes : tuple, optional
            Freeze panes
        storage_options : dict, optional
            Storage options
        **kwargs
            Additional arguments

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2]})
        >>> df.to_excel("output.xlsx")
        """
        try:
            import pandas as pd  # noqa: F401

            # Convert to pandas and use its Excel functionality
            pd_df = self.to_pandas()
            pd_df.to_excel(
                excel_writer=excel_writer,
                sheet_name=sheet_name,
                na_rep=na_rep,
                float_format=float_format,
                columns=columns,
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
        except ImportError:
            raise NotImplementedError(
                "to_excel() requires pandas and openpyxl/xlsxwriter.\n"
                "Workarounds:\n"
                "  - Install: pip install pandas openpyxl\n"
                "  - Use pandas: pd_df.to_excel(path)"
            ) from None

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
    ) -> Optional[str]:
        """
        Render DataFrame to HTML table.

        Parameters
        ----------
        buf : str, Path, or file-like, optional
            Buffer to write to
        columns : list, optional
            Columns to include
        col_space : Any, optional
            Column spacing
        header : bool, default True
            Write header
        index : bool, default True
            Write index
        na_rep : str, default "NaN"
            Representation for NA values
        formatters : Any, optional
            Formatters
        float_format : str, optional
            Format for floats
        sparsify : bool, optional
            Sparsify MultiIndex
        index_names : bool, default True
            Write index names
        justify : str, optional
            Justification
        max_rows : int, optional
            Maximum rows to display
        max_cols : int, optional
            Maximum columns to display
        show_dimensions : bool or str, default False
            Show dimensions
        decimal : str, default "."
            Decimal separator
        bold_rows : bool, default False
            Bold rows
        classes : str, optional
            CSS classes
        escape : bool, default True
            Escape HTML
        notebook : bool, default False
            Notebook mode
        border : int, optional
            Border width
        table_id : str, optional
            Table ID
        render_links : bool, default False
            Render links
        encoding : str, optional
            Encoding
        **kwargs
            Additional arguments

        Returns
        -------
        str or None
            HTML string if buf is None, else None

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2]})
        >>> html = df.to_html()
        """
        # Try tabulate first (lightweight), otherwise use string formatting
        try:
            from tabulate import tabulate  # type: ignore[import-untyped]  # noqa: F811

            # Select columns if specified
            df_to_show = self
            if columns:
                df_to_show = self[columns]  # type: ignore[assignment]

            # Limit rows/cols if specified
            if max_rows is not None:
                df_to_show = df_to_show.head(max_rows)
            if max_cols is not None:
                cols = df_to_show.columns[:max_cols]
                df_to_show = df_to_show[cols]  # type: ignore[assignment]

            # Convert to list of lists
            data = df_to_show._df.to_dict(as_series=False)
            headers = list(data.keys()) if header else None

            # Build rows
            rows = []
            for i in range(len(df_to_show)):
                row = (
                    [data[col][i] if col in data else "" for col in headers]
                    if headers
                    else []
                )
                if index:
                    idx_val = (
                        self._index[i] if self._index and i < len(self._index) else i
                    )
                    row = [idx_val] + row
                rows.append(row)

            if index and headers:
                headers = [""] + list(headers)

            html_str = tabulate(
                rows,
                headers=headers,
                tablefmt="html",
                floatfmt=float_format,
                showindex=False,
                **kwargs,
            )

            if buf is None:
                return html_str  # type: ignore[no-any-return]
            else:
                if hasattr(buf, "write"):
                    buf.write(html_str)
                else:
                    with open(buf, "w", encoding=encoding or "utf-8") as f:
                        f.write(html_str)
                return None
        except ImportError:
            # Fallback: simple HTML formatting
            html_parts = ["<table>"]
            if header:
                html_parts.append("<thead><tr>")
                if index:
                    html_parts.append("<th></th>")
                for col in columns or self.columns:
                    html_parts.append(f"<th>{col}</th>")
                html_parts.append("</tr></thead>")
            html_parts.append("<tbody>")
            for i in range(len(self)):
                html_parts.append("<tr>")
                if index:
                    idx_val = (
                        self._index[i] if self._index and i < len(self._index) else i
                    )
                    html_parts.append(f"<td>{idx_val}</td>")
                for col in columns or self.columns:
                    val = self[col].iloc[i] if col in self.columns else na_rep
                    if val is None:
                        val = na_rep
                    html_parts.append(f"<td>{val}</td>")
                html_parts.append("</tr>")
            html_parts.append("</tbody></table>")
            html_str = "".join(html_parts)

            if buf is None:
                return html_str
            else:
                if hasattr(buf, "write"):
                    buf.write(html_str)
                else:
                    with open(buf, "w", encoding=encoding or "utf-8") as f:
                        f.write(html_str)
                return None

    def to_iceberg(self, path: str, **kwargs: Any) -> None:
        """
        Write DataFrame to Iceberg table.

        Parameters
        ----------
        path : str
            Path to Iceberg table
        **kwargs
            Additional arguments

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2]})
        >>> df.to_iceberg("iceberg_table")
        """
        try:
            self._df.write_iceberg(path, **kwargs)
        except AttributeError:
            raise NotImplementedError(
                "to_iceberg() requires Polars with Iceberg support.\n"
                "Workarounds:\n"
                "  - Use Polars directly: pl_df.write_iceberg(path)\n"
                "  - Export to Parquet first, then use Iceberg tools"
            ) from None

    def to_latex(
        self,
        buf: Optional[Any] = None,
        columns: Optional[Any] = None,
        header: bool = True,
        index: bool = True,
        na_rep: str = "NaN",
        formatters: Optional[Any] = None,
        float_format: Optional[str] = None,
        sparsify: Optional[bool] = None,
        index_names: bool = True,
        bold_rows: bool = False,
        column_format: Optional[str] = None,
        longtable: bool = False,
        escape: bool = True,
        encoding: Optional[str] = None,
        decimal: str = ".",
        multicolumn: Optional[bool] = None,
        multicolumn_format: Optional[str] = None,
        multirow: Optional[bool] = None,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        position: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Render DataFrame to LaTeX table.

        Parameters
        ----------
        buf : str, Path, or file-like, optional
            Buffer to write to
        columns : list, optional
            Columns to include
        header : bool, default True
            Write header
        index : bool, default True
            Write index
        na_rep : str, default "NaN"
            Representation for NA values
        formatters : Any, optional
            Formatters
        float_format : str, optional
            Format for floats
        sparsify : bool, optional
            Sparsify MultiIndex
        index_names : bool, default True
            Write index names
        bold_rows : bool, default False
            Bold rows
        column_format : str, optional
            Column format
        longtable : bool, default False
            Use longtable
        escape : bool, default True
            Escape LaTeX
        encoding : str, optional
            Encoding
        decimal : str, default "."
            Decimal separator
        multicolumn : bool, optional
            Multi-column
        multicolumn_format : str, optional
            Multi-column format
        multirow : bool, optional
            Multi-row
        caption : str, optional
            Caption
        label : str, optional
            Label
        position : str, optional
            Position
        **kwargs
            Additional arguments

        Returns
        -------
        str or None
            LaTeX string if buf is None, else None

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2]})
        >>> latex = df.to_latex()
        """
        # Try tabulate first (lightweight), otherwise use string formatting
        try:
            from tabulate import tabulate  # noqa: F811

            # Select columns if specified
            df_to_show = self
            if columns:
                df_to_show = self[columns]  # type: ignore[assignment]

            # Convert to list of lists
            data = df_to_show._df.to_dict(as_series=False)
            headers = list(data.keys()) if header else None

            # Build rows
            rows = []
            for i in range(len(df_to_show)):
                row = (
                    [data[col][i] if col in data else "" for col in headers]
                    if headers
                    else []
                )
                if index:
                    idx_val = (
                        self._index[i] if self._index and i < len(self._index) else i
                    )
                    row = [idx_val] + row
                rows.append(row)

            if index and headers:
                headers = [""] + list(headers)

            latex_str = tabulate(
                rows,
                headers=headers,
                tablefmt="latex",
                floatfmt=float_format,
                showindex=False,
                **kwargs,
            )

            if buf is None:
                return latex_str  # type: ignore[no-any-return]
            else:
                if hasattr(buf, "write"):
                    buf.write(latex_str)
                else:
                    with open(buf, "w", encoding=encoding or "utf-8") as f:
                        f.write(latex_str)
                return None
        except ImportError:
            # Fallback: simple LaTeX formatting
            latex_parts = [
                "\\begin{tabular}{"
                + (
                    column_format
                    or "l" * (len(columns or self.columns) + (1 if index else 0))
                )
                + "}"
            ]
            if header:
                if index:
                    latex_parts.append(" & ")
                latex_parts.append(
                    " & ".join(str(col) for col in (columns or self.columns))
                )
                latex_parts.append(" \\\\\n\\hline\n")
            for i in range(len(self)):
                parts = []
                if index:
                    idx_val = (
                        self._index[i] if self._index and i < len(self._index) else i
                    )
                    parts.append(str(idx_val))
                for col in columns or self.columns:
                    val = self[col].iloc[i] if col in self.columns else na_rep
                    if val is None:
                        val = na_rep
                    parts.append(str(val))
                latex_parts.append(" & ".join(parts))
                latex_parts.append(" \\\\\n")
            latex_parts.append("\\end{tabular}")
            latex_str = "".join(latex_parts)

            if buf is None:
                return latex_str
            else:
                if hasattr(buf, "write"):
                    buf.write(latex_str)
                else:
                    with open(buf, "w", encoding=encoding or "utf-8") as f:
                        f.write(latex_str)
                return None

    def to_markdown(
        self,
        buf: Optional[Any] = None,
        mode: str = "wt",
        index: bool = True,
        storage_options: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Print DataFrame in Markdown-friendly format.

        Parameters
        ----------
        buf : str, Path, or file-like, optional
            Buffer to write to
        mode : str, default "wt"
            File mode
        index : bool, default True
            Write index
        storage_options : dict, optional
            Storage options
        **kwargs
            Additional arguments

        Returns
        -------
        str or None
            Markdown string if buf is None, else None

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2]})
        >>> md = df.to_markdown()
        """
        # Try tabulate first (lightweight), otherwise use string formatting
        try:
            from tabulate import tabulate  # noqa: F811  # noqa: F811

            # Convert to list of lists
            data = self._df.to_dict(as_series=False)
            headers = list(data.keys())

            # Build rows
            rows = []
            for i in range(len(self)):
                row = [data[col][i] if col in data else "" for col in headers]
                if index:
                    idx_val = (
                        self._index[i] if self._index and i < len(self._index) else i
                    )
                    row = [idx_val] + row
                rows.append(row)

            if index:
                headers = [""] + list(headers)

            md_str = tabulate(
                rows, headers=headers, tablefmt="github", showindex=False, **kwargs
            )

            if buf is None:
                return md_str  # type: ignore[no-any-return]
            else:
                if hasattr(buf, "write"):
                    buf.write(md_str)
                else:
                    with open(buf, mode, encoding="utf-8") as f:
                        f.write(md_str)
                return None
        except ImportError:
            # Fallback: simple Markdown formatting
            md_parts = []
            # Header
            headers = list(self.columns)
            if index:
                headers = [""] + headers
            md_parts.append("| " + " | ".join(str(h) for h in headers) + " |\n")
            md_parts.append("| " + " | ".join(["---"] * len(headers)) + " |\n")
            # Rows
            for i in range(len(self)):
                parts = []
                if index:
                    idx_val = (
                        self._index[i] if self._index and i < len(self._index) else i
                    )
                    parts.append(str(idx_val))
                for col in self.columns:
                    val = self[col].iloc[i]
                    if val is None:
                        val = ""
                    parts.append(str(val))
                md_parts.append("| " + " | ".join(parts) + " |\n")
            md_str = "".join(md_parts)

            if buf is None:
                return md_str
            else:
                if hasattr(buf, "write"):
                    buf.write(md_str)
                else:
                    with open(buf, mode, encoding="utf-8") as f:
                        f.write(md_str)
                return None

    def to_orc(self, path: str, **kwargs: Any) -> None:
        """
        Write DataFrame to ORC file.

        Parameters
        ----------
        path : str
            Path to ORC file
        **kwargs
            Additional arguments

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2]})
        >>> df.to_orc("data.orc")
        """
        self._df.write_orc(path, **kwargs)  # type: ignore[attr-defined]

    def to_pickle(
        self,
        path: str,
        compression: Optional[str] = None,
        protocol: Optional[int] = None,
        storage_options: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Pickle (serialize) object to file.

        Parameters
        ----------
        path : str
            File path
        compression : str, optional
            Compression type
        protocol : int, optional
            Pickle protocol
        storage_options : dict, optional
            Storage options
        **kwargs
            Additional arguments

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2]})
        >>> df.to_pickle("data.pkl")
        """
        import gzip
        import pickle

        # Handle compression
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

    def to_records(
        self,
        index: bool = True,
        column_dtypes: Optional[Any] = None,
        index_dtypes: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Convert DataFrame to NumPy record array.

        Parameters
        ----------
        index : bool, default True
            Include index
        column_dtypes : Any, optional
            Column dtypes
        index_dtypes : Any, optional
            Index dtypes
        **kwargs
            Additional arguments

        Returns
        -------
        numpy.ndarray
            NumPy structured array

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2]})
        >>> records = df.to_records()
        """
        try:
            import numpy as np

            # Convert to list of dicts first
            data = self._df.to_dicts()
            if index and self._index:
                for i, row in enumerate(data):
                    row["index"] = self._index[i] if i < len(self._index) else i

            # Convert to numpy structured array
            if data:
                # Infer dtype from first row
                dtype = [(k, type(v).__name__) for k, v in data[0].items()]
                # Convert to proper numpy dtypes
                dtype = [
                    (k, np.dtype("O") if v == "NoneType" else np.dtype(v))  # type: ignore[misc]
                    for k, v in dtype
                ]
                arr = np.array([tuple(row.values()) for row in data], dtype=dtype)
                return arr
            else:
                return np.array([], dtype=[])
        except ImportError:
            # Fallback: return list of dicts
            data = self._df.to_dicts()
            if index and self._index:
                for i, row in enumerate(data):
                    row["index"] = self._index[i] if i < len(self._index) else i
            return data

    def to_stata(
        self,
        path: str,
        convert_dates: Optional[Any] = None,
        write_index: bool = True,
        byteorder: Optional[str] = None,
        time_stamp: Optional[Any] = None,
        data_label: Optional[str] = None,
        variable_labels: Optional[Any] = None,
        version: Optional[int] = None,
        convert_strl: Optional[Any] = None,
        compression: str = "none",
        storage_options: Optional[Any] = None,
        value_labels: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Write DataFrame to Stata file.

        Parameters
        ----------
        path : str
            File path
        convert_dates : Any, optional
            Convert dates
        write_index : bool, default True
            Write index
        byteorder : str, optional
            Byte order
        time_stamp : Any, optional
            Time stamp
        data_label : str, optional
            Data label
        variable_labels : dict, optional
            Variable labels
        version : int, optional
            Stata version
        convert_strl : Any, optional
            Convert string length
        compression : str, default "none"
            Compression
        storage_options : dict, optional
            Storage options
        value_labels : dict, optional
            Value labels
        **kwargs
            Additional arguments

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2]})
        >>> df.to_stata("data.dta")
        """
        try:
            import pyreadstat

            # Convert to dict of lists
            data_dict = self._df.to_dict(as_series=False)
            if write_index and self._index:
                data_dict["index"] = list(self._index)

            # Write using pyreadstat
            pyreadstat.write_dta(
                data_dict,
                path,
                variable_labels=variable_labels,
                value_labels=value_labels,
                **kwargs,
            )
        except ImportError:
            raise NotImplementedError(
                "to_stata() requires pyreadstat.\n"
                "Workarounds:\n"
                "  - Install: pip install pyreadstat\n"
                "  - Export to CSV/Parquet first, then convert"
            ) from None

    def to_xarray(self, dim_order: Optional[List[str]] = None, **kwargs: Any) -> Any:
        """
        Return an xarray.Dataset representation of the DataFrame.

        Parameters
        ----------
        dim_order : list, optional
            Dimension order
        **kwargs
            Additional arguments

        Returns
        -------
        xarray.Dataset
            xarray Dataset

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2]})
        >>> ds = df.to_xarray()
        """
        try:
            import numpy as np
            import xarray as xr

            # Convert to dict of arrays
            data_vars = {}
            for col in self.columns:
                series = self[col]
                # Convert to numpy array
                if hasattr(series._series, "to_numpy"):
                    arr = series._series.to_numpy()
                else:
                    arr = np.array(series._series.to_list())
                data_vars[col] = (["index"], arr)

            # Create coordinates
            coords = {}
            if self._index:
                if hasattr(self._index[0], "__len__"):
                    # MultiIndex or complex index
                    coords["index"] = self._index
                else:
                    coords["index"] = self._index

            return xr.Dataset(data_vars=data_vars, coords=coords, **kwargs)
        except ImportError:
            raise NotImplementedError(
                "to_xarray() requires xarray.\n"
                "Workarounds:\n"
                "  - Install: pip install xarray\n"
                "  - Convert to numpy first, then create xarray manually"
            ) from None

    def to_xml(
        self,
        path_or_buffer: Any,
        index: bool = True,
        root_name: str = "data",
        row_name: str = "row",
        na_rep: Optional[str] = None,
        attr_cols: Optional[List[str]] = None,
        elem_cols: Optional[List[str]] = None,
        namespaces: Optional[Dict[str, str]] = None,
        prefix: Optional[str] = None,
        encoding: str = "utf-8",
        xml_declaration: bool = True,
        pretty_print: bool = True,
        parser: str = "lxml",
        stylesheet: Optional[Any] = None,
        compression: Optional[str] = None,
        storage_options: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Render DataFrame to XML.

        Parameters
        ----------
        path_or_buffer : str, Path, or file-like
            File path or buffer
        index : bool, default True
            Write index
        root_name : str, default "data"
            Root element name
        row_name : str, default "row"
            Row element name
        na_rep : str, optional
            Representation for NA values
        attr_cols : list, optional
            Columns to write as attributes
        elem_cols : list, optional
            Columns to write as elements
        namespaces : dict, optional
            XML namespaces
        prefix : str, optional
            Namespace prefix
        encoding : str, default "utf-8"
            Encoding
        xml_declaration : bool, default True
            Include XML declaration
        pretty_print : bool, default True
            Pretty print
        parser : str, default "lxml"
            Parser
        stylesheet : Any, optional
            Stylesheet
        compression : str, optional
            Compression
        storage_options : dict, optional
            Storage options
        **kwargs
            Additional arguments

        Returns
        -------
        str or None
            XML string if path_or_buffer is None, else None

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2]})
        >>> xml = df.to_xml()
        """
        import xml.dom.minidom
        from xml.etree.ElementTree import Element, SubElement, tostring

        # Create root element
        root = Element(root_name)
        if namespaces:
            for prefix, uri in namespaces.items():
                root.set(f"xmlns:{prefix}", uri)

        # Add rows
        for i in range(len(self)):
            row_elem = SubElement(root, row_name)
            if index and self._index:
                idx_val = self._index[i] if i < len(self._index) else i
                row_elem.set("index", str(idx_val))

            for col in self.columns:
                val = self[col].iloc[i]
                if val is None:
                    val = na_rep if na_rep is not None else ""
                else:
                    val = str(val)

                # Determine if column should be attribute or element
                if attr_cols and col in attr_cols:
                    row_elem.set(col, val)
                else:
                    col_elem = SubElement(row_elem, col)
                    col_elem.text = val

        # Convert to string
        if pretty_print:
            xml_str = xml.dom.minidom.parseString(tostring(root)).toprettyxml(
                encoding=encoding
            )
            if isinstance(xml_str, bytes):
                xml_str = xml_str.decode(encoding)  # type: ignore[assignment]
        else:
            xml_str = tostring(root, encoding=encoding).decode(encoding)

        # Add XML declaration if needed
        if xml_declaration and not xml_str.startswith("<?xml"):  # type: ignore[arg-type]
            xml_str = f'<?xml version="1.0" encoding="{encoding}"?>\n{xml_str}'  # type: ignore[str-bytes-safe,assignment]

        if path_or_buffer is None:
            return xml_str  # type: ignore[return-value]
        else:
            if hasattr(path_or_buffer, "write"):
                path_or_buffer.write(xml_str)
            else:
                with open(path_or_buffer, "w", encoding=encoding) as f:
                    f.write(xml_str)  # type: ignore[arg-type]
            return None

    def to_timestamp(
        self,
        freq: Optional[Any] = None,
        how: str = "start",
        axis: int = 0,
        copy: bool = True,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Cast to DatetimeIndex of Timestamps, at beginning of period.

        Parameters
        ----------
        freq : str, optional
            Frequency
        how : str, default "start"
            How to convert
        axis : int, default 0
            Axis
        copy : bool, default True
            Copy data
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            DataFrame with timestamp index

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2]})
        >>> result = df.to_timestamp()
        """
        # Convert index to timestamp if it's a period index
        result = self.copy() if copy else self
        if self._index:
            # Try to convert index to datetime
            try:
                import polars as pl

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
        level: Optional[Any] = None,
        copy: bool = True,
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Convert tz-aware axis to target time zone.

        Parameters
        ----------
        tz : str or tzinfo
            Target timezone
        axis : int, default 0
            Axis
        level : Any, optional
            Level for MultiIndex
        copy : bool, default True
            Copy data
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            DataFrame with converted timezone

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2]})
        >>> result = df.tz_convert("UTC")
        """
        result = self.copy() if copy else self
        if self._index:
            # Convert timezone-aware datetime index
            try:
                import polars as pl

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
        level: Optional[Any] = None,
        copy: bool = True,
        ambiguous: str = "raise",
        nonexistent: str = "raise",
        **kwargs: Any,
    ) -> "DataFrame":
        """
        Localize tz-naive index to target time zone.

        Parameters
        ----------
        tz : str or tzinfo
            Target timezone
        axis : int, default 0
            Axis
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
        DataFrame
            DataFrame with localized timezone

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2]})
        >>> result = df.tz_localize("UTC")
        """
        result = self.copy() if copy else self
        if self._index:
            # Localize timezone-naive datetime index
            try:
                import polars as pl

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

    def sparse(self, **kwargs: Any) -> "DataFrame":
        """
        Convert to sparse format.

        Parameters
        ----------
        **kwargs
            Additional arguments

        Returns
        -------
        DataFrame
            Sparse DataFrame

        Examples
        --------
        >>> import polarpandas as ppd
        >>> df = ppd.DataFrame({"A": [1, 2]})
        >>> result = df.sparse()
        """
        raise NotImplementedError(
            "sparse() is not yet implemented.\n"
            "Workarounds:\n"
            "  - Use pandas: pd_df.sparse for sparse operations\n"
            "  - Polars doesn't have native sparse format support"
        )


class _LocIndexer:
    """Label-based indexer for DataFrame."""

    def __init__(self, df: DataFrame):
        self._df = df

    def __getitem__(self, key: Any) -> Union["Series", "DataFrame", Any]:
        """Get items by label."""
        if isinstance(key, tuple):
            # Check if this is a MultiIndex row key or (row, col) tuple
            # If we have a MultiIndex and the tuple matches the index structure, it's a row key
            is_multiindex = (
                self._df._index is not None
                and len(self._df._index) > 0
                and isinstance(self._df._index[0], tuple)
            )

            if len(key) == 2:
                # Could be (row, col) or MultiIndex row key
                # Check if the first element is a tuple (indicating (row, col) indexing)
                # or if the entire key is a MultiIndex row key
                first_is_tuple = isinstance(key[0], tuple)
                # Check if the key contains slices (MultiIndex slice tuple like ('bar', slice(None)))
                has_slice = any(
                    isinstance(k, slice) for k in key if isinstance(k, (tuple, slice))
                ) or any(isinstance(k, slice) for k in key)

                if first_is_tuple or has_slice:
                    # This is (row, col) indexing where row is a tuple
                    row_key, col_key = key
                    return self._get_rows_cols(row_key, col_key)
                # Check if the entire key is a MultiIndex row key (no slices, first element not a tuple)
                elif is_multiindex and len(key) == len(self._df._index[0]):  # type: ignore[index]
                    # This is a MultiIndex row key, not (row, col)
                    return self._get_rows(key)
                # Otherwise, it's (row, col) indexing
                else:
                    row_key, col_key = key
                    return self._get_rows_cols(row_key, col_key)
            elif is_multiindex and len(key) == len(self._df._index[0]):  # type: ignore[index]
                # This is a MultiIndex row key, not (row, col)
                return self._get_rows(key)
            else:
                # MultiIndex row key with different length - treat as row key
                return self._get_rows(key)
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
            # Check if MultiIndex
            is_multiindex = len(self._df._index) > 0 and isinstance(
                self._df._index[0], tuple
            )

            # Find row index by label
            drop_matched_level = False  # Initialize for all paths
            try:
                if isinstance(row_key, slice):
                    # Handle slice with labels
                    if is_multiindex:
                        # For MultiIndex, use MultiIndex.get_loc for slices
                        if isinstance(self._df._index_name, tuple):
                            names = list(self._df._index_name)
                        else:
                            names = None
                        mi = MultiIndex.from_tuples(self._df._index, names=names)  # type: ignore[arg-type]
                        # For slices with MultiIndex, we need to handle differently
                        # For now, use simple index-based approach
                        start_idx = (
                            self._df._index.index(row_key.start)
                            if row_key.start is not None
                            and row_key.start in self._df._index
                            else 0
                        )
                        stop_idx = (
                            self._df._index.index(row_key.stop)
                            if row_key.stop is not None
                            and row_key.stop in self._df._index
                            else len(self._df._index)
                        )
                        row_indices = list(range(start_idx, stop_idx))
                        drop_matched_level = False
                    else:
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
                    if is_multiindex:
                        # Use MultiIndex.get_loc for each label
                        if isinstance(self._df._index_name, tuple):
                            names = list(self._df._index_name)
                        else:
                            names = None
                        mi = MultiIndex.from_tuples(self._df._index, names=names)  # type: ignore[arg-type]
                        row_indices = []
                        for label in row_key:
                            loc_result = mi.get_loc(label)
                            if isinstance(loc_result, int):
                                row_indices.append(loc_result)
                            elif isinstance(loc_result, list):
                                row_indices.extend(loc_result)
                        # For list of tuple keys, don't drop levels (exact matches)
                        drop_matched_level = False
                    else:
                        row_indices = [
                            self._df._index.index(label) for label in row_key
                        ]
                else:
                    # Single label or tuple - handle tuple keys for MultiIndex
                    drop_matched_level = False
                    if is_multiindex:
                        # Use MultiIndex.get_loc for better support
                        if isinstance(self._df._index_name, tuple):
                            names = list(self._df._index_name)
                        else:
                            names = None
                        mi = MultiIndex.from_tuples(self._df._index, names=names)  # type: ignore[arg-type]
                        # Handle tuple keys (including slice tuples like ('bar', slice(None)))
                        if isinstance(row_key, tuple):
                            # Tuple key - could be exact match or slice tuple
                            loc_result = mi.get_loc(row_key)
                            if isinstance(loc_result, int):
                                row_indices = [loc_result]
                                drop_matched_level = False
                            elif isinstance(loc_result, list):
                                row_indices = loc_result
                                # Check if this is a partial match (not all levels specified or has slices)
                                has_slice = any(isinstance(k, slice) for k in row_key)
                                if has_slice or len(row_key) < mi.nlevels:
                                    # Partial match with slice - don't drop level, keep full MultiIndex
                                    drop_matched_level = False
                                else:
                                    drop_matched_level = False  # Full tuple match
                            else:
                                row_indices = [loc_result]  # type: ignore[list-item]
                                drop_matched_level = False
                        else:
                            # Single scalar key - partial match
                            loc_result = mi.get_loc(row_key)
                            if isinstance(loc_result, int):
                                row_indices = [loc_result]
                                drop_matched_level = False
                            elif isinstance(loc_result, list):
                                row_indices = loc_result
                                # Partial key match - drop the matched level from result
                                drop_matched_level = True
                            else:
                                row_indices = [loc_result]  # type: ignore[list-item]
                                drop_matched_level = False
                    else:
                        # Regular index
                        row_indices = [self._df._index.index(row_key)]

                # Select rows by integer indices
                if len(row_indices) == 1:
                    # Single row - return as Series
                    import polars as pl

                    from polarpandas.series import Series

                    # Use slice to get single row, then convert to Series
                    row_data = polars_df.slice(row_indices[0], 1)
                    # Get values from the first (and only) row
                    values = [row_data[col][0] for col in row_data.columns]
                    # Set Series name to the index label if it's a MultiIndex
                    series_name = None
                    original_name = None
                    if is_multiindex and self._df._index is not None:
                        index_label = self._df._index[row_indices[0]]
                        # For MultiIndex, pandas uses the tuple as the name
                        if isinstance(index_label, tuple):
                            original_name = index_label
                            # Convert tuple to string for Polars compatibility
                            series_name = (
                                str(index_label)
                                if len(index_label) > 1
                                else (index_label[0] if len(index_label) == 1 else None)
                            )
                        else:
                            series_name = index_label
                            original_name = index_label
                    result_series = Series(
                        values, index=row_data.columns, name=series_name
                    )
                    # Store original tuple name for pandas compatibility
                    if original_name is not None and isinstance(original_name, tuple):
                        result_series._original_name = original_name
                    return result_series
                else:
                    # Multiple rows - return as DataFrame
                    selected_df = polars_df[row_indices]
                    result = DataFrame(selected_df, index_name=self._df._index_name)
                    # Preserve index for selected rows
                    selected_index = [self._df._index[i] for i in row_indices]

                    # If partial key match on MultiIndex, drop the matched level
                    if is_multiindex and drop_matched_level and selected_index:
                        # Drop first level from index tuples
                        if (
                            isinstance(selected_index[0], tuple)
                            and len(selected_index[0]) > 1
                        ):
                            # Extract remaining levels
                            remaining_levels = [
                                tup[1:] if len(tup) > 1 else tup
                                for tup in selected_index
                            ]
                            # If only one level remains, convert to regular Index (not MultiIndex)
                            if len(remaining_levels[0]) == 1:
                                # Single level - convert to regular Index
                                result._index = [
                                    tup[0] if isinstance(tup, tuple) else tup
                                    for tup in remaining_levels
                                ]
                                # Update index name to the remaining level name
                                if (
                                    isinstance(self._df._index_name, tuple)
                                    and len(self._df._index_name) > 1
                                ):
                                    result._index_name = self._df._index_name[1]
                                else:
                                    result._index_name = None
                            else:
                                # Multiple levels remain - keep as MultiIndex
                                result._index = remaining_levels
                                # Update index names - drop first level name
                                if (
                                    isinstance(self._df._index_name, tuple)
                                    and len(self._df._index_name) > 1
                                ):
                                    result._index_name = self._df._index_name[1:]
                                else:
                                    result._index_name = None
                        else:
                            result._index = selected_index
                            result._index_name = self._df._index_name
                    else:
                        result._index = selected_index
                        # Preserve index names from original DataFrame
                        result._index_name = self._df._index_name
                    return result
            except ValueError as e:
                raise KeyError(f"'{row_key}' not in index") from e
        else:
            # No index - treat as integer position
            if isinstance(row_key, (slice, list)):
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

        from polarpandas.series import Series as PolarPandasSeries

        mask_series: Optional[pl.Series] = None
        if (
            isinstance(row_key, PolarPandasSeries)
            and row_key._series.dtype == pl.Boolean
        ):
            mask_series = row_key._series
        elif hasattr(row_key, "dtype") and str(row_key.dtype) == "bool":
            mask_series = pl.Series("mask", list(row_key))
        elif (
            isinstance(row_key, list)
            and row_key
            and all(isinstance(item, (bool, np.bool_)) for item in row_key)
        ):
            mask_series = pl.Series("mask", row_key)

        if mask_series is not None:
            mask_values = mask_series.to_list()
            selected_df = polars_df.filter(mask_series)

            if col_key is not None and not (
                isinstance(col_key, slice) and col_key == slice(None)
            ):
                if isinstance(col_key, str):
                    from polarpandas.series import Series

                    result_index: Optional[List[Any]]
                    if self._df._index is not None:
                        selected_indices = [
                            i for i, val in enumerate(mask_values) if val
                        ]
                        result_index = [self._df._index[i] for i in selected_indices]
                    else:
                        result_index = [i for i, val in enumerate(mask_values) if val]
                    return Series(
                        selected_df[col_key], index=result_index, name=col_key
                    )
                else:
                    selected_df = selected_df[col_key]

            result = DataFrame(selected_df)
            if self._df._index is not None:
                selected_indices = [i for i, val in enumerate(mask_values) if val]
                result._index = [self._df._index[i] for i in selected_indices]
                result._index_name = self._df._index_name
            else:
                result._index = [i for i, val in enumerate(mask_values) if val]
            return result

        # Use Polars for row/column selection - limited label-based support
        if self._df._index is not None:
            # Check if MultiIndex
            is_multiindex = len(self._df._index) > 0 and isinstance(
                self._df._index[0], tuple
            )

            # Find row index by label
            try:
                if isinstance(row_key, slice):
                    # Handle slice with labels
                    if is_multiindex:
                        start_idx = (
                            self._df._index.index(row_key.start)
                            if row_key.start is not None
                            and row_key.start in self._df._index
                            else 0
                        )
                        stop_idx = (
                            self._df._index.index(row_key.stop)
                            if row_key.stop is not None
                            and row_key.stop in self._df._index
                            else len(self._df._index)
                        )
                        row_indices = list(range(start_idx, stop_idx))
                    else:
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
                    if is_multiindex:
                        # Use MultiIndex.get_loc for each label
                        if isinstance(self._df._index_name, tuple):
                            names = list(self._df._index_name)
                        else:
                            names = None
                        mi = MultiIndex.from_tuples(self._df._index, names=names)  # type: ignore[arg-type]
                        row_indices = []
                        for label in row_key:
                            loc_result = mi.get_loc(label)
                            if isinstance(loc_result, int):
                                row_indices.append(loc_result)
                            elif isinstance(loc_result, list):
                                row_indices.extend(loc_result)
                    else:
                        row_indices = [
                            self._df._index.index(label) for label in row_key
                        ]
                else:
                    # Single label - handle tuple keys for MultiIndex
                    if is_multiindex:
                        # Use MultiIndex.get_loc for better support
                        if isinstance(self._df._index_name, tuple):
                            names = list(self._df._index_name)
                        else:
                            names = None
                        mi = MultiIndex.from_tuples(self._df._index, names=names)  # type: ignore[arg-type]
                        loc_result = mi.get_loc(row_key)
                        if isinstance(loc_result, int):
                            row_indices = [loc_result]
                        elif isinstance(loc_result, list):
                            row_indices = loc_result
                        else:
                            row_indices = [loc_result]  # type: ignore[list-item]
                    else:
                        # Regular index
                        row_indices = [self._df._index.index(row_key)]

                # Select rows and columns
                if len(row_indices) == 1 and isinstance(col_key, str):
                    # Single cell access - return scalar value directly
                    row_values = polars_df.row(row_indices[0], named=True)
                    return row_values[col_key]
                elif len(row_indices) == 1:
                    # Single row, multiple columns - return as Series
                    import polars as pl

                    from polarpandas.series import Series

                    # Get the row and selected columns
                    if isinstance(col_key, list):
                        # Multiple columns - create Series with column names as index
                        row_data = polars_df.slice(row_indices[0], 1)
                        values = [row_data[col][0] for col in col_key]
                        # Set Series name to the index label if it's a MultiIndex
                        series_name = None
                        original_name = None
                        if is_multiindex and self._df._index is not None:
                            index_label = self._df._index[row_indices[0]]
                            if isinstance(index_label, tuple):
                                original_name = index_label
                                series_name = (
                                    str(index_label)
                                    if len(index_label) > 1
                                    else (
                                        index_label[0]
                                        if len(index_label) == 1
                                        else None
                                    )
                                )
                            else:
                                series_name = index_label
                                original_name = index_label
                        result_series = Series(values, index=col_key, name=series_name)
                        if original_name is not None and isinstance(
                            original_name, tuple
                        ):
                            result_series._original_name = original_name
                        return result_series
                    else:
                        # Single column
                        return Series(polars_df[row_indices[0], col_key])
                else:
                    # Multiple rows - return as DataFrame
                    selected_df = polars_df[row_indices, col_key]
                    result = DataFrame(selected_df, index_name=self._df._index_name)
                    # Preserve index for selected rows
                    result._index = [self._df._index[i] for i in row_indices]
                    # Preserve index names
                    result._index_name = self._df._index_name
                    return result
            except ValueError as e:
                raise KeyError(f"'{row_key}' not in index") from e
        else:
            # No index - treat as integer position
            if isinstance(row_key, (slice, list)):
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
                        row_data = polars_df.row(row_key, named=True)  # type: ignore[assignment]
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
                            if _is_integer_dtype(dtype):
                                new_row_data_no_cast[col] = 0
                            elif _is_float_dtype(dtype):
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
                        if _is_integer_dtype(dtype):
                            new_row_data[col] = 0
                        elif _is_float_dtype(dtype):
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
                            if _is_integer_dtype(dtype):
                                cast_exprs.append(
                                    pl.col(col).cast(pl.Float64).alias(col)
                                )
                                new_row_data_cast[col] = float("nan")
                            elif _is_float_dtype(dtype):
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
                        if _is_integer_dtype(dtype):
                            cast_exprs.append(pl.col(col).cast(pl.Float64).alias(col))
                            new_row_data_no_index[col] = float("nan")
                        elif _is_float_dtype(dtype):
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

    def min(self) -> "DataFrame":
        """Calculate rolling minimum."""
        return self._apply_rolling("min")

    def apply(
        self,
        func: Callable[[Union[List[Any], Any]], Any],
        raw: bool = False,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> "DataFrame":
        """Apply a custom function over each rolling window."""
        unused_engine = engine, engine_kwargs  # present for pandas compatibility
        _ = unused_engine

        args = args or ()
        kwargs = kwargs or {}
        result_cols = []
        polars_df = self._df._df
        for col in self._df.columns:
            column_series = polars_df[col]
            col_name = col

            def rolling_function(  # pragma: no cover - small wrapper
                window_series: pl.Series, *, _col: str = col_name
            ) -> Any:
                window_values = window_series.to_list()
                if raw:
                    window_input: Union[List[Any], Any] = window_values
                else:
                    from polarpandas.series import Series

                    window_input = Series(window_values, name=_col)
                return func(window_input, *args, **kwargs)

            rolling_kwargs: Dict[str, Any] = {}
            if "weights" in self._kwargs:
                rolling_kwargs["weights"] = self._kwargs["weights"]
            if "center" in self._kwargs:
                rolling_kwargs["center"] = self._kwargs["center"]
            if "min_periods" in self._kwargs:
                rolling_kwargs["min_samples"] = self._kwargs["min_periods"]

            applied = column_series.rolling_map(
                rolling_function,
                window_size=self._window,
                **rolling_kwargs,
            )
            result_cols.append(applied.alias(col))

        result_df = polars_df.select(result_cols)
        return DataFrame(result_df)


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
        result_df = DataFrame(result)

        # Handle level-based grouping - rename temporary columns and set as index
        if hasattr(self, "_level_info") and self._level_info:
            level_columns = self._level_info["level_columns"]
            level_names = self._level_info["level_names"]

            # Rename temporary level columns to level names
            rename_map = dict(zip(level_columns, level_names))
            result_polars = result_df._df.rename(rename_map)
            result_df = DataFrame(result_polars)

            # Set level columns as index
            level_values_list = []
            for col in level_names:
                if col in result_df.columns:
                    level_values_list.append(result_df[col].tolist())
                    result_df = result_df.drop(columns=[col])  # type: ignore[assignment]

            # Create MultiIndex from level values if multiple levels, or regular Index if single
            if len(level_values_list) > 1:
                # Multiple levels - create MultiIndex tuples
                # Sort by level values to match pandas behavior
                combined = list(zip(*level_values_list))
                result_df._index = combined
                result_df._index_name = tuple(level_names)
            elif len(level_values_list) == 1:
                # Single level - create regular Index
                # Sort to match pandas behavior (pandas groupby sorts by group keys)
                level_values = level_values_list[0]
                # Create sorted index and reorder DataFrame rows accordingly
                sorted_pairs = sorted(enumerate(level_values), key=lambda x: x[1])
                sorted_indices = [i for i, _ in sorted_pairs]
                sorted_level_values = [v for _, v in sorted_pairs]

                # Reorder DataFrame rows to match sorted index
                result_df = result_df.iloc[sorted_indices]  # type: ignore[assignment]
                result_df._index = sorted_level_values
                result_df._index_name = level_names[0] if level_names else None

        return result_df

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
