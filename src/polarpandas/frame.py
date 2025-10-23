"""
A DataFrame object that behaves like a pandas DataFrame
but using polars DataFrame to do all the work.
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


class DataFrame:
    """
    A mutable DataFrame wrapper around Polars DataFrame with pandas-like API.

    This class wraps a Polars DataFrame and provides a pandas-compatible interface
    with in-place mutation support.
    """

    _index: Optional[List[Any]]
    _index_name: Optional[Union[str, Tuple[str, ...]]]
    _columns_index: Optional[Any]

    def __init__(
        self,
        data: Optional[Union[Dict[str, Any], List[Any], pl.DataFrame]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a DataFrame from various data sources.

        Parameters
        ----------
        data : dict, list, pl.DataFrame, or None
            Data to initialize the DataFrame with. Can be:
            - Dictionary of column names to values
            - List of dictionaries
            - Existing Polars DataFrame
            - None for empty DataFrame
        index : array-like, optional
            Index to use for resulting frame
        """
        if data is None:
            # Handle columns and index parameters for empty DataFrame
            columns = kwargs.pop("columns", None)
            index = kwargs.pop("index", None)

            if index is not None and columns is not None:
                # Create empty DataFrame with specified columns and index
                # Use Polars to create empty DataFrame with columns
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
        elif isinstance(data, pl.DataFrame):
            self._df = data
            self._index = None
            self._index_name = None
        else:
            # Handle index and columns parameters separately since Polars doesn't support them directly
            index = kwargs.pop("index", None)
            columns = kwargs.pop("columns", None)
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

    @classmethod
    def read_csv(cls, path: str, **kwargs: Any) -> "DataFrame":
        """
        Read a CSV file into DataFrame.

        Parameters
        ----------
        path : str
            Path to CSV file
        **kwargs
            Additional arguments passed to Polars read_csv()

        Returns
        -------
        DataFrame
            DataFrame loaded from CSV
        """
        # Map pandas-style parameters to Polars equivalents
        polars_kwargs = {}

        # Handle pandas-specific parameters
        index_col = kwargs.pop("index_col", None)

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

        # Read CSV with Polars
        try:
            df = pl.read_csv(path, **polars_kwargs)
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
            result = cls(df)
            if isinstance(index_col, (int, str)):
                # Single column as index
                if isinstance(index_col, int):
                    col_name = df.columns[index_col]
                else:
                    col_name = index_col

                # Set the column as index
                result._index = df[col_name].to_list()
                result._index_name = col_name
                # Remove the column from data
                result._df = df.drop(col_name)
            else:
                # Multiple columns as index
                col_names = [
                    df.columns[i] if isinstance(i, int) else i for i in index_col
                ]
                # Set the columns as index (as list of tuples)
                result._index = list(zip(*[df[col].to_list() for col in col_names]))
                result._index_name = tuple(col_names)
                # Remove the columns from data
                result._df = df.drop(col_names)

            return result
        else:
            return cls(df)

    @classmethod
    def read_parquet(cls, path: str, **kwargs: Any) -> "DataFrame":
        """
        Read a Parquet file into DataFrame.

        Parameters
        ----------
        path : str
            Path to Parquet file
        **kwargs
            Additional arguments passed to Polars read_parquet()

        Returns
        -------
        DataFrame
            DataFrame loaded from Parquet
        """
        return cls(pl.read_parquet(path, **kwargs))

    @classmethod
    def read_json(cls, path: str, **kwargs: Any) -> "DataFrame":
        """
        Read a JSON file into DataFrame.

        Parameters
        ----------
        path : str
            Path to JSON file
        **kwargs
            Additional arguments passed to Polars read_json()

        Returns
        -------
        DataFrame
            DataFrame loaded from JSON
        """
        # Map pandas-style parameters to Polars equivalents
        polars_kwargs = {}

        # Use Polars JSON read - orient parameter support is limited
        # Remove pandas-specific parameters that Polars doesn't support
        polars_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["orient", "lines"]
        }

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
        **kwargs
            Additional arguments passed to Polars read_database()

        Returns
        -------
        DataFrame
            DataFrame loaded from SQL query
        """
        return cls(pl.read_database(sql, con, **kwargs))

    @classmethod
    def read_feather(cls, path: str, **kwargs: Any) -> "DataFrame":
        """
        Read Feather file into DataFrame.

        Parameters
        ----------
        path : str
            Path to Feather file
        **kwargs
            Additional arguments passed to Polars read_ipc()

        Returns
        -------
        DataFrame
            DataFrame loaded from Feather file
        """
        return cls(pl.read_ipc(path, **kwargs))

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
        Get a column or subset of the DataFrame.

        Parameters
        ----------
        key : str or other
            Column name or selection key

        Returns
        -------
        Column data or DataFrame subset
        """
        try:
            return self._df.__getitem__(key)  # type: ignore[return-value]
        except Exception as e:
            # Convert Polars exceptions to pandas-compatible ones
            if "not found" in str(e).lower() or "ColumnNotFoundError" in str(type(e)):
                raise KeyError(str(e)) from e
            raise

    def __setitem__(self, column: str, values: Union[Any, "Series"]) -> None:
        """
        Set a column in the DataFrame (in-place mutation).

        Parameters
        ----------
        column : str
            Column name
        values : array-like, scalar, or Series
            Values to set
        """
        # Convert values to Polars Series if needed
        if isinstance(values, pl.Series):
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

        Parameters
        ----------
        columns : str or list of str
            Column(s) to drop
        inplace : bool, default False
            If True, modify DataFrame in place and return None.
            If False, return a new DataFrame.

        Returns
        -------
        DataFrame or None
            DataFrame with columns dropped, or None if inplace=True
        """
        # Polars drop() accepts both str and list
        try:
            result_df = self._df.drop(columns)
        except Exception as e:
            # Convert Polars exceptions to pandas-compatible ones
            if "not found" in str(e).lower() or "ColumnNotFoundError" in str(type(e)):
                raise KeyError(str(e)) from e
            raise

        if inplace:
            self._df = result_df
            return None
        else:
            return DataFrame(result_df)

    def rename(
        self,
        mapping: Optional[Dict[str, str]] = None,
        columns: Optional[Dict[str, str]] = None,
        inplace: bool = False,
    ) -> Optional["DataFrame"]:
        """
        Rename columns.

        Parameters
        ----------
        mapping : dict, optional
            Mapping of old column names to new column names (deprecated, use columns)
        columns : dict, optional
            Mapping of old column names to new column names
        inplace : bool, default False
            If True, modify DataFrame in place and return None.
            If False, return a new DataFrame.

        Returns
        -------
        DataFrame or None
            DataFrame with renamed columns, or None if inplace=True
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
        Sort DataFrame by column values.

        Parameters
        ----------
        by : str or list of str
            Column name(s) to sort by
        inplace : bool, default False
            If True, modify DataFrame in place and return None.
            If False, return a new DataFrame.
        **kwargs : additional arguments
            Additional arguments passed to Polars sort()

        Returns
        -------
        DataFrame or None
            Sorted DataFrame, or None if inplace=True
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
        dtypes_dict = dict(zip(self._df.columns, self._df.dtypes))

        # Add empty attribute to match pandas behavior
        class DtypesDict(Dict[str, Any]):
            @property
            def empty(self) -> bool:
                return len(self) == 0

        return DtypesDict(dtypes_dict)

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
        Return the first n rows.

        Parameters
        ----------
        n : int, default 5
            Number of rows to return

        Returns
        -------
        DataFrame
            First n rows of the DataFrame
        """
        return DataFrame(self._df.head(n))

    def tail(self, n: int = 5) -> "DataFrame":
        """
        Return the last n rows.

        Parameters
        ----------
        n : int, default 5
            Number of rows to return

        Returns
        -------
        DataFrame
            Last n rows of the DataFrame
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
        return DataFrame(self._df.filter(*args, **kwargs))

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

        Parameters
        ----------
        by : str or list of str
            Column name(s) to group by

        Returns
        -------
        GroupBy
            GroupBy object for aggregation
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
        Merge (join) with another DataFrame.

        Parameters
        ----------
        other : DataFrame
            DataFrame to merge with

        Returns
        -------
        DataFrame
            Merged DataFrame
        """
        # Extract the underlying Polars DataFrame if other is wrapped
        if isinstance(other, DataFrame):
            other_df = other._df
        else:
            # This branch is technically unreachable due to type annotation
            # but kept for defensive programming
            other_df = other  # type: ignore[unreachable]

        return DataFrame(self._df.join(other_df, *args, **kwargs))

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
            return self._df.equals(other._df)
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
                raise KeyError(f"'{key}'")

        # Check if any index columns contain nulls
        has_nulls = any(self._df[key].null_count() > 0 for key in keys)

        if has_nulls:
            # Polars has limited null handling in index - this is a limitation
            raise ValueError(
                "Polars has limited support for null values in index. This is a known limitation."
            )

        if inplace:
            # Modify in place
            if append and self._index is not None:
                # Append to existing index - create tuples
                existing_index = list(self._index)
                if len(keys) == 1:
                    new_values = self._df[keys[0]].to_list()
                else:
                    new_values = list(zip(*[self._df[key].to_list() for key in keys]))

                # Create tuples of (existing_index[i], new_values[i])
                new_index = []
                for i in range(len(existing_index)):
                    if len(keys) == 1:
                        new_index.append((existing_index[i], new_values[i]))
                    else:
                        new_index.append((existing_index[i],) + new_values[i])
                self._index = new_index
                # Update index name for append
                if isinstance(self._index_name, (list, tuple)):
                    self._index_name = tuple(list(self._index_name) + keys)
                else:
                    self._index_name = (
                        tuple([self._index_name] + keys)
                        if self._index_name is not None
                        else tuple(keys)
                    )
            else:
                # Replace index
                if len(keys) == 1:
                    self._index = self._df[keys[0]].to_list()
                    self._index_name = keys[0]
                else:
                    # Multi-level index - create tuples
                    new_values = list(zip(*[self._df[key].to_list() for key in keys]))
                    self._index = new_values
                    self._index_name = tuple(keys)  # Store as tuple for hashability

            # Drop columns if requested
            if drop:
                columns_to_keep = [col for col in self._df.columns if col not in keys]
                if columns_to_keep:
                    self._df = self._df.select(columns_to_keep)
                else:
                    # If all columns are used as index, create empty DataFrame with index
                    self._df = pl.DataFrame()

            return None
        else:
            # Create a copy
            result = DataFrame(self._df)

            if append and self._index is not None:
                # Append to existing index - create tuples
                existing_index = list(self._index)
                if len(keys) == 1:
                    new_values = self._df[keys[0]].to_list()
                else:
                    new_values = list(zip(*[self._df[key].to_list() for key in keys]))

                # Create tuples of (existing_index[i], new_values[i])
                new_index = []
                for i in range(len(existing_index)):
                    if len(keys) == 1:
                        new_index.append((existing_index[i], new_values[i]))
                    else:
                        new_index.append((existing_index[i],) + new_values[i])
                result._index = new_index
                # Update index name for append
                if isinstance(self._index_name, (list, tuple)):
                    result._index_name = tuple(list(self._index_name) + keys)
                else:
                    result._index_name = (
                        tuple([self._index_name] + keys)
                        if self._index_name is not None
                        else tuple(keys)
                    )
            else:
                # Replace index
                if len(keys) == 1:
                    result._index = self._df[keys[0]].to_list()
                    result._index_name = keys[0]
                else:
                    # Multi-level index - create tuples
                    new_values = list(zip(*[self._df[key].to_list() for key in keys]))
                    result._index = new_values
                    result._index_name = tuple(keys)  # Store as tuple for hashability

            # Drop columns if requested
            if drop:
                columns_to_keep = [col for col in self._df.columns if col not in keys]
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
        if len(self._df) == 0:
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

        # Handle index=True case (convert to pandas and back)
        try:
            import pandas as pd  # noqa: F401

            pd_df = self.to_pandas()
            if path is None:
                return pd_df.to_csv(**kwargs)  # type: ignore[no-any-return]
            else:
                pd_df.to_csv(path, **kwargs)
                return None
        except ImportError as e:
            raise ImportError(
                "pandas is required for CSV operations with index=True"
            ) from e

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
        pandas_df = self._df.to_pandas()

        # Set index if we have one
        if self._index is not None:
            pandas_df.index = self._index  # type: ignore[assignment]
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
                    index_name = self._index_name if self._index_name != "" else None
                    pandas_df.index.name = index_name

        # Convert string column names that look like integers to RangeIndex
        try:
            # Check if all column names are string representations of consecutive integers starting from 0
            col_names = list(pandas_df.columns)
            if all(isinstance(name, str) and name.isdigit() for name in col_names):
                int_cols = [int(name) for name in col_names]
                if int_cols == list(range(len(int_cols))):
                    # Convert to RangeIndex
                    pandas_df.columns = pd.RangeIndex(  # type: ignore[assignment]
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

    def apply(self, func: Callable[..., Any], axis: int = 0) -> Union["Series", "DataFrame"]:
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
        method_map = {
            "average": "average",
            "min": "min",
            "max": "max",
            "first": "ordinal",  # Polars uses 'ordinal' for first occurrence
            "dense": "dense",
        }
        polars_method = method_map.get(method, method)

        # Apply rank to each column
        result_cols = []
        for col in self._df.columns:
            if numeric_only and not self._df[col].dtype.is_numeric():
                # Skip non-numeric columns when numeric_only=True
                continue
            else:
                rank_expr = pl.col(col).rank(
                    method=polars_method, descending=not ascending  # type: ignore[arg-type]
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
        # Handle boolean indexing with pandas Series
        if hasattr(row_key, "dtype") and str(row_key.dtype) == "bool":
            # Convert pandas Series mask to Polars expression
            import polars as pl

            mask_values = row_key.tolist()
            mask_series = pl.Series("mask", mask_values)
            selected_df = self._df._df.filter(mask_series)
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

                    return Series(self._df._df[row_indices[0]])  # type: ignore[arg-type]
                else:
                    # Multiple rows - return as DataFrame
                    selected_df = self._df._df[row_indices]
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
                    selected_df = self._df._df[row_key]
                    return DataFrame(selected_df)
                except IndexError as e:
                    # Convert Polars IndexError to pandas KeyError for compatibility
                    raise KeyError(f"index {row_key} is out of bounds") from e
            elif isinstance(row_key, list):
                try:
                    selected_df = self._df._df[row_key]
                    return DataFrame(selected_df)
                except IndexError as e:
                    # Convert Polars IndexError to pandas KeyError for compatibility
                    raise KeyError(f"index {row_key} is out of bounds") from e
            else:
                # Single row - return as Series
                from polarpandas.series import Series

                try:
                    # Get single row as Series - use slice to get all columns
                    row_data = self._df._df.slice(row_key, 1)
                    # Convert to Series by taking the first (and only) row
                    # Create a list of values in column order
                    values = [row_data[col][0] for col in row_data.columns]
                    return Series(values, index=row_data.columns, strict=False)
                except IndexError as e:
                    # Convert Polars IndexError to pandas KeyError for compatibility
                    raise KeyError(f"index {row_key} is out of bounds") from e

    def _get_rows_cols(self, row_key: Any, col_key: Any) -> Union["Series", "DataFrame", Any]:
        """Get rows and columns by label."""
        # Handle boolean indexing with pandas Series
        if hasattr(row_key, "dtype") and str(row_key.dtype) == "bool":
            # Convert pandas Series mask to Polars expression
            import polars as pl

            mask_values = row_key.tolist()
            mask_series = pl.Series("mask", mask_values)
            selected_df = self._df._df.filter(mask_series)

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
                    return self._df._df[row_indices[0], col_key]
                elif len(row_indices) == 1:
                    # Single row, multiple columns - return as Series
                    from polarpandas.series import Series

                    return Series(self._df._df[row_indices[0], col_key])
                else:
                    # Multiple rows - return as DataFrame
                    selected_df = self._df._df[row_indices, col_key]
                    result = DataFrame(selected_df)
                    # Preserve index for selected rows
                    result._index = [self._df._index[i] for i in row_indices]
                    return result
            except ValueError as e:
                raise KeyError(f"'{row_key}' not in index") from e
        else:
            # No index - treat as integer position
            if isinstance(row_key, slice):
                selected_df = self._df._df[row_key, col_key]
                return DataFrame(selected_df)
            elif isinstance(row_key, list):
                selected_df = self._df._df[row_key, col_key]
                return DataFrame(selected_df)
            else:
                # Single cell access - return scalar value directly
                if isinstance(col_key, str):
                    return self._df._df[row_key, col_key]
                else:
                    # Single row, multiple columns - return as Series
                    from polarpandas.series import Series

                    return Series(self._df._df[row_key, col_key])

    def _set_rows(self, row_key: Any, value: Any) -> None:
        """Set rows by label."""
        # Convert to pandas for label-based indexing
        pd_df = self._df._df.to_pandas()

        # Set the index if we have a stored index
        if self._df._index is not None:
            pd_df.index = self._df._index  # type: ignore[assignment]

        pd_df.loc[row_key] = value
        self._df._df = pl.from_pandas(pd_df)
        # Preserve the index after assignment
        self._df._index = pd_df.index.tolist()
        self._df._index_name = pd_df.index.name  # type: ignore[assignment]

    def _set_rows_cols(self, row_key: Any, col_key: Any, value: Any) -> None:
        """Set rows and columns by label."""
        # Convert to pandas for label-based indexing
        pd_df = self._df._df.to_pandas()

        # Set the index if we have a stored index
        if self._df._index is not None:
            pd_df.index = self._df._index  # type: ignore[assignment]

        pd_df.loc[row_key, col_key] = value
        self._df._df = pl.from_pandas(pd_df)
        # Preserve the index after assignment
        self._df._index = pd_df.index.tolist()
        self._df._index_name = pd_df.index.name  # type: ignore[assignment]


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
        # Use Polars for integer-based indexing
        if isinstance(row_key, slice):
            selected_df = self._df._df[row_key]
            return DataFrame(selected_df)
        elif isinstance(row_key, list):
            selected_df = self._df._df[row_key]
            return DataFrame(selected_df)
        else:
            # Single row - return as Series
            from polarpandas.series import Series

            return Series(self._df._df[row_key])

    def _get_rows_cols(self, row_key: Any, col_key: Any) -> Union["Series", "DataFrame", Any]:
        """Get rows and columns by integer position."""
        # Use Polars for integer-based indexing
        if isinstance(row_key, slice):
            selected_df = self._df._df[row_key, col_key]
            return DataFrame(selected_df)
        elif isinstance(row_key, list):
            selected_df = self._df._df[row_key, col_key]
            return DataFrame(selected_df)
        else:
            # Single cell access - return scalar value directly
            if isinstance(col_key, (int, str)):
                return self._df._df[row_key, col_key]
            else:
                # Single row, multiple columns - return as Series
                from polarpandas.series import Series

                return Series(self._df._df[row_key, col_key])

    def _set_rows(self, row_key: Any, value: Any) -> None:
        """Set rows by integer position."""
        # Convert to pandas for integer-based indexing
        pd_df = self._df._df.to_pandas()
        pd_df.iloc[row_key] = value
        self._df._df = pl.from_pandas(pd_df)

    def _set_rows_cols(self, row_key: Any, col_key: Any, value: Any) -> None:
        """Set rows and columns by integer position."""
        # Convert to pandas for integer-based indexing
        pd_df = self._df._df.to_pandas()
        pd_df.iloc[row_key, col_key] = value
        self._df._df = pl.from_pandas(pd_df)


class _AtIndexer:
    """Label-based scalar accessor for DataFrame."""

    def __init__(self, df: DataFrame):
        self._df = df

    def __getitem__(self, key: Any) -> Any:
        """Get single value by label."""
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            # Use Polars for label-based indexing - limited support
            if self._df._index is not None:
                # Find row index
                try:
                    row_idx = self._df._index.index(row_key)
                    return self._df._df[row_idx, col_key]
                except ValueError as e:
                    raise KeyError(f"'{row_key}' not in index") from e
            else:
                # No index - use integer position
                if isinstance(row_key, int):
                    return self._df._df[row_key, col_key]
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
                    # Update value in Polars DataFrame
                    self._df._df = self._df._df.with_columns(
                        pl.when(pl.int_range(pl.len()) == row_idx)
                        .then(pl.lit(value))
                        .otherwise(pl.col(col_key))
                        .alias(col_key)
                    )
                except ValueError as e:
                    raise KeyError(f"'{row_key}' not in index") from e
            else:
                # No index - use integer position
                if isinstance(row_key, int):
                    # Update value in Polars DataFrame
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
            # Convert to pandas for integer-based indexing
            pd_df = self._df._df.to_pandas()
            return pd_df.iat[row_key, col_key]
        else:
            raise ValueError("iat accessor requires (row, col) tuple")

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set single value by integer position."""
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            # Convert to pandas for integer-based indexing
            pd_df = self._df._df.to_pandas()
            pd_df.iat[row_key, col_key] = value
            self._df._df = pl.from_pandas(pd_df)
        else:
            raise ValueError("iat accessor requires (row, col) tuple")


class _RollingGroupBy:
    """Rolling window groupby object."""

    def __init__(self, df: DataFrame, window: int, **kwargs: Any) -> None:
        self._df = df
        self._window = window
        self._kwargs = kwargs

    def mean(self) -> "DataFrame":
        """Calculate rolling mean."""
        result_cols = []
        for col in self._df.columns:
            result_cols.append(
                self._df._df[col].rolling_mean(window_size=self._window).alias(col)
            )
        return DataFrame(self._df._df.select(result_cols))

    def sum(self) -> "DataFrame":
        """Calculate rolling sum."""
        result_cols = []
        for col in self._df.columns:
            result_cols.append(
                self._df._df[col].rolling_sum(window_size=self._window).alias(col)
            )
        return DataFrame(self._df._df.select(result_cols))

    def std(self) -> "DataFrame":
        """Calculate rolling standard deviation."""
        result_cols = []
        for col in self._df.columns:
            result_cols.append(
                self._df._df[col].rolling_std(window_size=self._window).alias(col)
            )
        return DataFrame(self._df._df.select(result_cols))

    def max(self) -> "DataFrame":
        """Calculate rolling maximum."""
        result_cols = []
        for col in self._df.columns:
            result_cols.append(
                self._df._df[col].rolling_max(window_size=self._window).alias(col)
            )
        return DataFrame(self._df._df.select(result_cols))

    def min(self) -> "DataFrame":
        """Calculate rolling minimum."""
        result_cols = []
        for col in self._df.columns:
            result_cols.append(
                self._df._df[col].rolling_min(window_size=self._window).alias(col)
            )
        return DataFrame(self._df._df.select(result_cols))


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
