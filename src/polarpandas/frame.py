"""
A DataFrame object that behaves like a pandas DataFrame
but using polars DataFrame to do all the work.
"""

from typing import Any, Dict, List, Optional, Union
import polars as pl
from polarpandas.index import Index


class DataFrame:
    """
    A mutable DataFrame wrapper around Polars DataFrame with pandas-like API.

    This class wraps a Polars DataFrame and provides a pandas-compatible interface
    with in-place mutation support.
    """

    def __init__(
        self,
        data: Optional[Union[Dict[str, Any], List[Any], pl.DataFrame]] = None,
        *args: Any,
        **kwargs: Any,
    ):
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
        """
        if data is None:
            self._df = pl.DataFrame()
        elif isinstance(data, pl.DataFrame):
            self._df = data
        else:
            # Handle dict, list, or other data
            self._df = pl.DataFrame(data, *args, **kwargs)

    @classmethod
    def read_csv(cls, path, **kwargs):
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
        return cls(pl.read_csv(path, **kwargs))

    @classmethod
    def read_parquet(cls, path, **kwargs):
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
    def read_json(cls, path, **kwargs):
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
        return cls(pl.read_json(path, **kwargs))

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
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __repr__(self) -> str:
        """Return string representation of the DataFrame."""
        return repr(self._df)

    def __str__(self) -> str:
        """Return string representation of the DataFrame."""
        return str(self._df)

    def __len__(self) -> int:
        """Return the number of rows in the DataFrame."""
        return len(self._df)

    def __getitem__(self, key):
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
        return self._df.__getitem__(key)

    def __setitem__(self, column: str, values) -> None:
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

    def drop(self, columns, inplace: bool = False):
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
        result_df = self._df.drop(columns)

        if inplace:
            self._df = result_df
            return None
        else:
            return DataFrame(result_df)

    def rename(self, mapping: Dict[str, str], inplace: bool = False):
        """
        Rename columns.

        Parameters
        ----------
        mapping : dict
            Mapping of old column names to new column names
        inplace : bool, default False
            If True, modify DataFrame in place and return None.
            If False, return a new DataFrame.

        Returns
        -------
        DataFrame or None
            DataFrame with renamed columns, or None if inplace=True
        """
        result_df = self._df.rename(mapping)

        if inplace:
            self._df = result_df
            return None
        else:
            return DataFrame(result_df)

    def sort_values(self, by, inplace: bool = False, **kwargs):
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

    def fillna(self, value, inplace: bool = False, **kwargs):
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

    def dropna(self, inplace: bool = False, **kwargs):
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
    def empty(self) -> bool:
        """Return True if DataFrame is empty."""
        return len(self._df) == 0

    @property
    def values(self):
        """Return the values of the DataFrame as a numpy array."""
        return self._df.to_numpy()

    @property
    def index(self):
        """Return the index (row labels) of the DataFrame."""
        # Create a simple RangeIndex-like object
        return Index(list(range(len(self._df))))

    @property
    def loc(self):
        """Access a group of rows and columns by label(s)."""
        # For now, return a simple stub
        # Full implementation would return a LocIndexer object
        return _LocIndexer(self)

    @property
    def iloc(self):
        """Access a group of rows and columns by integer position(s)."""
        # For now, return a simple stub
        # Full implementation would return an ILocIndexer object
        return _ILocIndexer(self)

    # Methods
    def head(self, n: int = 5):
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

    def tail(self, n: int = 5):
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

    def copy(self):
        """
        Make a copy of this DataFrame.

        Returns
        -------
        DataFrame
            A copy of the DataFrame
        """
        return DataFrame(self._df.clone())

    def select(self, *args, **kwargs):
        """
        Select columns from DataFrame.

        Returns wrapped DataFrame.
        """
        return DataFrame(self._df.select(*args, **kwargs))

    def filter(self, *args, **kwargs):
        """
        Filter rows from DataFrame.

        Returns wrapped DataFrame.
        """
        return DataFrame(self._df.filter(*args, **kwargs))

    def isna(self):
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

    def notna(self):
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

    def groupby(self, by, *args, **kwargs):
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

    def melt(self, id_vars=None, value_vars=None, **kwargs):
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

        return DataFrame(self._df.unpivot(**unpivot_kwargs))

    def merge(self, other, *args, **kwargs):
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
            other_df = other

        return DataFrame(self._df.join(other_df, *args, **kwargs))

    def join(self, other, *args, **kwargs):
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

    def describe(self):
        """
        Generate descriptive statistics.

        Returns
        -------
        DataFrame
            Summary statistics
        """
        return DataFrame(self._df.describe())

    def info(self):
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

    def drop_duplicates(self, subset=None, inplace: bool = False, **kwargs):
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

    def duplicated(self, subset=None, keep="first"):
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

    def sort_index(self, inplace: bool = False, **kwargs):
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

    def isin(self, values):
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

    def equals(self, other):
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

    def reset_index(self, drop=False, inplace: bool = False):
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

    def to_csv(self, path=None, **kwargs):
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
        if path is None:
            return self._df.write_csv()
        else:
            self._df.write_csv(path, **kwargs)
            return None

    def to_parquet(self, path, **kwargs):
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

    def to_json(self, path=None, **kwargs):
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
        if path is None:
            return self._df.write_json()
        else:
            self._df.write_json(path, **kwargs)
            return None

    def sample(self, n=None, frac=None, **kwargs):
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

    def pivot(self, index=None, columns=None, values=None):
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
        return DataFrame(self._df.pivot(on=columns, index=index, values=values))

    def rolling(self, window, **kwargs):
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

    def apply(self, func, axis=0):
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

    def applymap(self, func):
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


class _LocIndexer:
    """Label-based indexer for DataFrame."""

    def __init__(self, df: DataFrame):
        self._df = df

    def __getitem__(self, key):
        """Get rows by label."""
        if isinstance(key, int):
            # Single row by index
            return self._df._df[key]
        elif isinstance(key, slice):
            # Slice of rows
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(self._df)
            return DataFrame(self._df._df[start:stop])
        else:
            # Other indexing
            return self._df._df[key]


class _ILocIndexer:
    """Integer position-based indexer for DataFrame."""

    def __init__(self, df: DataFrame):
        self._df = df

    def __getitem__(self, key):
        """Get rows by integer position."""
        if isinstance(key, int):
            # Single row by position
            return self._df._df[key]
        elif isinstance(key, slice):
            # Slice of rows
            return DataFrame(self._df._df[key])
        elif isinstance(key, tuple):
            # Row and column indexing
            row_key, col_key = key
            result = self._df._df[row_key]
            if isinstance(col_key, (int, str)):
                return result[col_key]
            else:
                return DataFrame(result.select(col_key))
        else:
            # List or array indexing
            return DataFrame(self._df._df[key])


class _RollingGroupBy:
    """Rolling window groupby object."""

    def __init__(self, df: DataFrame, window: int, **kwargs):
        self._df = df
        self._window = window
        self._kwargs = kwargs

    def mean(self):
        """Calculate rolling mean."""
        result_cols = []
        for col in self._df.columns:
            result_cols.append(
                self._df._df[col].rolling_mean(window_size=self._window).alias(col)
            )
        return DataFrame(self._df._df.select(result_cols))

    def sum(self):
        """Calculate rolling sum."""
        result_cols = []
        for col in self._df.columns:
            result_cols.append(
                self._df._df[col].rolling_sum(window_size=self._window).alias(col)
            )
        return DataFrame(self._df._df.select(result_cols))

    def std(self):
        """Calculate rolling standard deviation."""
        result_cols = []
        for col in self._df.columns:
            result_cols.append(
                self._df._df[col].rolling_std(window_size=self._window).alias(col)
            )
        return DataFrame(self._df._df.select(result_cols))

    def max(self):
        """Calculate rolling maximum."""
        result_cols = []
        for col in self._df.columns:
            result_cols.append(
                self._df._df[col].rolling_max(window_size=self._window).alias(col)
            )
        return DataFrame(self._df._df.select(result_cols))

    def min(self):
        """Calculate rolling minimum."""
        result_cols = []
        for col in self._df.columns:
            result_cols.append(
                self._df._df[col].rolling_min(window_size=self._window).alias(col)
            )
        return DataFrame(self._df._df.select(result_cols))


class _GroupBy:
    """GroupBy object for grouped operations."""

    def __init__(self, polars_groupby, parent_df: DataFrame):
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

    def agg(self, *args, **kwargs):
        """
        Aggregate using one or more operations.

        Returns
        -------
        DataFrame
            Aggregated DataFrame
        """
        result = self._gb.agg(*args, **kwargs)
        return DataFrame(result)

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying Polars GroupBy object."""
        attr = getattr(self._gb, name)
        # If it's a method that returns a DataFrame, wrap it
        if callable(attr):

            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                if hasattr(result, "columns"):  # It's a DataFrame-like object
                    return DataFrame(result)
                return result

            return wrapper
        return attr
