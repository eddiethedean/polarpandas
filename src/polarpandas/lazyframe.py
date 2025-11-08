"""
LazyFrame implementation with pandas-compatible API built on Polars.

This module provides the LazyFrame class that wraps Polars LazyFrame and
provides a pandas-compatible interface with lazy execution. All operations
are deferred until materialization, allowing Polars to optimize the entire
query plan before execution.

The LazyFrame class supports:
- Lazy execution by default (deferred until .collect())
- Query optimization through Polars query planner
- Efficient processing of large datasets
- Full pandas API compatibility where implemented
- Direct access to Polars methods via delegation

Examples
--------
>>> import polarpandas as ppd
>>> import polars as pl
>>> lf = ppd.scan_csv("large_file.csv")
>>> result = lf.filter(pl.col("value") > 100).select(["name", "value"])
>>> df = result.collect()  # Execute optimized plan

Notes
-----
- Use LazyFrame for large datasets (>1M rows) or complex operations
- Operations are not executed until .collect() is called
- Query planner optimizes operations before execution
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import polars as pl

if TYPE_CHECKING:
    from .frame import DataFrame
    from .series import Series


class LazyFrame:
    """
    Two-dimensional lazy DataFrame for deferred execution and optimization.

    LazyFrame is the lazy execution variant of DataFrame in PolarPandas. It wraps
    a Polars LazyFrame and provides a pandas-compatible interface where operations
    are deferred until materialization via `.collect()`. This allows Polars to
    optimize the entire query plan before execution.

    Parameters
    ----------
    data : dict, list of dicts, pl.DataFrame, pl.LazyFrame, or None, optional
        Input data. Can be:
        - Dictionary of {column_name: [values]} pairs
        - List of dictionaries (each dict becomes a row)
        - Existing Polars DataFrame (converted to LazyFrame)
        - Existing Polars LazyFrame (used directly)
        - None for empty LazyFrame
    index : array-like, optional
        Index to use for resulting LazyFrame. Stored separately for pandas compatibility.

    Attributes
    ----------
    _df : pl.LazyFrame
        The underlying Polars LazyFrame.
    _index : list or None
        Stored index values for pandas compatibility.
    _index_name : str, tuple, or None
        Name(s) for the index.

    Examples
    --------
    >>> import polarpandas as ppd
    >>> import polars as pl
    >>> # From DataFrame
    >>> df = ppd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    >>> lf = df.lazy()
    >>> # From scan operation
    >>> lf = ppd.scan_csv("large_file.csv")
    >>> # Chain operations
    >>> result = lf.filter(pl.col("A") > 1).select(["A", "B"])
    >>> df_final = result.collect()  # Materialize when ready

    See Also
    --------
    DataFrame : For eager execution
    scan_csv, scan_parquet, scan_json : Lazy I/O operations

    Notes
    -----
    - All operations are deferred until `.collect()` is called
    - Query planner optimizes operations before execution
    - Use LazyFrame for large datasets or complex query chains
    - Materialization can be expensive; avoid calling `.collect()` in loops
    """

    _index: Optional[List[Any]]
    _index_name: Optional[Union[str, Tuple[str, ...]]]
    _columns_index: Optional[Any]
    _df: pl.LazyFrame
    _known_row_count: Optional[int]

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
        data: Optional[
            Union[Dict[str, Any], List[Any], pl.DataFrame, pl.LazyFrame]
        ] = None,
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
        provided_row_count = kwargs.pop("_known_row_count", None)
        requested_index = kwargs.pop("index", None)
        requested_index_name = kwargs.pop("index_name", None)

        self._columns_index = None

        if data is None:
            columns = kwargs.pop("columns", None)
            if columns is not None:
                self._df = pl.DataFrame({col: [] for col in columns}).lazy()
            else:
                self._df = pl.DataFrame().lazy()
            self._index = requested_index
            self._index_name = requested_index_name
            self._known_row_count = (
                provided_row_count if provided_row_count is not None else 0
            )
        elif isinstance(data, pl.LazyFrame):
            self._df = data
            self._index = requested_index
            self._index_name = requested_index_name
            self._known_row_count = provided_row_count
        elif isinstance(data, pl.DataFrame):
            self._df = data.lazy()
            self._index = requested_index
            self._index_name = requested_index_name
            default_rows = data.height
            self._known_row_count = (
                provided_row_count if provided_row_count is not None else default_rows
            )
        elif hasattr(data, "_df"):
            base_index = (
                requested_index
                if requested_index is not None
                else getattr(data, "_index", None)
            )
            base_index_name = (
                requested_index_name
                if requested_index_name is not None
                else getattr(data, "_index_name", None)
            )

            if isinstance(data._df, pl.DataFrame):
                polars_df = data._df  # type: ignore[attr-defined]
                self._df = polars_df.lazy()
                default_rows = len(data) if hasattr(data, "__len__") else polars_df.height
            else:
                self._df = data._df  # type: ignore[attr-defined]
                default_rows = provided_row_count

            self._index = base_index
            self._index_name = base_index_name
            self._known_row_count = (
                provided_row_count if provided_row_count is not None else default_rows
            )
        else:
            polars_df = pl.DataFrame(data, *args, **kwargs)
            self._df = polars_df.lazy()
            self._index = requested_index
            self._index_name = requested_index_name
            default_rows = polars_df.height
            self._known_row_count = (
                provided_row_count if provided_row_count is not None else default_rows
            )

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
        self._known_row_count = materialized.height
        return DataFrame(materialized, index=self._index, index_name=self._index_name)

    def __repr__(self) -> str:
        """String representation of the LazyFrame."""
        if self._known_row_count is not None:
            preview_target = min(self._known_row_count, 10)
            preview_df: Optional[pl.DataFrame] = None
            try:
                if self._known_row_count <= 10:
                    preview_df = self._df.collect()
                else:
                    preview_df = self._df.fetch(preview_target)
            except Exception:
                preview_df = None

            if preview_df is not None:
                header = (
                    f"LazyFrame(rows={self._known_row_count}, "
                    f"columns={len(preview_df.columns)})"
                )
                body = preview_df.__repr__()
                if self._known_row_count > preview_target:
                    remaining = self._known_row_count - preview_target
                    body = f"{body}\n… {remaining} more rows (not shown)"
                return f"{header}\n{body}"

        schema = self._df.collect_schema()
        column_names = schema.names()
        dtypes = schema.dtypes()
        preview_pairs = ", ".join(
            f"{name}: {dtype}" for name, dtype in zip(column_names[:4], dtypes[:4])
        )
        if len(column_names) > 4:
            preview_pairs += ", ..."
        plan_fn = getattr(self._df, "describe_plan", None)
        plan = plan_fn() if callable(plan_fn) else self._df.explain()
        return (
            "LazyFrame(\n"
            f"  columns={len(column_names)}, schema={{ {preview_pairs} }}\n"
            ")\n"
            f"{plan}"
        )

    def __str__(self) -> str:
        """String representation of the LazyFrame."""
        return self.__repr__()

    def __len__(self) -> int:
        """Length of the LazyFrame."""
        if self._known_row_count is not None:
            return self._known_row_count
        raise TypeError(
            "len() is not defined for LazyFrame without a known row count. "
            "Call '.collect()' to materialize rows."
        )

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
            elif isinstance(key, Series):  # type: ignore[unreachable]
                # Boolean indexing with Series
                materialized = self._materialize()
                polars_key = key._series
                filtered = materialized.filter(polars_key)
                return LazyFrame(filtered.lazy())
            elif (
                hasattr(key, "__iter__")
                and not isinstance(key, str)
                and not isinstance(key, list)
            ):
                # Boolean indexing with array-like (but not list, which is handled above)
                # Materialize to handle boolean indexing
                materialized = self._materialize()
                polars_key = key
                # Use filter for boolean indexing
                filtered = materialized.filter(polars_key)
                return LazyFrame(filtered.lazy())
            else:
                # Other key types - delegate to Polars
                return self._df.__getitem__(key)
        except Exception as e:
            # Convert Polars exceptions to pandas-compatible ones
            from polarpandas._exceptions import convert_to_keyerror

            converted = convert_to_keyerror(e)
            if converted is not e:
                raise converted from e
            raise

    @property
    def columns(self) -> List[str]:
        """Column names of the LazyFrame."""
        return self._df.collect_schema().names()

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the LazyFrame."""
        schema = self._df.collect_schema()
        if self._known_row_count is None:
            return (None, len(schema.names()))
        return (self._known_row_count, len(schema.names()))

    @property
    def height(self) -> int:
        """Number of rows in the LazyFrame."""
        if self._known_row_count is not None:
            return self._known_row_count
        raise TypeError(
            "height is not available for LazyFrame without materializing. "
            "Call '.collect()' first."
        )

    @property
    def width(self) -> int:
        """Number of columns in the LazyFrame."""
        return len(self._df.collect_schema().names())

    @property
    def dtypes(self) -> Dict[str, Any]:
        """Data types of columns."""
        schema = self._df.collect_schema()
        return dict(zip(schema.names(), schema.dtypes()))

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
        from typing import Literal, cast

        other_lazy: Optional[pl.LazyFrame] = None
        if isinstance(other, LazyFrame):
            other_lazy = other._df
        elif hasattr(other, "_df"):
            other_df = other._df  # type: ignore[attr-defined]
            if isinstance(other_df, pl.LazyFrame):
                other_lazy = other_df
            elif isinstance(other_df, pl.DataFrame):
                other_lazy = other_df.lazy()
        elif isinstance(other, pl.LazyFrame):
            other_lazy = other
        elif isinstance(other, pl.DataFrame):
            other_lazy = other.lazy()

        if other_lazy is None:
            raise TypeError(
                "Join partner must be a PolarPandas DataFrame/LazyFrame or a Polars "
                "DataFrame/LazyFrame."
            )

        valid_hows: Tuple[str, ...] = (
            "inner",
            "left",
            "right",
            "full",
            "semi",
            "anti",
            "cross",
            "outer",
        )
        if how not in valid_hows:
            raise ValueError(f"Unsupported join type '{how}'.")

        how_literal = cast(
            "Literal['inner','left','right','full','semi','anti','cross','outer']",
            how,
        )
        result_lazy = self._df.join(other_lazy, on=on, how=how_literal, **kwargs)
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
        schema = self._df.collect_schema()
        column_names = schema.names()
        dtypes = schema.dtypes()
        print(
            f"LazyFrame with {len(column_names)} columns; "
            "row count unknown until collect()."
        )
        for name, dtype in zip(column_names, dtypes):
            print(f"  - {name}: {dtype}")

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the underlying LazyFrame."""
        return getattr(self._df, name)
