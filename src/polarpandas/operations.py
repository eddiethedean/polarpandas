"""
Data manipulation operations for PolarPandas.

This module provides pandas-compatible functions for data manipulation and
transformation, including concatenation, merging, pivoting, and dummy variable
creation. These functions operate on DataFrame and Series objects and return
new objects with the transformed data.

Functions
---------
concat : Concatenate DataFrames along specified axis
merge : Merge (join) two DataFrames
get_dummies : Convert categorical variables to dummy/indicator variables
pivot_table : Create pivot tables from DataFrames

Examples
--------
>>> import polarpandas as ppd
>>> df1 = ppd.DataFrame({"A": [1, 2]})
>>> df2 = ppd.DataFrame({"A": [3, 4]})
>>> result = ppd.concat([df1, df2])
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

from . import utils
from .frame import DataFrame
from .series import Series


def concat(objs: List[DataFrame], axis: int = 0, **kwargs: Any) -> DataFrame:
    """
    Concatenate DataFrames along specified axis.

    Parameters
    ----------
    objs : List[DataFrame]
        List of DataFrames to concatenate
    axis : int, default 0
        Axis to concatenate along (0 for rows, 1 for columns)
    **kwargs
        Additional arguments passed to Polars concat()

    Returns
    -------
    DataFrame
        Concatenated DataFrame

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df1 = ppd.DataFrame({"A": [1, 2]})
    >>> df2 = ppd.DataFrame({"A": [3, 4]})
    >>> result = ppd.concat([df1, df2])
    """
    if not objs:
        return DataFrame()

    if axis == 0:
        # Concatenate vertically (rows)
        import polars as pl

        # Check if all DataFrames have MultiIndex
        all_have_multiindex = all(
            obj._index is not None
            and len(obj._index) > 0
            and isinstance(obj._index[0], tuple)
            for obj in objs
        )

        result = DataFrame(pl.concat([obj._df for obj in objs], **kwargs))

        # Preserve MultiIndex if all DataFrames had it
        if all_have_multiindex:
            # Combine all indices
            combined_index = []
            combined_index_name = None
            for obj in objs:
                if obj._index is not None:
                    combined_index.extend(obj._index)
                    if combined_index_name is None:
                        combined_index_name = obj._index_name

            result._index = combined_index
            result._index_name = combined_index_name

        return result
    else:
        # Concatenate horizontally (columns)
        import polars as pl

        return DataFrame(
            pl.concat([obj._df for obj in objs], how="horizontal", **kwargs)
        )


def merge(left: DataFrame, right: DataFrame, **kwargs: Any) -> DataFrame:
    """
    Merge two DataFrames.

    Parameters
    ----------
    left : DataFrame
        Left DataFrame
    right : DataFrame
        Right DataFrame
    **kwargs
        Additional arguments passed to Polars join()

    Returns
    -------
    DataFrame
        Merged DataFrame

    Examples
    --------
    >>> import polarpandas as ppd
    >>> left = ppd.DataFrame({"key": [1, 2], "A": [1, 2]})
    >>> right = ppd.DataFrame({"key": [1, 2], "B": [3, 4]})
    >>> result = ppd.merge(left, right, on="key")
    """
    return left.merge(right, **kwargs)


def get_dummies(data: Any, **kwargs: Any) -> DataFrame:
    """
    Convert categorical variables into dummy/indicator variables.

    Parameters
    ----------
    data : DataFrame, Series, or list
        Data to convert
    **kwargs
        Additional arguments passed to Polars get_dummies()

    Returns
    -------
    DataFrame
        DataFrame with dummy variables

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.DataFrame({"category": ["A", "B", "A"]})
    >>> result = ppd.get_dummies(df)
    """
    if isinstance(data, DataFrame):
        return data.get_dummies(**kwargs)
    elif isinstance(data, Series):
        # Convert Series to DataFrame for get_dummies
        temp_df = DataFrame({"col": data._series})
        return temp_df.get_dummies(**kwargs)
    elif isinstance(data, list):
        # Convert list to DataFrame for get_dummies
        import polars as pl

        temp_df = DataFrame(pl.DataFrame({"col": data}))
        return temp_df.get_dummies(**kwargs)
    else:
        raise ValueError(f"Unsupported type for get_dummies: {type(data)}")


def pivot_table(
    data: DataFrame,
    values: str,
    index: str,
    columns: str,
    aggfunc: str = "mean",
    **kwargs: Any,
) -> DataFrame:
    """
    Create a pivot table from DataFrame.

    Parameters
    ----------
    data : DataFrame
        DataFrame to pivot
    values : str
        Column to aggregate
    index : str
        Column to use as index
    columns : str
        Column to use as columns
    aggfunc : str, default "mean"
        Aggregation function
    **kwargs
        Additional arguments passed to Polars pivot()

    Returns
    -------
    DataFrame
        Pivot table

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.DataFrame({
    ...     "A": ["foo", "foo", "bar"],
    ...     "B": ["one", "two", "one"],
    ...     "C": [1, 2, 3]
    ... })
    >>> result = ppd.pivot_table(df, values="C", index="A", columns="B")
    """
    return data.pivot_table(
        values=values, index=index, columns=columns, aggfunc=aggfunc, **kwargs
    )


def melt(
    frame: DataFrame,
    id_vars: Optional[Any] = None,
    value_vars: Optional[Any] = None,
    **kwargs: Any,
) -> DataFrame:
    """
    Unpivot DataFrame from wide to long format.

    Parameters
    ----------
    frame : DataFrame
        DataFrame to melt
    id_vars : str or list of str, optional
        Column(s) to use as identifier variables
    value_vars : str or list of str, optional
        Column(s) to unpivot
    **kwargs
        Additional arguments passed to melt()

    Returns
    -------
    DataFrame
        Melted DataFrame

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    >>> result = ppd.melt(df, id_vars=["A"], value_vars=["B", "C"])
    """
    return frame.melt(id_vars=id_vars, value_vars=value_vars, **kwargs)


def pivot(
    data: DataFrame,
    index: Optional[Any] = None,
    columns: Optional[Any] = None,
    values: Optional[Any] = None,
) -> DataFrame:
    """
    Reshape data (produce a "pivot" table).

    Parameters
    ----------
    data : DataFrame
        DataFrame to pivot
    index : str or list of str, optional
        Column(s) to use as index
    columns : str, optional
        Column to use for columns
    values : str, optional
        Column to use for values

    Returns
    -------
    DataFrame
        Pivoted DataFrame

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.DataFrame({
    ...     "A": ["foo", "foo", "bar"],
    ...     "B": ["one", "two", "one"],
    ...     "C": [1, 2, 3]
    ... })
    >>> result = ppd.pivot(df, index="A", columns="B", values="C")
    """
    return data.pivot(index=index, columns=columns, values=values)


def factorize(
    values: Any,
    sort: bool = False,
    na_sentinel: Optional[int] = -1,
    use_na_sentinel: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Encode object as an enumerated type or categorical variable.

    Parameters
    ----------
    values : Series or array-like
        Sequence to factorize
    sort : bool, default False
        Sort uniques and shuffle to maintain the relationship
    na_sentinel : int, default -1
        Value to mark "not found"
    use_na_sentinel : bool, default True
        If True, the sentinel -1 will be used for NaN values
    **kwargs
        Additional arguments

    Returns
    -------
    codes : ndarray
        Integer codes for values
    uniques : ndarray
        The unique valid values

    Examples
    --------
    >>> import polarpandas as ppd
    >>> codes, uniques = ppd.factorize(["a", "b", "a", "c"])
    """
    from .series import Series

    if isinstance(values, Series):
        return values.factorize(
            sort=sort,
            na_sentinel=na_sentinel,
            use_na_sentinel=use_na_sentinel,
            **kwargs,
        )
    else:
        # Convert to Series and factorize
        temp_series = Series(values)
        return temp_series.factorize(
            sort=sort,
            na_sentinel=na_sentinel,
            use_na_sentinel=use_na_sentinel,
            **kwargs,
        )


def cut(x: Any, bins: Any, labels: Optional[List[str]] = None, **kwargs: Any) -> Any:
    """
    Bin values into discrete intervals.

    Parameters
    ----------
    x : array-like, Series
        Input data to be binned
    bins : int or list
        Number of bins (int) or explicit bin edges (list)
    labels : list, optional
        Labels for the resulting bins
    **kwargs
        Additional arguments

    Returns
    -------
    Series
        Series with bin labels or categories

    Examples
    --------
    >>> import polarpandas as ppd
    >>> s = ppd.Series([1, 2, 3, 4, 5])
    >>> result = ppd.operations.cut(s, bins=3)
    """
    # If it's a Series, convert to list first
    values = x.tolist() if isinstance(x, Series) else x

    # Use the utils.cut function
    return utils.cut(values, bins=bins, labels=labels, **kwargs)


def qcut(
    x: Any,
    q: int,
    labels: Optional[Any] = None,
    retbins: bool = False,
    precision: int = 3,
    duplicates: str = "raise",
    **kwargs: Any,
) -> Any:
    """
    Quantile-based discretization function.

    Parameters
    ----------
    x : array-like
        Input array to be binned
    q : int
        Number of quantiles
    labels : array-like, optional
        Labels for the resulting bins
    retbins : bool, default False
        Whether to return the bins
    precision : int, default 3
        Precision for bin edges
    duplicates : str, default "raise"
        How to handle duplicates
    **kwargs
        Additional arguments

    Returns
    -------
    Series or tuple
        Binned data, optionally with bins

    Examples
    --------
    >>> import polarpandas as ppd
    >>> result = ppd.qcut([1, 2, 3, 4, 5], q=3)
    """
    import polars as pl

    from .series import Series

    # Convert to Polars Series
    pl_series = x._series if isinstance(x, Series) else pl.Series(x)

    # Use Polars qcut
    result = pl_series.qcut(q, labels=labels, **kwargs)
    result_series = Series(result)

    if retbins:
        # Calculate bins - quantile() expects a single float, so compute each individually
        bins = [pl_series.quantile(i / q) for i in range(q + 1)]
        return result_series, bins
    return result_series


def wide_to_long(
    df: DataFrame,
    stubnames: Any,
    i: Any,
    j: str = "j",
    sep: str = "",
    suffix: str = r"\d+",
    **kwargs: Any,
) -> DataFrame:
    """
    Unpivot a DataFrame from wide to long format.

    Parameters
    ----------
    df : DataFrame
        DataFrame to reshape
    stubnames : str or list of str
        The stub name(s)
    i : str or list of str
        Column(s) to use as id variables
    j : str, default "j"
        Name of sub-observation variable
    sep : str, default ""
        Separator in column names
    suffix : str, default r"\\d+"
        Regular expression capturing the suffix
    **kwargs
        Additional arguments

    Returns
    -------
    DataFrame
        Reshaped DataFrame

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.DataFrame({
    ...     "id": [1, 2],
    ...     "A2000": [1, 2],
    ...     "A2001": [3, 4]
    ... })
    >>> result = ppd.wide_to_long(df, stubnames="A", i="id", j="year")
    """
    # Simplified implementation using melt
    # This is a complex operation that would need more sophisticated parsing
    # For now, we'll use a basic approach

    # Get columns matching stubnames
    if isinstance(stubnames, str):
        stubnames = [stubnames]

    # Find columns that start with stubnames
    all_cols = df.columns
    id_cols = i if isinstance(i, list) else [i]

    # Collect value columns
    value_cols = []
    for stub in stubnames:
        for col in all_cols:
            if col.startswith(stub) and col not in id_cols:
                value_cols.append(col)

    # Use melt to reshape
    return df.melt(id_vars=id_cols, value_vars=value_cols, **kwargs)


def crosstab(
    index: Any,
    columns: Any,
    values: Optional[Any] = None,
    rownames: Optional[Any] = None,
    colnames: Optional[Any] = None,
    aggfunc: Optional[Any] = None,
    margins: bool = False,
    margins_name: str = "All",
    dropna: bool = True,
    normalize: bool = False,
    **kwargs: Any,
) -> DataFrame:
    """
    Compute a simple cross-tabulation of two (or more) factors.

    Parameters
    ----------
    index : array-like, Series, or list of arrays/Series
        Values to group by in the rows
    columns : array-like, Series, or list of arrays/Series
        Values to group by in the columns
    values : array-like, optional
        Array of values to aggregate according to the factors
    rownames : sequence, optional
        Names for row groups
    colnames : sequence, optional
        Names for column groups
    aggfunc : function, optional
        Aggregation function
    margins : bool, default False
        Add row/column margins
    margins_name : str, default "All"
        Name of the row/column margin
    dropna : bool, default True
        Do not include columns whose entries are all NaN
    normalize : bool, default False
        Normalize by dividing all values by the sum of values
    **kwargs
        Additional arguments

    Returns
    -------
    DataFrame
        Cross-tabulation table

    Examples
    --------
    >>> import polarpandas as ppd
    >>> result = ppd.crosstab([1, 2, 1], [3, 3, 4])
    """
    import polars as pl

    from .series import Series

    # Convert inputs to Series if needed
    if not isinstance(index, Series):
        index = Series(index)
    if not isinstance(columns, Series):
        columns = Series(columns)

    # Create DataFrame for pivot
    if values is not None:
        if not isinstance(values, Series):
            values = Series(values)
        df = DataFrame(
            pl.DataFrame(
                {
                    "index": index._series,
                    "columns": columns._series,
                    "values": values._series,
                }
            )
        )
        if aggfunc is None:
            aggfunc = "sum"
    else:
        # Count occurrences
        df = DataFrame(
            pl.DataFrame(
                {
                    "index": index._series,
                    "columns": columns._series,
                    "values": pl.Series([1] * len(index._series)),
                }
            )
        )
        aggfunc = "sum"

    # Pivot table
    result = df.pivot_table(
        values="values", index="index", columns="columns", aggfunc=aggfunc
    )

    # Handle margins
    if margins:
        # Add row margins
        row_margins = result.sum(axis=1)
        result[margins_name] = row_margins
        # Add column margins
        col_margins = result.sum(axis=0)
        result.loc[len(result)] = col_margins

    return result


def from_dummies(
    data: DataFrame,
    sep: str = "_",
    default_category: Optional[Any] = None,
    **kwargs: Any,
) -> DataFrame:
    """
    Create a DataFrame from dummy variables.

    Parameters
    ----------
    data : DataFrame
        DataFrame with dummy variables
    sep : str, default "_"
        Separator used in dummy column names
    default_category : Any, optional
        Default category for missing values
    **kwargs
        Additional arguments

    Returns
    -------
    DataFrame
        DataFrame with categorical columns

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.DataFrame({"A_0": [1, 0], "A_1": [0, 1]})
    >>> result = ppd.from_dummies(df, sep="_")
    """
    import polars as pl

    # Group columns by prefix
    prefixes: Dict[str, List[str]] = {}
    for col in data.columns:
        if sep in col:
            prefix = col.split(sep)[0]
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append(col)

    result_cols: Dict[str, List[Any]] = {}
    for prefix, cols in prefixes.items():
        # Find the column with value 1 for each row
        result_cols[prefix] = []
        for idx in range(len(data)):
            for col in cols:
                if data[col].iloc[idx] == 1:
                    suffix = col.split(sep, 1)[1] if sep in col else col
                    result_cols[prefix].append(suffix)
                    break
            else:
                result_cols[prefix].append(
                    default_category if default_category is not None else None
                )

    return DataFrame(pl.DataFrame(result_cols))


def lreshape(
    data: DataFrame, groups: Dict[str, List[str]], dropna: bool = True, **kwargs: Any
) -> DataFrame:
    """
    Reshape wide-format data to long format.

    Parameters
    ----------
    data : DataFrame
        DataFrame to reshape
    groups : dict
        Dictionary mapping new column names to lists of old column names
    dropna : bool, default True
        Drop rows with all NaN values
    **kwargs
        Additional arguments

    Returns
    -------
    DataFrame
        Reshaped DataFrame

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.DataFrame({"A1": [1, 2], "A2": [3, 4], "B1": [5, 6]})
    >>> result = ppd.lreshape(df, {"A": ["A1", "A2"], "B": ["B1"]})
    """
    from typing import cast

    # Use melt for each group
    result_dfs: List[DataFrame] = []
    for new_col, old_cols in groups.items():
        id_vars = [col for col in data.columns if col not in old_cols]
        melted = data.melt(
            id_vars=id_vars,
            value_vars=old_cols,
            var_name="variable",
            value_name=new_col,
        )
        if id_vars:
            result_dfs.append(melted)
        else:
            # Column selection returns DataFrame
            result_dfs.append(cast("DataFrame", melted[[new_col]]))

    # Combine results
    if result_dfs:
        result = result_dfs[0]
        for df in result_dfs[1:]:
            result = result.join(df, how="outer")
        return result
    return DataFrame()


def merge_asof(
    left: DataFrame,
    right: DataFrame,
    on: Optional[str] = None,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    left_by: Optional[Any] = None,
    right_by: Optional[Any] = None,
    suffixes: Tuple[str, str] = ("_x", "_y"),
    tolerance: Optional[Any] = None,
    allow_exact_matches: bool = True,
    direction: str = "backward",
    **kwargs: Any,
) -> DataFrame:
    """
    Perform an asof merge.

    Parameters
    ----------
    left : DataFrame
        Left DataFrame
    right : DataFrame
        Right DataFrame
    on : str, optional
        Column name to join on
    left_on : str, optional
        Left join key
    right_on : str, optional
        Right join key
    left_by : str or list, optional
        Group by columns in left DataFrame
    right_by : str or list, optional
        Group by columns in right DataFrame
    suffixes : tuple, default ("_x", "_y")
        Suffixes for overlapping columns
    tolerance : Any, optional
        Tolerance for asof merge
    allow_exact_matches : bool, default True
        Allow exact matches
    direction : str, default "backward"
        Direction of merge ("backward", "forward", "nearest")
    **kwargs
        Additional arguments

    Returns
    -------
    DataFrame
        Merged DataFrame

    Examples
    --------
    >>> import polarpandas as ppd
    >>> left = ppd.DataFrame({"time": [1, 5, 10], "value": [1, 2, 3]})
    >>> right = ppd.DataFrame({"time": [2, 6, 11], "value": [4, 5, 6]})
    >>> result = ppd.merge_asof(left, right, on="time")
    """

    # Determine join keys
    left_key = on or left_on or "time"
    right_key = on or right_on or "time"

    # Use Polars join_asof - strategy needs to be a specific literal
    from typing import Literal, cast

    valid_strategy = cast("Literal['backward', 'forward', 'nearest']", direction)

    result = left._df.join_asof(
        right._df,
        left_on=left_key,
        right_on=right_key,
        strategy=valid_strategy,
        tolerance=tolerance,
        **kwargs,
    )

    return DataFrame(result)


def merge_ordered(
    left: DataFrame,
    right: DataFrame,
    on: Optional[str] = None,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    left_by: Optional[Any] = None,
    right_by: Optional[Any] = None,
    fill_method: Optional[str] = None,
    suffixes: Tuple[str, str] = ("_x", "_y"),
    **kwargs: Any,
) -> DataFrame:
    """
    Perform a merge with ordered keys.

    Parameters
    ----------
    left : DataFrame
        Left DataFrame
    right : DataFrame
        Right DataFrame
    on : str, optional
        Column name to join on
    left_on : str, optional
        Left join key
    right_on : str, optional
        Right join key
    left_by : str or list, optional
        Group by columns in left DataFrame
    right_by : str or list, optional
        Group by columns in right DataFrame
    fill_method : str, optional
        Fill method for missing values
    suffixes : tuple, default ("_x", "_y")
        Suffixes for overlapping columns
    **kwargs
        Additional arguments

    Returns
    -------
    DataFrame
        Merged DataFrame

    Examples
    --------
    >>> import polarpandas as ppd
    >>> left = ppd.DataFrame({"key": [1, 2, 3], "A": [1, 2, 3]})
    >>> right = ppd.DataFrame({"key": [1, 2, 4], "B": [4, 5, 6]})
    >>> result = ppd.merge_ordered(left, right, on="key")
    """
    # Sort both DataFrames by join key
    from typing import cast

    join_key = on or left_on or right_on
    if join_key:
        # sort_values with inplace=False returns DataFrame
        left_sorted = cast("DataFrame", left.sort_values(join_key))
        right_sorted = cast("DataFrame", right.sort_values(join_key))
        # Perform outer join
        result = left_sorted.merge(
            right_sorted, on=join_key, how="outer", suffixes=suffixes, **kwargs
        )
        # Sort result
        return cast("DataFrame", result.sort_values(join_key))
    else:
        # Fallback to regular merge
        return left.merge(
            right,
            on=on,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes,
            **kwargs,
        )


def array(
    data: Any, dtype: Optional[Any] = None, copy: bool = True, **kwargs: Any
) -> Any:
    """
    Create an array.

    Parameters
    ----------
    data : array-like
        Data to convert to array
    dtype : dtype, optional
        Data type
    copy : bool, default True
        Copy data
    **kwargs
        Additional arguments

    Returns
    -------
    Series
        Series created from array

    Examples
    --------
    >>> import polarpandas as ppd
    >>> result = ppd.array([1, 2, 3])
    """
    from .series import Series

    return Series(data, dtype=dtype, **kwargs)


def col(name: str) -> Any:
    """
    Column selector (Polars expression).

    Parameters
    ----------
    name : str
        Column name

    Returns
    -------
    Expr
        Polars column expression

    Examples
    --------
    >>> import polarpandas as ppd
    >>> import polars as pl
    >>> expr = ppd.col("A")
    """
    import polars as pl

    return pl.col(name)


def eval(
    expr: str,
    parser: str = "pandas",
    engine: Optional[str] = None,
    truediv: bool = True,
    local_dict: Optional[Dict[str, Any]] = None,
    global_dict: Optional[Dict[str, Any]] = None,
    resolvers: Optional[Any] = None,
    level: int = 0,
    target: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """
    Evaluate a Python expression as a string using various backends.

    Parameters
    ----------
    expr : str
        Expression to evaluate
    parser : str, default "pandas"
        Parser to use
    engine : str, optional
        Engine to use
    truediv : bool, default True
        Use true division
    local_dict : dict, optional
        Local variables
    global_dict : dict, optional
        Global variables
    resolvers : Any, optional
        Resolvers
    level : int, default 0
        Level
    target : Any, optional
        Target object
    **kwargs
        Additional arguments

    Returns
    -------
    Any
        Result of evaluation

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.DataFrame({"A": [1, 2], "B": [3, 4]})
    >>> result = ppd.eval("A + B", target=df)
    """
    # Simplified implementation - use Python eval with safe context
    import builtins

    import polars as pl

    if target is not None:
        # If target is DataFrame, evaluate expression on it
        if isinstance(target, DataFrame):
            # Create evaluation context
            context = {}
            if local_dict:
                context.update(local_dict)
            if global_dict:
                context.update(global_dict)
            # Add DataFrame columns to context
            for col in target.columns:
                context[col] = target[col]._series
            # Evaluate expression using built-in eval
            try:
                result = builtins.eval(expr, {"__builtins__": {}}, context)
                # Check if result is a Polars Series
                if isinstance(result, pl.Series):
                    from .series import Series

                    return Series(result)
                return result
            except Exception as e:
                raise ValueError(f"Error evaluating expression: {e}") from e
        else:
            raise ValueError(f"Unsupported target type: {type(target)}")
    else:
        # Evaluate without target
        context = {}
        if local_dict:
            context.update(local_dict)
        if global_dict:
            context.update(global_dict)
        try:
            return builtins.eval(expr, {"__builtins__": {}}, context)
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {e}") from e


def json_normalize(
    data: Any,
    record_path: Optional[Any] = None,
    meta: Optional[Any] = None,
    meta_prefix: Optional[str] = None,
    record_prefix: Optional[str] = None,
    errors: str = "raise",
    sep: str = ".",
    max_level: Optional[int] = None,
    **kwargs: Any,
) -> DataFrame:
    """
    Normalize semi-structured JSON data into a flat table.

    Parameters
    ----------
    data : dict or list of dicts
        JSON data to normalize
    record_path : str or list, optional
        Path in each object to list of records
    meta : str or list, optional
        Fields to use as metadata
    meta_prefix : str, optional
        Prefix for metadata columns
    record_prefix : str, optional
        Prefix for record columns
    errors : str, default "raise"
        How to handle errors
    sep : str, default "."
        Separator for nested fields
    max_level : int, optional
        Maximum level to normalize
    **kwargs
        Additional arguments

    Returns
    -------
    DataFrame
        Normalized DataFrame

    Examples
    --------
    >>> import polarpandas as ppd
    >>> data = [{"name": "A", "values": [1, 2]}, {"name": "B", "values": [3, 4]}]
    >>> result = ppd.json_normalize(data)
    """
    try:
        import pandas as pd

        # Use pandas to normalize, then convert
        pd_df = pd.json_normalize(
            data=data,
            record_path=record_path,
            meta=meta,
            meta_prefix=meta_prefix,
            record_prefix=record_prefix,
            errors=errors,
            sep=sep,
            max_level=max_level,
            **kwargs,
        )
        return DataFrame(pd_df)
    except ImportError:
        raise NotImplementedError(
            "json_normalize() requires pandas.\n"
            "Workarounds:\n"
            "  - Install pandas: pip install pandas\n"
            "  - Use pandas: pd.json_normalize(data) then convert with polarpandas.DataFrame(df)"
        ) from None


# Configuration functions - simplified stubs
_OPTIONS: Dict[str, Any] = {}


def describe_option(pat: str, _print_desc: bool = True, **kwargs: Any) -> Optional[str]:
    """
    Print the description of one or more registered options.

    Parameters
    ----------
    pat : str
        Regexp pattern
    _print_desc : bool, default True
        Print description
    **kwargs
        Additional arguments

    Returns
    -------
    str or None
        Description if _print_desc=False

    Examples
    --------
    >>> import polarpandas as ppd
    >>> ppd.describe_option("display")
    """
    # Simplified stub - polarpandas doesn't have options system
    desc = f"Option '{pat}' is not available in polarpandas."
    if _print_desc:
        print(desc)
        return None
    return desc


def get_option(pat: str, **kwargs: Any) -> Any:
    """
    Get the value of a single option.

    Parameters
    ----------
    pat : str
        Regexp pattern
    **kwargs
        Additional arguments

    Returns
    -------
    Any
        Option value

    Examples
    --------
    >>> import polarpandas as ppd
    >>> value = ppd.get_option("display.max_rows")
    """
    # Simplified stub - return None for all options
    return _OPTIONS.get(pat)


def option_context(*args: Any, **kwargs: Any) -> Any:
    """
    Context manager for setting options.

    Parameters
    ----------
    *args
        Positional arguments
    **kwargs
        Options to set

    Returns
    -------
    contextmanager
        Context manager

    Examples
    --------
    >>> import polarpandas as ppd
    >>> with ppd.option_context("display.max_rows", 10):
    ...     print(df)
    """
    from contextlib import contextmanager

    @contextmanager
    def _option_context() -> Iterator[None]:
        # Store old values
        old_values = {}
        for key, value in kwargs.items():
            old_values[key] = _OPTIONS.get(key)
            _OPTIONS[key] = value
        try:
            yield
        finally:
            # Restore old values
            for key, old_value in old_values.items():
                if old_value is None:
                    _OPTIONS.pop(key, None)
                else:
                    _OPTIONS[key] = old_value

    return _option_context()


def reset_option(pat: str, **kwargs: Any) -> None:
    """
    Reset one or more options to their default value.

    Parameters
    ----------
    pat : str
        Regexp pattern
    **kwargs
        Additional arguments

    Examples
    --------
    >>> import polarpandas as ppd
    >>> ppd.reset_option("display")
    """
    # Simplified stub - remove matching options
    keys_to_remove = [key for key in _OPTIONS if pat in key]
    for key in keys_to_remove:
        _OPTIONS.pop(key, None)


def set_eng_float_format(
    accuracy: int = 3, use_eng_prefix: bool = False, **kwargs: Any
) -> None:
    """
    Set the engineering notation for the display of floating point numbers.

    Parameters
    ----------
    accuracy : int, default 3
        Number of decimal places
    use_eng_prefix : bool, default False
        Use engineering prefix
    **kwargs
        Additional arguments

    Examples
    --------
    >>> import polarpandas as ppd
    >>> ppd.set_eng_float_format(accuracy=2)
    """
    # Simplified stub - polarpandas doesn't have display options
    pass


def set_option(pat: str, value: Any, **kwargs: Any) -> None:
    """
    Set the value of a single option.

    Parameters
    ----------
    pat : str
        Regexp pattern
    value : Any
        Option value
    **kwargs
        Additional arguments

    Examples
    --------
    >>> import polarpandas as ppd
    >>> ppd.set_option("display.max_rows", 10)
    """
    # Simplified stub - store in options dict
    _OPTIONS[pat] = value


def show_versions(as_json: bool = False, **kwargs: Any) -> Optional[str]:
    """
    Print the versions of polarpandas and its dependencies.

    Parameters
    ----------
    as_json : bool, default False
        Return as JSON string
    **kwargs
        Additional arguments

    Returns
    -------
    str or None
        Version information as JSON if as_json=True

    Examples
    --------
    >>> import polarpandas as ppd
    >>> ppd.show_versions()
    """
    import json
    import sys

    versions = {
        "polarpandas": __import__("polarpandas").__version__,
        "polars": __import__("polars").__version__,
        "python": sys.version,
    }

    try:
        import pandas as pd

        versions["pandas"] = pd.__version__
    except ImportError:
        pass

    if as_json:
        return json.dumps(versions, indent=2)
    else:
        print("INSTALLED VERSIONS")
        print("------------------")
        for key, value in versions.items():
            print(f"{key}: {value}")
        return None


def test(verbose: bool = False, **kwargs: Any) -> None:
    """
    Run tests (placeholder).

    Parameters
    ----------
    verbose : bool, default False
        Verbose output
    **kwargs
        Additional arguments

    Examples
    --------
    >>> import polarpandas as ppd
    >>> ppd.test()
    """
    raise NotImplementedError(
        "test() is not implemented.\n"
        "Workarounds:\n"
        "  - Run pytest: pytest tests/\n"
        "  - Use unittest: python -m unittest discover tests"
    )


def infer_freq(index: Any, **kwargs: Any) -> Optional[str]:
    """
    Infer the most likely frequency given the input index.

    Parameters
    ----------
    index : Index or array-like
        Index to infer frequency from
    **kwargs
        Additional arguments

    Returns
    -------
    str or None
        Inferred frequency

    Examples
    --------
    >>> import polarpandas as ppd
    >>> freq = ppd.infer_freq(index)
    """
    # Simplified implementation
    try:
        import pandas as pd

        if hasattr(index, "to_pandas"):
            pd_index = index.to_pandas()
        elif hasattr(index, "_index"):
            pd_index = pd.Index(index._index)
        else:
            pd_index = pd.Index(index)
        freq_result: Optional[str] = pd.infer_freq(pd_index, **kwargs)
        return freq_result
    except ImportError:
        # Fallback: try to detect pattern
        if hasattr(index, "__len__") and len(index) > 1:
            # Very basic frequency detection
            return None
        return None
