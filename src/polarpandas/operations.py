"""
Data manipulation operations for PolarPandas.

Provides pandas-compatible functions for data manipulation and transformation.
"""

from typing import Any, List

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

        return DataFrame(pl.concat([obj._df for obj in objs], **kwargs))
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
