"""
Aggregation helpers extracted from `polarpandas.frame`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import polars as pl

from polarpandas._dtypes import NUMERIC_DTYPES, NUMERIC_OR_BOOLEAN_DTYPES

if TYPE_CHECKING:
    from polarpandas.frame import DataFrame
    from polarpandas.series import Series


def _require_pandas(feature: str) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised when pandas missing
        raise ImportError(
            f"{feature} requires pandas to be installed. "
            "Install with `pip install polarpandas[pandas]`."
        ) from exc
    return pd


class DataFrameRolling:
    """
    Lightweight rolling-window helper mirroring pandas' Rolling API.
    """

    def __init__(
        self,
        owner: "DataFrame",
        window: Union[int, str],
        *,
        min_periods: Optional[int] = None,
        center: bool = False,
        win_type: Optional[str] = None,
        on: Optional[str] = None,
        axis: int = 0,
        closed: Optional[str] = None,
        method: Optional[str] = None,
    ) -> None:
        self._owner = owner
        self._window = window
        self._min_periods = min_periods
        self._center = center
        self._win_type = win_type
        self._on = on
        self._axis = axis
        self._closed = closed
        self._method = method

    def _apply(self, name: str, *args: Any, **kwargs: Any) -> "DataFrame":
        _require_pandas("DataFrame.rolling")
        pandas_df = self._owner.to_pandas()
        rolling_kwargs: Dict[str, Any] = {
            "window": self._window,
            "center": self._center,
            "method": self._method if self._method is not None else "single",
        }
        if self._min_periods is not None:
            rolling_kwargs["min_periods"] = self._min_periods
        if self._win_type is not None:
            rolling_kwargs["win_type"] = self._win_type
        if self._on is not None:
            rolling_kwargs["on"] = self._on
        if self._axis != 0:
            rolling_kwargs["axis"] = self._axis
        if self._closed is not None:
            rolling_kwargs["closed"] = self._closed

        roller = pandas_df.rolling(**rolling_kwargs)
        pandas_result = getattr(roller, name)(*args, **kwargs)
        return self._owner._from_pandas_dataframe(pandas_result)

    def mean(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return self._apply("mean", *args, **kwargs)

    def sum(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return self._apply("sum", *args, **kwargs)

    def std(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return self._apply("std", *args, **kwargs)

    def var(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return self._apply("var", *args, **kwargs)

    def min(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return self._apply("min", *args, **kwargs)

    def max(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return self._apply("max", *args, **kwargs)

    def count(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return self._apply("count", *args, **kwargs)


class DataFrameAggregationsMixin:
    """
    Mixin that provides aggregation helpers for the DataFrame class.
    """

    def _typed_column_names(
        self, numeric_only: bool, include_boolean: bool = False
    ) -> List[str]:
        if not numeric_only:
            return list(self.columns)  # type: ignore[attr-defined]
        allowed = NUMERIC_OR_BOOLEAN_DTYPES if include_boolean else NUMERIC_DTYPES
        return [
            column
            for column in self.columns  # type: ignore[attr-defined]
            if self._df[column].dtype in allowed  # type: ignore[attr-defined]
        ]

    def mean(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        level: Optional[Union[int, str, List[Union[int, str]]]] = None,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        from polarpandas.series import Series

        if level is not None and axis == 0:
            gb = self.groupby(level=level)  # type: ignore[attr-defined]
            expressions = [pl.col(col).mean().alias(col) for col in self.columns]  # type: ignore[attr-defined]
            result = gb.agg(expressions)
            return result

        if axis is None or axis == 1 or axis == "columns":
            if numeric_only:
                numeric_cols = self._typed_column_names(True)
                if not numeric_cols:
                    return Series(pl.Series([], dtype=pl.Float64))
                result_series = self._df.select(pl.mean_horizontal(numeric_cols))[
                    "literal"
                ]  # type: ignore[attr-defined]
            else:
                result_series = self._df.select(pl.mean_horizontal(self.columns))[
                    "literal"
                ]  # type: ignore[attr-defined]
            index = (
                getattr(self, "_index", None)
                if getattr(self, "_index", None) is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)

        if numeric_only:
            numeric_cols = self._typed_column_names(True)
            if not numeric_cols:
                return Series(pl.Series([], dtype=pl.Float64))
            result_pl = self._df.select(
                [pl.col(col).mean() for col in numeric_cols]
            )  # type: ignore[attr-defined]
        else:
            result_pl = self._df.select(
                [pl.col(col).mean() for col in self.columns]
            )  # type: ignore[attr-defined]
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
        from polarpandas.series import Series

        if level is not None and axis == 0:
            gb = self.groupby(level=level)  # type: ignore[attr-defined]
            expressions = [pl.col(col).min().alias(col) for col in self.columns]  # type: ignore[attr-defined]
            result = gb.agg(expressions)
            return result

        if axis is None or axis == 1 or axis == "columns":
            if numeric_only:
                numeric_cols = self._typed_column_names(True)
                if not numeric_cols:
                    return Series(pl.Series([], dtype=pl.Float64))
                result_series = self._df.select(pl.min_horizontal(numeric_cols))[
                    "literal"
                ]  # type: ignore[attr-defined]
            else:
                result_series = self._df.select(pl.min_horizontal(self.columns))[
                    "literal"
                ]  # type: ignore[attr-defined]
            index = (
                getattr(self, "_index", None)
                if getattr(self, "_index", None) is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)

        if numeric_only:
            numeric_cols = self._typed_column_names(True)
            if not numeric_cols:
                return Series(pl.Series([], dtype=pl.Float64))
            result_pl = self._df.select(
                [pl.col(col).min() for col in numeric_cols]
            )  # type: ignore[attr-defined]
        else:
            result_pl = self._df.select(
                [pl.col(col).min() for col in self.columns]
            )  # type: ignore[attr-defined]
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
        from polarpandas.series import Series

        if level is not None and axis == 0:
            gb = self.groupby(level=level)  # type: ignore[attr-defined]
            expressions = [pl.col(col).max().alias(col) for col in self.columns]  # type: ignore[attr-defined]
            result = gb.agg(expressions)
            return result

        if axis is None or axis == 1 or axis == "columns":
            if numeric_only:
                numeric_cols = self._typed_column_names(True)
                if not numeric_cols:
                    return Series(pl.Series([], dtype=pl.Float64))
                result_series = self._df.select(pl.max_horizontal(numeric_cols))[
                    "literal"
                ]  # type: ignore[attr-defined]
            else:
                result_series = self._df.select(pl.max_horizontal(self.columns))[
                    "literal"
                ]  # type: ignore[attr-defined]
            index = (
                getattr(self, "_index", None)
                if getattr(self, "_index", None) is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)

        if numeric_only:
            numeric_cols = self._typed_column_names(True)
            if not numeric_cols:
                return Series(pl.Series([], dtype=pl.Float64))
            result_pl = self._df.select(
                [pl.col(col).max() for col in numeric_cols]
            )  # type: ignore[attr-defined]
        else:
            result_pl = self._df.select(
                [pl.col(col).max() for col in self.columns]
            )  # type: ignore[attr-defined]
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
        from polarpandas.series import Series

        if level is not None and axis == 0:
            gb = self.groupby(level=level)  # type: ignore[attr-defined]
            expressions = [pl.col(col).median().alias(col) for col in self.columns]  # type: ignore[attr-defined]
            result = gb.agg(expressions)
            return result

        if axis is None or axis == 1 or axis == "columns":
            target_cols = (
                self._typed_column_names(True)
                if numeric_only
                else list(self.columns)  # type: ignore[attr-defined]
            )
            if numeric_only and not target_cols:
                return Series(pl.Series([], dtype=pl.Float64))
            result_series = self._df.select(  # type: ignore[attr-defined]
                pl.concat_list(target_cols).list.median().alias("literal")
            )[
                "literal"
            ]
            index = (
                getattr(self, "_index", None)
                if getattr(self, "_index", None) is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)

        target_cols = (
            self._typed_column_names(True)
            if numeric_only
            else list(self.columns)  # type: ignore[attr-defined]
        )
        if numeric_only and not target_cols:
            return Series(pl.Series([], dtype=pl.Float64))
        result_pl = self._df.select(  # type: ignore[attr-defined]
            [pl.col(col).median() for col in target_cols]
        )
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
        from polarpandas.series import Series

        if level is not None and axis == 0:
            gb = self.groupby(level=level)  # type: ignore[attr-defined]
            expressions = [
                pl.col(col).std(ddof=ddof).alias(col) for col in self.columns  # type: ignore[attr-defined]
            ]
            result = gb.agg(expressions)
            return result

        if axis is None or axis == 1 or axis == "columns":
            target_cols = (
                self._typed_column_names(True)
                if numeric_only
                else list(self.columns)  # type: ignore[attr-defined]
            )
            if numeric_only and not target_cols:
                return Series(pl.Series([], dtype=pl.Float64))
            result_series = self._df.select(  # type: ignore[attr-defined]
                pl.concat_list(target_cols).list.std(ddof=ddof).alias("literal")
            )["literal"]
            index = (
                getattr(self, "_index", None)
                if getattr(self, "_index", None) is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)

        target_cols = (
            self._typed_column_names(True)
            if numeric_only
            else list(self.columns)  # type: ignore[attr-defined]
        )
        if numeric_only and not target_cols:
            return Series(pl.Series([], dtype=pl.Float64))
        result_pl = self._df.select(  # type: ignore[attr-defined]
            [pl.col(col).std(ddof=ddof) for col in target_cols]
        )
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
        from polarpandas.series import Series

        if level is not None and axis == 0:
            gb = self.groupby(level=level)  # type: ignore[attr-defined]
            expressions = [
                pl.col(col).var(ddof=ddof).alias(col) for col in self.columns  # type: ignore[attr-defined]
            ]
            result = gb.agg(expressions)
            return result

        if axis is None or axis == 1 or axis == "columns":
            target_cols = (
                self._typed_column_names(True)
                if numeric_only
                else list(self.columns)  # type: ignore[attr-defined]
            )
            if numeric_only and not target_cols:
                return Series(pl.Series([], dtype=pl.Float64))
            result_series = self._df.select(  # type: ignore[attr-defined]
                pl.concat_list(target_cols).list.var(ddof=ddof).alias("literal")
            )["literal"]
            index = (
                getattr(self, "_index", None)
                if getattr(self, "_index", None) is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)

        if numeric_only:
            numeric_cols = self._typed_column_names(True)
            if not numeric_cols:
                return Series(pl.Series([], dtype=pl.Float64))
            result_pl = self._df.select(  # type: ignore[attr-defined]
                [pl.col(col).var(ddof=ddof) for col in numeric_cols]
            )
        else:
            result_pl = self._df.select(  # type: ignore[attr-defined]
                [pl.col(col).var(ddof=ddof) for col in self.columns]
            )
        values = [result_pl[col].to_list()[0] for col in result_pl.columns]
        return Series(values, index=result_pl.columns)

    def sum(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        level: Optional[Union[int, str, List[Union[int, str]]]] = None,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        from polarpandas.series import Series

        if level is not None and axis == 0:
            gb = self.groupby(level=level)  # type: ignore[attr-defined]
            expressions = [pl.col(col).sum().alias(col) for col in self.columns]  # type: ignore[attr-defined]
            result = gb.agg(expressions)
            return result

        if axis is None or axis == 1 or axis == "columns":
            target_cols = (
                self._typed_column_names(True)
                if numeric_only
                else list(self.columns)  # type: ignore[attr-defined]
            )
            if numeric_only and not target_cols:
                return Series(pl.Series([], dtype=pl.Float64))
            result_series = self._df.select(  # type: ignore[attr-defined]
                pl.sum_horizontal(target_cols).alias("literal")
            )[
                "literal"
            ]
            index = (
                getattr(self, "_index", None)
                if getattr(self, "_index", None) is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)

        target_cols = (
            self._typed_column_names(True)
            if numeric_only
            else list(self.columns)  # type: ignore[attr-defined]
        )
        if numeric_only and not target_cols:
            return Series(pl.Series([], dtype=pl.Float64))
        result_pl = self._df.select(  # type: ignore[attr-defined]
            [pl.col(col).sum() for col in target_cols]
        )
        values = [result_pl[col].to_list()[0] for col in result_pl.columns]
        return Series(values, index=result_pl.columns)

    def count(
        self,
        axis: Optional[Union[int, Literal["index", "columns"]]] = 0,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Union["Series", Any]:
        from polarpandas.series import Series

        if axis is None or axis == 1 or axis == "columns":
            target_cols = (
                self._typed_column_names(True, include_boolean=True)
                if numeric_only
                else list(self.columns)  # type: ignore[attr-defined]
            )
            if numeric_only and not target_cols:
                return Series(pl.Series([], dtype=pl.UInt32))
            expressions = [
                pl.col(col).is_not_null().cast(pl.UInt32) for col in target_cols
            ]
            result_series = self._df.select(  # type: ignore[attr-defined]
                pl.sum_horizontal(expressions).alias("literal")
            )[
                "literal"
            ]
            index = (
                getattr(self, "_index", None)
                if getattr(self, "_index", None) is not None
                else list(range(len(result_series)))
            )
            return Series(result_series, index=index)

        target_cols = (
            self._typed_column_names(True, include_boolean=True)
            if numeric_only
            else list(self.columns)  # type: ignore[attr-defined]
        )
        if numeric_only and not target_cols:
            return Series(pl.Series([], dtype=pl.UInt32))
        result_pl = self._df.select(  # type: ignore[attr-defined]
            [pl.col(col).count() for col in target_cols]
        )
        values = [result_pl[col].to_list()[0] for col in result_pl.columns]
        return Series(values, index=result_pl.columns)

    # ------------------------------------------------------------------
    # Pandas-backed helpers
    # ------------------------------------------------------------------

    def _from_pandas_dataframe(self, pandas_df: Any) -> "DataFrame":
        """Convert a pandas.DataFrame back into a polarpandas.DataFrame."""

        pd = _require_pandas("pandas conversion")
        from polarpandas.frame import DataFrame

        try:
            polars_df = pl.from_pandas(pandas_df, include_index=False)
        except Exception:
            # Fallback for heterogeneous column data that Arrow cannot coerce
            series_list: List[pl.Series] = []
            for column in pandas_df.columns:
                values = pandas_df[column].tolist()
                series_name = column if isinstance(column, str) else str(column)
                try:
                    series_list.append(pl.Series(series_name, values))
                except Exception:
                    series_list.append(pl.Series(series_name, values, dtype=pl.Object))
            polars_df = pl.DataFrame(series_list)
        result = DataFrame(polars_df)

        index = pandas_df.index
        if isinstance(index, pd.MultiIndex):
            result._index = [tuple(item) for item in index.tolist()]
            names = (
                tuple(index.names)
                if any(name is not None for name in index.names)
                else None
            )
            result._index_name = names
        elif (
            isinstance(index, pd.RangeIndex)
            and index.start == 0
            and index.step == 1
            and len(index) == len(pandas_df)
        ):
            result._index = None
            result._index_name = None
        else:
            result._index = index.tolist()
            result._index_name = index.name

        columns = pandas_df.columns
        if hasattr(columns, "copy"):
            result._columns_index = columns.copy()
        return result

    def _pandas_call(
        self,
        feature: str,
        *,
        method: str,
        call_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "DataFrame":
        pd = _require_pandas(feature)
        pandas_df = self.to_pandas()  # type: ignore[attr-defined]
        pandas_method = getattr(pandas_df, method)
        pandas_result = pandas_method(**(call_kwargs or {}))
        if isinstance(pandas_result, pd.Series):
            pandas_result = pandas_result.to_frame()
        return self._from_pandas_dataframe(pandas_result)

    def corr(
        self,
        method: str = "pearson",
        min_periods: Optional[int] = None,
        numeric_only: Optional[bool] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        supported_methods = {"pearson"}
        if method not in supported_methods:
            raise NotImplementedError(
                f"corr(method='{method}') is not implemented; supported methods: {sorted(supported_methods)}"
            )

        call_kwargs: Dict[str, Any] = {"method": method}
        if min_periods is not None:
            call_kwargs["min_periods"] = min_periods
        if numeric_only is not None:
            call_kwargs["numeric_only"] = numeric_only
        call_kwargs.update(kwargs)
        return self._pandas_call("DataFrame.corr", method="corr", call_kwargs=call_kwargs)

    def cov(
        self,
        min_periods: Optional[int] = None,
        ddof: int = 1,
        numeric_only: Optional[bool] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        call_kwargs: Dict[str, Any] = {"ddof": ddof}
        if min_periods is not None:
            call_kwargs["min_periods"] = min_periods
        if numeric_only is not None:
            call_kwargs["numeric_only"] = numeric_only
        call_kwargs.update(kwargs)
        return self._pandas_call("DataFrame.cov", method="cov", call_kwargs=call_kwargs)

    def rank(
        self,
        axis: Union[int, Literal["index", "columns"]] = 0,
        method: str = "average",
        numeric_only: Optional[bool] = None,
        na_option: str = "keep",
        ascending: bool = True,
        pct: bool = False,
        **kwargs: Any,
    ) -> "DataFrame":
        call_kwargs: Dict[str, Any] = {
            "axis": axis,
            "method": method,
            "na_option": na_option,
            "ascending": ascending,
            "pct": pct,
        }
        if numeric_only is not None:
            call_kwargs["numeric_only"] = numeric_only
        call_kwargs.update(kwargs)
        return self._pandas_call("DataFrame.rank", method="rank", call_kwargs=call_kwargs)

    def diff(
        self,
        periods: int = 1,
        axis: Union[int, Literal["index", "columns"]] = 0,
        **kwargs: Any,
    ) -> "DataFrame":
        call_kwargs: Dict[str, Any] = {"periods": periods, "axis": axis}
        call_kwargs.update(kwargs)
        return self._pandas_call("DataFrame.diff", method="diff", call_kwargs=call_kwargs)

    def pct_change(
        self,
        periods: int = 1,
        fill_method: str = "pad",
        limit: Optional[int] = None,
        freq: Optional[Any] = None,
        **kwargs: Any,
    ) -> "DataFrame":
        call_kwargs: Dict[str, Any] = {
            "periods": periods,
            "fill_method": fill_method,
            "limit": limit,
            "freq": freq,
        }
        call_kwargs.update(kwargs)
        return self._pandas_call(
            "DataFrame.pct_change", method="pct_change", call_kwargs=call_kwargs
        )

    def cumsum(
        self,
        axis: Union[int, Literal["index", "columns"]] = 0,
        skipna: bool = True,
        **kwargs: Any,
    ) -> "DataFrame":
        call_kwargs: Dict[str, Any] = {"axis": axis, "skipna": skipna}
        call_kwargs.update(kwargs)
        return self._pandas_call("DataFrame.cumsum", method="cumsum", call_kwargs=call_kwargs)

    def cumprod(
        self,
        axis: Union[int, Literal["index", "columns"]] = 0,
        skipna: bool = True,
        **kwargs: Any,
    ) -> "DataFrame":
        call_kwargs: Dict[str, Any] = {"axis": axis, "skipna": skipna}
        call_kwargs.update(kwargs)
        return self._pandas_call("DataFrame.cumprod", method="cumprod", call_kwargs=call_kwargs)

    def cummax(
        self,
        axis: Union[int, Literal["index", "columns"]] = 0,
        skipna: bool = True,
        **kwargs: Any,
    ) -> "DataFrame":
        call_kwargs: Dict[str, Any] = {"axis": axis, "skipna": skipna}
        call_kwargs.update(kwargs)
        return self._pandas_call("DataFrame.cummax", method="cummax", call_kwargs=call_kwargs)

    def cummin(
        self,
        axis: Union[int, Literal["index", "columns"]] = 0,
        skipna: bool = True,
        **kwargs: Any,
    ) -> "DataFrame":
        call_kwargs: Dict[str, Any] = {"axis": axis, "skipna": skipna}
        call_kwargs.update(kwargs)
        return self._pandas_call("DataFrame.cummin", method="cummin", call_kwargs=call_kwargs)

    def rolling(
        self,
        window: Union[int, str],
        min_periods: Optional[int] = None,
        center: bool = False,
        win_type: Optional[str] = None,
        on: Optional[str] = None,
        axis: int = 0,
        closed: Optional[str] = None,
        method: Optional[str] = None,
    ) -> DataFrameRolling:
        return DataFrameRolling(
            self,  # type: ignore[arg-type]
            window,
            min_periods=min_periods,
            center=center,
            win_type=win_type,
            on=on,
            axis=axis,
            closed=closed,
            method=method,
        )

