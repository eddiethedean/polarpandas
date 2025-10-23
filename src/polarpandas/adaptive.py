"""
OPTIONAL: Adaptive backend that automatically chooses pandas vs polars.

This is an EXPERIMENTAL feature for advanced users. Based on benchmarks,
this adds complexity without significant benefit (polarpandas wins 87% of cases).

RECOMMENDATION: Use regular polarpandas instead!
"""

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


if PANDAS_AVAILABLE:
    from polarpandas import DataFrame as PolarPandasDataFrame

    class AdaptiveDataFrame:
        """
        Experimental: Automatically choose between pandas and polarpandas.

        This class monitors performance and switches between backends.

        WARNING: This adds complexity. Benchmarks show polarpandas wins
        87% of operations, so using it consistently is simpler and better.
        """

        # Thresholds for switching (based on benchmarks)
        SMALL_DATA_THRESHOLD = 1000  # Below this, pandas might be faster for some ops

        def __init__(self, data, backend="auto"):
            """
            Initialize with automatic or manual backend selection.

            Parameters
            ----------
            data : dict, list, or DataFrame
                Data to initialize
            backend : {'auto', 'pandas', 'polars'}, default 'auto'
                Which backend to use
            """
            self._backend = backend

            # Create both if auto mode
            if backend == "auto":
                self._pd_df = (
                    pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
                )
                self._ppd_df = (
                    PolarPandasDataFrame(data)
                    if not isinstance(data, PolarPandasDataFrame)
                    else data
                )
                self._size = len(self._pd_df)
                # Choose based on size
                self._current = "polarpandas"  # polarpandas wins 87% of time
            elif backend == "pandas":
                self._pd_df = (
                    pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
                )
                self._ppd_df = None  # type: ignore
                self._current = "pandas"
            else:
                self._ppd_df = (
                    PolarPandasDataFrame(data)
                    if not isinstance(data, PolarPandasDataFrame)
                    else data
                )
                self._pd_df = None  # type: ignore
                self._current = "polarpandas"

        def _get_current_df(self):
            """Get the currently active DataFrame."""
            if self._current == "pandas":
                return self._pd_df
            else:
                return self._ppd_df

        def __getattr__(self, name):
            """Delegate to the active backend."""
            return getattr(self._get_current_df(), name)

        def __getitem__(self, key):
            """Column access."""
            return self._get_current_df()[key]

        def __setitem__(self, key, value):
            """Column assignment."""
            if self._current == "pandas":
                self._pd_df[key] = value
            else:
                self._ppd_df[key] = value

        def use_pandas(self):
            """Switch to pandas backend."""
            if not PANDAS_AVAILABLE:
                raise ImportError("pandas not installed")

            if self._current == "polarpandas" and self._ppd_df is not None:
                # Convert polarpandas to pandas
                self._pd_df = pd.DataFrame(self._ppd_df.to_dict())

            self._current = "pandas"
            return self

        def use_polarpandas(self):
            """Switch to polarpandas backend."""
            if self._current == "pandas" and self._pd_df is not None:
                # Convert pandas to polarpandas
                self._ppd_df = PolarPandasDataFrame(dict(self._pd_df.to_dict()))  # type: ignore

            self._current = "polarpandas"
            return self

        @property
        def backend(self):
            """Get current backend."""
            return self._current

    def create_adaptive_dataframe(data, **kwargs):
        """
        Create an adaptive DataFrame.

        WARNING: This is experimental. Use regular polarpandas instead!

        Examples
        --------
        >>> from polarpandas.adaptive import create_adaptive_dataframe
        >>> df = create_adaptive_dataframe({"a": [1, 2, 3]})
        >>> df.use_pandas()  # Switch to pandas
        >>> df.use_polarpandas()  # Switch to polarpandas
        """
        return AdaptiveDataFrame(data, **kwargs)

else:

    def create_adaptive_dataframe(data, **kwargs):
        """
        Adaptive mode requires pandas to be installed.

        Install pandas with: pip install pandas

        However, we recommend just using regular polarpandas!
        """
        raise ImportError(
            "Adaptive mode requires pandas. However, benchmarks show "
            "polarpandas wins 87% of operations, so just use polarpandas directly!"
        )
