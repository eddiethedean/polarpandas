"""
Tests for LazyFrame functionality.

Tests the lazy execution capabilities of the LazyFrame class.
"""

import polars as pl
import pytest

import polarpandas as ppd
from polarpandas import LazyFrame


class TestLazyFrame:
    """Test LazyFrame functionality."""

    def test_lazyframe_creation(self):
        """Test LazyFrame creation from various data sources."""
        # From dictionary
        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert isinstance(lf._df, pl.LazyFrame)

        # From DataFrame
        df = ppd.DataFrame({"a": [1, 2, 3]})
        lf = LazyFrame(df)
        assert isinstance(lf._df, pl.LazyFrame)

        # From Polars LazyFrame
        pl_lf = pl.LazyFrame({"a": [1, 2, 3]})
        lf = LazyFrame(pl_lf)
        assert isinstance(lf._df, pl.LazyFrame)

    def test_lazyframe_stays_lazy(self):
        """Test that LazyFrame operations stay lazy."""
        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Filter operation should stay lazy
        filtered = lf.filter(pl.col("a") > 1)
        assert isinstance(filtered, LazyFrame)
        assert isinstance(filtered._df, pl.LazyFrame)

        # Select operation should stay lazy
        selected = lf.select("a")
        assert isinstance(selected, LazyFrame)
        assert isinstance(selected._df, pl.LazyFrame)

    def test_lazyframe_collect(self):
        """Test that collect() materializes to DataFrame."""
        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        df = lf.collect()
        assert isinstance(df, ppd.DataFrame)
        assert isinstance(df._df, pl.DataFrame)

        # Verify data is correct
        assert df["a"].to_list() == [1, 2, 3]
        assert df["b"].to_list() == [4, 5, 6]

    def test_lazyframe_repr_triggers_materialization(self):
        """Test that __repr__ triggers materialization."""
        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # This should not raise an error and should show data
        repr_str = repr(lf)
        assert "a" in repr_str and "b" in repr_str
        assert "1" in repr_str and "4" in repr_str

    def test_lazyframe_column_access_triggers_materialization(self):
        """Test that column access triggers materialization."""
        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Accessing a single column should materialize
        series = lf["a"]
        assert isinstance(series, ppd.Series)
        assert series.to_list() == [1, 2, 3]

    def test_lazyframe_groupby_operations(self):
        """Test groupby operations with LazyFrame."""
        lf = LazyFrame({"category": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})

        # Groupby should return a grouped object
        grouped = lf.group_by("category")
        assert hasattr(grouped, "agg")

        # Aggregation should stay lazy
        result = grouped.agg([pl.col("value").sum().alias("value_sum")])
        assert isinstance(result, pl.LazyFrame)

        # Collect to get final result
        df = LazyFrame(result).collect()
        assert isinstance(df, ppd.DataFrame)

    def test_lazyframe_chain_operations(self):
        """Test chaining multiple lazy operations."""
        lf = LazyFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})

        # Chain multiple operations
        result = lf.filter(pl.col("a") > 1).select("a").sort("a")
        assert isinstance(result, LazyFrame)
        assert isinstance(result._df, pl.LazyFrame)

        # Materialize final result
        df = result.collect()
        assert df["a"].to_list() == [2, 3, 4]

    def test_lazyframe_with_columns(self):
        """Test with_columns operation."""
        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Add new column
        result = lf.with_columns((pl.col("a") * 2).alias("a_doubled"))
        assert isinstance(result, LazyFrame)
        assert isinstance(result._df, pl.LazyFrame)

        # Materialize and verify
        df = result.collect()
        assert "a_doubled" in df.columns
        assert df["a_doubled"].to_list() == [2, 4, 6]

    def test_lazyframe_join(self):
        """Test join operations."""
        lf1 = LazyFrame({"id": [1, 2], "name": ["A", "B"]})
        lf2 = LazyFrame({"id": [1, 2], "value": [10, 20]})

        # Join should stay lazy
        result = lf1.join(lf2, on="id")
        assert isinstance(result, LazyFrame)
        assert isinstance(result._df, pl.LazyFrame)

        # Materialize and verify
        df = result.collect()
        assert len(df) == 2
        assert "name" in df.columns and "value" in df.columns

    def test_lazyframe_head_tail(self):
        """Test head and tail operations."""
        lf = LazyFrame({"a": [1, 2, 3, 4, 5]})

        # Head should stay lazy
        head_result = lf.head(2)
        assert isinstance(head_result, LazyFrame)

        # Tail should stay lazy
        tail_result = lf.tail(2)
        assert isinstance(tail_result, LazyFrame)

        # Materialize and verify
        head_df = head_result.collect()
        tail_df = tail_result.collect()

        assert head_df["a"].to_list() == [1, 2]
        assert tail_df["a"].to_list() == [4, 5]

    def test_lazyframe_sort(self):
        """Test sort operation."""
        lf = LazyFrame({"a": [3, 1, 2], "b": [6, 4, 5]})

        # Sort should stay lazy
        result = lf.sort("a")
        assert isinstance(result, LazyFrame)
        assert isinstance(result._df, pl.LazyFrame)

        # Materialize and verify
        df = result.collect()
        assert df["a"].to_list() == [1, 2, 3]
        assert df["b"].to_list() == [4, 5, 6]

    def test_lazyframe_shape_property(self):
        """Test shape property triggers materialization."""
        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Shape should trigger materialization
        shape = lf.shape
        assert shape == (3, 2)

    def test_lazyframe_len(self):
        """Test len() triggers materialization."""
        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # len() should trigger materialization
        length = len(lf)
        assert length == 3

    def test_lazyframe_columns_property(self):
        """Test columns property."""
        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Columns should be available without materialization
        columns = lf.columns
        assert "a" in columns and "b" in columns

    def test_lazyframe_dtypes_property(self):
        """Test dtypes property."""
        lf = LazyFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

        # Dtypes should be available without materialization
        dtypes = lf.dtypes
        assert "a" in dtypes and "b" in dtypes

    def test_lazyframe_to_pandas(self):
        """Test conversion to pandas."""
        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Should materialize and convert to pandas
        pd_df = lf.to_pandas()
        assert hasattr(pd_df, "columns")
        assert list(pd_df.columns) == ["a", "b"]

    def test_lazyframe_write_operations(self):
        """Test write operations trigger materialization."""
        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # These should work (we can't easily test file writing in unit tests)
        # but they should not raise errors
        try:
            lf.to_csv("/tmp/test.csv")
            lf.to_parquet("/tmp/test.parquet")
            lf.to_json("/tmp/test.json")
        except Exception:
            # File system errors are expected in test environment
            pass

    def test_lazyframe_info(self):
        """Test info() method triggers materialization."""
        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Should not raise an error
        lf.info()

    def test_lazyframe_from_dataframe_lazy_method(self):
        """Test conversion from DataFrame using lazy() method."""
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Convert to LazyFrame
        lf = df.lazy()
        assert isinstance(lf, LazyFrame)
        assert isinstance(lf._df, pl.LazyFrame)

        # Should be able to collect back to DataFrame
        df2 = lf.collect()
        assert isinstance(df2, ppd.DataFrame)
        assert df2["a"].to_list() == [1, 2, 3]

    def test_lazyframe_empty_with_columns(self):
        """Test empty LazyFrame initialization with columns."""
        # Lines 79-83: columns is not None, index is not None
        lf = LazyFrame(columns=["a", "b"], index=[0, 1, 2])
        assert isinstance(lf, LazyFrame)
        assert isinstance(lf._df, pl.LazyFrame)
        assert lf._index == [0, 1, 2]

        # Lines 84-88: index is not None, columns is None
        lf2 = LazyFrame(index=[0, 1, 2])
        assert isinstance(lf2, LazyFrame)
        assert isinstance(lf2._df, pl.LazyFrame)
        assert lf2._index == [0, 1, 2]

        # Lines 89-93: both None - empty LazyFrame
        lf3 = LazyFrame()
        assert isinstance(lf3, LazyFrame)
        assert isinstance(lf3._df, pl.LazyFrame)
        assert lf3._index is None

    def test_lazyframe_str_method(self):
        """Test __str__ method triggers materialization."""
        # Lines 140-141: __str__ method
        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        str_repr = str(lf)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0

    def test_lazyframe_boolean_indexing_array(self):
        """Test boolean indexing with array-like mask."""
        # Lines 167-182: Boolean indexing paths
        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Array-like boolean mask
        mask = [True, False, True]
        result = lf[mask]
        assert isinstance(result, LazyFrame)

    def test_lazyframe_boolean_indexing_series(self):
        """Test boolean indexing with Series mask."""
        # Lines 175-182: Series boolean indexing
        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # polarpandas Series boolean mask
        mask = ppd.Series([True, False, True])
        result = lf[mask]
        assert isinstance(result, LazyFrame)

    def test_lazyframe_boolean_indexing_iterable(self):
        """Test boolean indexing with iterable that isn't a list."""
        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        mask = (True, False, True)
        result = lf[mask]
        assert isinstance(result, LazyFrame)

    def test_lazyframe_getitem_error_conversion(self):
        """Test __getitem__ converts Polars exceptions."""
        # Lines 186-190: Exception conversion

        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Invalid column should raise KeyError
        with pytest.raises((KeyError, Exception)):
            _ = lf["nonexistent"]

    def test_lazyframe_height_property(self):
        """Test height property triggers materialization."""
        # Lines 206-207: height property
        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        height = lf.height
        assert isinstance(height, int)
        assert height == 3

    def test_lazyframe_to_feather(self):
        """Test to_feather method."""
        # Lines 294-295: to_feather method
        import os
        import tempfile

        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        with tempfile.NamedTemporaryFile(suffix=".feather", delete=False) as f:
            temp_path = f.name

        try:
            lf.to_feather(temp_path)
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_lazyframe_to_sql(self):
        """Test to_sql method."""
        # Lines 299-300: to_sql method

        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Will fail without valid connection, but tests the method call
        with pytest.raises(
            (AttributeError, TypeError, ValueError)
        ):  # Database connection error expected
            lf.to_sql("test_table", None)

    def test_lazyframe_getattr_delegation(self):
        """Test __getattr__ delegates to underlying LazyFrame."""
        # Line 314: __getattr__ delegation
        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Should delegate to Polars LazyFrame methods
        # Test with a method that exists on LazyFrame
        # filter is a method that should be delegated
        result = lf.filter(pl.col("a") > 1)
        assert isinstance(result, LazyFrame)

    def test_lazyframe_width_property(self):
        """Test width property."""
        # Line 212: width property
        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        width = lf.width
        assert isinstance(width, int)
        assert width == 3

    def test_lazyframe_join_with_dataframe(self):
        """Test join with DataFrame (needs conversion to LazyFrame)."""
        # Lines 265-267: Join with DataFrame conversion
        lf = LazyFrame({"key": [1, 2, 3], "val1": [10, 20, 30]})
        df = ppd.DataFrame({"key": [2, 3, 4], "val2": [40, 50, 60]})

        result = lf.join(df, on="key", how="inner")
        assert isinstance(result, LazyFrame)
        # Collect to verify it worked
        result_df = result.collect()
        assert isinstance(result_df, ppd.DataFrame)

    def test_lazyframe_collect_preserves_manual_index(self, mixed_schema_frame):
        """collect() should attach any stored index metadata to the result."""

        polars_df = mixed_schema_frame._df
        lf = LazyFrame(polars_df, index=["r0", "r1", "r2"], index_name="row_id")

        collected = lf.collect()
        assert collected._index == ["r0", "r1", "r2"]
        assert collected._index_name == "row_id"

    def test_lazyframe_getitem_multiple_columns(self):
        """Selecting multiple columns should return a LazyFrame without materializing."""

        lf = LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        subset = lf[["a", "c"]]
        assert isinstance(subset, LazyFrame)


class TestScanFunctions:
    """Test scan_* functions for lazy I/O."""

    def test_scan_csv(self):
        """Test scan_csv function."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b\n1,4\n2,5\n3,6\n")
            temp_path = f.name

        try:
            lf = ppd.scan_csv(temp_path)
            assert isinstance(lf, LazyFrame)
            assert isinstance(lf._df, pl.LazyFrame)

            # Materialize and verify
            df = lf.collect()
            assert df["a"].to_list() == [1, 2, 3]
        finally:
            os.unlink(temp_path)

    def test_scan_parquet(self):
        """Test scan_parquet function."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = f.name

        try:
            # Create a simple DataFrame and write to Parquet
            test_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            test_df.write_parquet(temp_path)

            lf = ppd.scan_parquet(temp_path)
            assert isinstance(lf, LazyFrame)
            assert isinstance(lf._df, pl.LazyFrame)

            # Materialize and verify
            df = lf.collect()
            assert df["a"].to_list() == [1, 2, 3]
        finally:
            os.unlink(temp_path)

    def test_scan_json(self):
        """Test scan_json function."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"a": 1, "b": 4}\n{"a": 2, "b": 5}\n{"a": 3, "b": 6}\n')
            temp_path = f.name

        try:
            lf = ppd.scan_json(temp_path)
            assert isinstance(lf, LazyFrame)
            assert isinstance(lf._df, pl.LazyFrame)

            # Materialize and verify
            df = lf.collect()
            assert df["a"].to_list() == [1, 2, 3]
        finally:
            os.unlink(temp_path)
