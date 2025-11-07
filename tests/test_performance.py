"""
Performance and regression tests.

This test file ensures polarpandas maintains good performance
and doesn't regress on critical operations.
"""

import time

import polars as pl

import polarpandas as ppd


class TestPerformanceBasics:
    """Basic performance tests to ensure no major regressions."""

    def test_large_dataframe_creation(self):
        """Test creating large DataFrame is reasonably fast."""
        start = time.time()

        # Create DataFrame with 100k rows
        df = ppd.DataFrame({"A": list(range(100000)), "B": list(range(100000, 200000))})

        elapsed = time.time() - start

        assert len(df) == 100000
        # Should complete in under 2 seconds
        assert elapsed < 2.0

    def test_large_groupby_performance(self):
        """Test groupby on large DataFrame."""
        # Create DataFrame with 50k rows
        df = ppd.DataFrame(
            {"group": [i % 100 for i in range(50000)], "value": list(range(50000))}
        )

        start = time.time()
        result = df.groupby("group").agg([pl.col("value").sum().alias("total")])
        elapsed = time.time() - start

        assert len(result) == 100
        # Should complete in under 1 second
        assert elapsed < 1.0

    def test_large_concat_performance(self):
        """Test concatenating many DataFrames."""
        # Create 100 small DataFrames
        dfs = [ppd.DataFrame({"A": [i]}) for i in range(100)]

        start = time.time()
        result = ppd.concat(dfs)
        elapsed = time.time() - start

        assert len(result) == 100
        # Should complete quickly
        assert elapsed < 1.0

    def test_column_selection_performance(self):
        """Test column selection on wide DataFrame."""
        # Create DataFrame with 100 columns
        data = {f"col_{i}": list(range(1000)) for i in range(100)}
        df = ppd.DataFrame(data)

        start = time.time()
        result = df[["col_0", "col_50", "col_99"]]
        elapsed = time.time() - start

        assert len(result.columns) == 3
        # Should be instant
        assert elapsed < 0.5


class TestFilterPerformance:
    """Performance tests for filtering operations."""

    def test_boolean_filter_performance(self):
        """Test boolean filtering on large DataFrame."""
        df = ppd.DataFrame({"A": list(range(100000))})

        start = time.time()
        result = df[df["A"] > 50000]
        elapsed = time.time() - start

        assert len(result) < len(df)
        assert elapsed < 1.0

    def test_multiple_filter_conditions(self):
        """Test filtering with multiple conditions."""
        df = ppd.DataFrame({"A": list(range(10000)), "B": list(range(10000, 20000))})

        start = time.time()
        result = df[(df["A"] > 5000) & (df["B"] < 15000)]
        elapsed = time.time() - start

        assert len(result) < len(df)
        assert elapsed < 1.0


class TestAggregationPerformance:
    """Performance tests for aggregations."""

    def test_sum_performance(self):
        """Test sum on large DataFrame."""
        df = ppd.DataFrame({"A": list(range(100000)), "B": list(range(100000))})

        start = time.time()
        result = df.sum()
        elapsed = time.time() - start

        assert result["A"] > 0
        assert elapsed < 0.5

    def test_mean_performance(self):
        """Test mean computation."""
        df = ppd.DataFrame({"values": list(range(100000))})

        start = time.time()
        result = df.mean()
        elapsed = time.time() - start

        assert result["values"] > 0
        assert elapsed < 0.5


class TestSortingPerformance:
    """Performance tests for sorting."""

    def test_sort_large_dataframe(self):
        """Test sorting large DataFrame."""
        import random

        random.seed(42)

        df = ppd.DataFrame({"A": [random.randint(1, 1000) for _ in range(10000)]})

        start = time.time()
        result = df.sort_values("A")
        elapsed = time.time() - start

        assert len(result) == len(df)
        assert elapsed < 1.0


class TestMergePerformance:
    """Performance tests for merge operations."""

    def test_merge_large_dataframes(self):
        """Test merging large DataFrames."""
        df1 = ppd.DataFrame({"key": list(range(10000)), "val1": list(range(10000))})
        df2 = ppd.DataFrame(
            {"key": list(range(5000, 15000)), "val2": list(range(10000))}
        )

        start = time.time()
        result = df1.merge(df2, on="key", how="inner")
        elapsed = time.time() - start

        assert len(result) == 5000  # Intersection
        assert elapsed < 2.0


class TestIOPerformance:
    """Performance tests for I/O operations."""

    def test_csv_write_performance(self):
        """Test CSV writing performance."""
        import os
        import tempfile

        df = ppd.DataFrame({"A": list(range(10000)), "B": list(range(10000, 20000))})

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            csv_path = f.name

        try:
            start = time.time()
            df.to_csv(csv_path, index=False)
            elapsed = time.time() - start

            assert elapsed < 2.0
        finally:
            os.unlink(csv_path)

    def test_parquet_write_performance(self):
        """Test Parquet writing performance."""
        import os
        import tempfile

        df = ppd.DataFrame({"A": list(range(10000)), "B": list(range(10000))})

        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, suffix=".parquet"
        ) as f:
            parquet_path = f.name

        try:
            start = time.time()
            df.to_parquet(parquet_path)
            elapsed = time.time() - start

            # Parquet should be fast
            assert elapsed < 1.0
        finally:
            os.unlink(parquet_path)


class TestMemoryEfficiency:
    """Tests for memory efficiency."""

    def test_copy_doesnt_explode_memory(self):
        """Test that copying doesn't unnecessarily duplicate memory."""
        df = ppd.DataFrame({"A": list(range(10000))})

        # Create multiple copies
        copies = [df.copy() for _ in range(10)]

        # All copies should have correct length
        assert all(len(c) == 10000 for c in copies)

    def test_chained_operations_memory(self):
        """Test chained operations don't cause memory issues."""
        df = ppd.DataFrame({"A": list(range(10000))})

        # Chain multiple operations
        result = df.copy()
        result["B"] = result["A"] * 2
        result["C"] = result["B"] + result["A"]
        result = result.drop(columns=["A"])

        assert "B" in result.columns
        assert "C" in result.columns


class TestRegressionTests:
    """Regression tests for previously fixed bugs."""

    def test_multiindex_loc_regression(self):
        """Test MultiIndex loc doesn't regress."""
        df = ppd.DataFrame(
            {"A": ["bar", "bar", "baz"], "B": ["one", "two", "one"], "C": [1, 2, 3]}
        )
        df = df.set_index(["A", "B"])

        # This should work without errors
        result = df.loc[("bar", "one")]
        assert result["C"] == 1

    def test_series_between_regression(self):
        """Test Series.between doesn't regress."""
        s = ppd.Series([1, 2, 3, 4, 5])

        result = s.between(2, 4, inclusive="both")
        assert result.tolist() == [False, True, True, True, False]

    def test_reset_index_column_order_regression(self):
        """Test reset_index column order doesn't regress."""
        df = ppd.DataFrame({"C": [1, 2], "D": [3, 4]})
        df = df.set_index(["C"])

        result = df.reset_index()
        # Index columns should come first
        assert list(result.columns)[0] == "C"

    def test_droplevel_name_preservation_regression(self):
        """Test droplevel preserves remaining level name."""
        df = ppd.DataFrame({"A": ["bar", "baz"], "B": ["one", "two"], "C": [1, 2]})
        df = df.set_index(["A", "B"])

        result = df.droplevel(0)
        # Should preserve 'B' as index name
        assert result._index_name == "B"


class TestEdgeCasePerformance:
    """Performance tests for edge cases."""

    def test_empty_dataframe_operations(self):
        """Test operations on empty DataFrame are fast."""
        df = ppd.DataFrame()

        start = time.time()
        result = df.copy()
        elapsed = time.time() - start

        assert len(result) == 0
        assert elapsed < 0.1

    def test_single_row_operations(self):
        """Test operations on single-row DataFrame."""
        df = ppd.DataFrame({"A": [1]})

        start = time.time()
        result = df.copy()
        result["B"] = [2]
        elapsed = time.time() - start

        assert len(result) == 1
        assert elapsed < 0.1


class TestBulkOperations:
    """Tests for bulk operations performance."""

    def test_bulk_column_addition(self):
        """Test adding many columns."""
        df = ppd.DataFrame({"A": list(range(1000))})

        start = time.time()
        for i in range(50):
            df[f"col_{i}"] = [i] * 1000
        elapsed = time.time() - start

        assert len(df.columns) == 51
        # Should complete in reasonable time
        assert elapsed < 5.0

    def test_multiple_aggregations(self):
        """Test multiple aggregations."""
        df = ppd.DataFrame(
            {"group": [i % 10 for i in range(10000)], "value": list(range(10000))}
        )

        start = time.time()
        mean = df.groupby("group").agg([pl.col("value").mean().alias("mean")])
        sum_result = df.groupby("group").agg([pl.col("value").sum().alias("sum")])
        elapsed = time.time() - start

        assert len(mean) == 10
        assert len(sum_result) == 10
        assert elapsed < 2.0
