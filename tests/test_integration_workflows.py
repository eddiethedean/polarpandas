"""
Integration tests for complex workflows.

This test file verifies complex, real-world workflows combining
multiple operations to ensure they work together correctly.
"""

import polars as pl

import polarpandas as ppd


class TestDataCleaningWorkflow:
    """Tests for end-to-end data cleaning workflows."""

    def test_full_data_cleaning_pipeline(self):
        """Test complete data cleaning workflow."""
        # Create messy data
        df = ppd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "Bob", None, "David", "Eve"],
                "score": [85.5, None, 92.0, 78.5, 95.0],
                "category": ["A", "B", "A", "C", "B"],
            }
        )

        # Clean data: drop nulls, reset index
        clean_df = df.dropna()
        clean_df = clean_df.reset_index(drop=True)

        assert len(clean_df) == 3
        assert "name" in clean_df.columns

    def test_data_transformation_pipeline(self):
        """Test data transformation workflow."""
        df = ppd.DataFrame(
            {"category": ["A", "A", "B", "B", "C"], "value": [10, 20, 30, 40, 50]}
        )

        # Group, aggregate, rename
        result = df.groupby("category").agg([pl.col("value").sum().alias("total")])

        assert "total" in result.columns
        assert len(result) == 3


class TestGroupByAggregationWorkflow:
    """Tests for groupby and aggregation workflows."""

    def test_multi_column_groupby_aggregation(self):
        """Test groupby with multiple columns and aggregations."""
        df = ppd.DataFrame(
            {
                "region": ["North", "North", "South", "South"],
                "product": ["A", "B", "A", "B"],
                "sales": [100, 150, 200, 250],
                "quantity": [10, 15, 20, 25],
            }
        )

        # Group by multiple columns
        result = df.groupby(["region", "product"]).agg(
            [
                pl.col("sales").sum().alias("total_sales"),
                pl.col("quantity").sum().alias("total_quantity"),
            ]
        )

        assert "total_sales" in result.columns
        assert "total_quantity" in result.columns

    def test_groupby_transform_workflow(self):
        """Test groupby with transformation."""
        df = ppd.DataFrame({"group": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})

        # Group and compute group statistics
        grouped = df.groupby("group").agg([pl.col("value").mean().alias("group_mean")])

        assert len(grouped) == 2


class TestMergeJoinWorkflow:
    """Tests for merge and join workflows."""

    def test_multiple_merge_workflow(self):
        """Test workflow with multiple merges."""
        df1 = ppd.DataFrame({"id": [1, 2, 3], "val1": [10, 20, 30]})
        df2 = ppd.DataFrame({"id": [1, 2, 3], "val2": [40, 50, 60]})
        df3 = ppd.DataFrame({"id": [1, 2, 3], "val3": [70, 80, 90]})

        # Merge multiple DataFrames
        result = df1.merge(df2, on="id")
        result = result.merge(df3, on="id")

        assert "val1" in result.columns
        assert "val2" in result.columns
        assert "val3" in result.columns

    def test_merge_then_filter_workflow(self):
        """Test merge followed by filtering."""
        customers = ppd.DataFrame(
            {"customer_id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}
        )
        orders = ppd.DataFrame(
            {"customer_id": [1, 1, 2, 3], "amount": [100, 150, 200, 250]}
        )

        # Merge and filter
        result = customers.merge(orders, on="customer_id")
        high_value = result[result["amount"] > 150]

        assert len(high_value) >= 1


class TestPivotReshapeWorkflow:
    """Tests for pivot and reshape workflows."""

    def test_melt_then_aggregate_workflow(self):
        """Test melt followed by aggregation."""
        df = ppd.DataFrame(
            {"id": [1, 2], "score_math": [85, 90], "score_english": [78, 88]}
        )

        # Melt to long format
        melted = df.melt(id_vars="id", value_vars=["score_math", "score_english"])

        assert "variable" in melted.columns
        assert "value" in melted.columns
        assert len(melted) == 4


class TestIndexingWorkflow:
    """Tests for complex indexing workflows."""

    def test_multiindex_groupby_workflow(self):
        """Test MultiIndex with groupby workflow."""
        df = ppd.DataFrame(
            {
                "A": ["bar", "bar", "baz", "baz"],
                "B": ["one", "two", "one", "two"],
                "C": [1, 2, 3, 4],
                "D": [5, 6, 7, 8],
            }
        )

        # Set MultiIndex
        df = df.set_index(["A", "B"])

        # Group by level and aggregate
        result = df.sum(level=0)

        assert isinstance(result, ppd.DataFrame)
        assert len(result) == 2  # 'bar' and 'baz'

    def test_multiindex_slice_aggregate_workflow(self):
        """Test MultiIndex slicing followed by aggregation."""
        df = ppd.DataFrame(
            {
                "A": ["foo", "foo", "bar", "bar"],
                "B": ["one", "two", "one", "two"],
                "values": [1, 2, 3, 4],
            }
        )

        df = df.set_index(["A", "B"])

        # Slice and aggregate
        subset = df.loc["foo"]
        total = subset.sum()

        assert total["values"] == 3


class TestConcatMergeWorkflow:
    """Tests for combined concat and merge workflows."""

    def test_concat_then_merge_workflow(self):
        """Test concatenating then merging."""
        df1 = ppd.DataFrame({"id": [1, 2], "A": [10, 20]})
        df2 = ppd.DataFrame({"id": [3, 4], "A": [30, 40]})

        # Concat vertically
        combined = ppd.concat([df1, df2])

        # Merge with another DataFrame
        lookup = ppd.DataFrame({"id": [1, 2, 3, 4], "B": ["a", "b", "c", "d"]})
        result = combined.merge(lookup, on="id")

        assert "A" in result.columns
        assert "B" in result.columns
        assert len(result) == 4


class TestFilterSortWorkflow:
    """Tests for filtering and sorting workflows."""

    def test_filter_sort_select_workflow(self):
        """Test filtering, sorting, and column selection."""
        df = ppd.DataFrame(
            {
                "A": [5, 2, 8, 1, 9],
                "B": [10, 20, 30, 40, 50],
                "C": ["x", "y", "z", "w", "v"],
            }
        )

        # Filter values
        filtered = df[df["A"] > 2]

        # Sort
        sorted_df = filtered.sort_values("A")

        # Select columns
        result = sorted_df[["A", "C"]]

        assert list(result.columns) == ["A", "C"]
        assert len(result) == 3


class TestStatisticalAnalysisWorkflow:
    """Tests for statistical analysis workflows."""

    def test_descriptive_statistics_workflow(self):
        """Test workflow computing multiple statistics."""
        df = ppd.DataFrame({"scores": [85, 90, 78, 92, 88, 95, 82]})

        mean = df.mean()
        std = df.std()
        min_val = df.min()
        max_val = df.max()

        assert mean["scores"] > 0
        assert std["scores"] > 0
        assert min_val["scores"] == 78
        assert max_val["scores"] == 95

    def test_correlation_analysis_workflow(self):
        """Test correlation analysis workflow."""
        df = ppd.DataFrame(
            {"X": [1, 2, 3, 4, 5], "Y": [2, 4, 6, 8, 10], "Z": [5, 4, 3, 2, 1]}
        )

        # Compute correlations
        corr_matrix = df.corr()

        assert isinstance(corr_matrix, ppd.DataFrame)
        # X and Y should be perfectly correlated
        assert abs(corr_matrix.loc["X", "Y"] - 1.0) < 0.01


class TestDataNormalizationWorkflow:
    """Tests for data normalization workflows."""

    def test_normalize_by_group_workflow(self):
        """Test normalizing values by group."""
        df = ppd.DataFrame(
            {"group": ["A", "A", "A", "B", "B", "B"], "value": [10, 20, 30, 40, 50, 60]}
        )

        # Compute group means
        group_stats = df.groupby("group").agg(
            [pl.col("value").mean().alias("group_mean")]
        )

        assert len(group_stats) == 2


class TestMultiIndexComplexWorkflow:
    """Tests for complex MultiIndex workflows."""

    def test_multiindex_creation_manipulation_workflow(self):
        """Test creating and manipulating MultiIndex."""
        # Create DataFrame
        df = ppd.DataFrame(
            {
                "level1": ["A", "A", "B", "B"],
                "level2": ["x", "y", "x", "y"],
                "values": [1, 2, 3, 4],
            }
        )

        # Set MultiIndex
        df = df.set_index(["level1", "level2"])

        # Drop a level
        df_dropped = df.droplevel(1)

        # Verify structure
        assert not isinstance(df_dropped.index, ppd.MultiIndex)

    def test_multiindex_swap_reorder_workflow(self):
        """Test swapping and reordering MultiIndex levels."""
        df = ppd.DataFrame(
            {
                "A": ["foo", "foo", "bar", "bar"],
                "B": ["one", "two", "one", "two"],
                "C": [1, 2, 3, 4],
            }
        )

        df = df.set_index(["A", "B"])

        # Swap levels
        swapped = df.swaplevel()

        assert isinstance(swapped.index, ppd.MultiIndex)


class TestAggregationFilteringWorkflow:
    """Tests for aggregation and filtering workflows."""

    def test_aggregate_then_filter_workflow(self):
        """Test aggregating then filtering results."""
        df = ppd.DataFrame(
            {
                "category": ["A", "A", "B", "B", "C", "C"],
                "value": [10, 20, 15, 25, 5, 10],
            }
        )

        # Aggregate by category
        agg_result = df.groupby("category").agg([pl.col("value").sum().alias("total")])

        # Filter aggregated results
        high_totals = agg_result[agg_result["total"] > 20]

        assert len(high_totals) >= 1


class TestTimeSeriesLikeWorkflow:
    """Tests for time-series-like workflows."""

    def test_sequential_data_workflow(self):
        """Test workflow with sequential data operations."""
        df = ppd.DataFrame(
            {"date_id": [1, 2, 3, 4, 5], "value": [100, 110, 105, 115, 120]}
        )

        # Compute differences
        df["diff"] = df["value"].diff()

        # Compute cumulative sum
        df["cumsum"] = df["value"].cumsum()

        assert "diff" in df.columns
        assert "cumsum" in df.columns
        assert df["cumsum"].tolist()[-1] == 550


class TestJoinAndAggregateWorkflow:
    """Tests for join and aggregate workflows."""

    def test_join_multiple_tables_workflow(self):
        """Test joining multiple tables and aggregating."""
        # Create related tables
        products = ppd.DataFrame(
            {
                "product_id": [1, 2, 3],
                "product_name": ["Widget", "Gadget", "Doohickey"],
                "category": ["A", "B", "A"],
            }
        )

        sales = ppd.DataFrame(
            {"product_id": [1, 1, 2, 3], "amount": [100, 150, 200, 250]}
        )

        # Join and aggregate
        joined = sales.merge(products, on="product_id")
        category_sales = joined.groupby("category").agg(
            [pl.col("amount").sum().alias("total_sales")]
        )

        assert len(category_sales) == 2  # Categories A and B


class TestReshapeAnalyzeWorkflow:
    """Tests for reshape and analyze workflows."""

    def test_wide_to_long_analysis_workflow(self):
        """Test converting wide to long format for analysis."""
        df = ppd.DataFrame(
            {
                "student": ["Alice", "Bob"],
                "math": [85, 90],
                "english": [78, 88],
                "science": [92, 85],
            }
        )

        # Melt to long format
        long_df = df.melt(
            id_vars="student",
            value_vars=["math", "english", "science"],
            var_name="subject",
            value_name="score",
        )

        # Aggregate by subject
        subject_avg = long_df.groupby("subject").agg(
            [pl.col("score").mean().alias("avg_score")]
        )

        assert len(subject_avg) == 3


class TestConcatFilterWorkflow:
    """Tests for concat and filter workflows."""

    def test_concat_multiple_filter_workflow(self):
        """Test concatenating multiple DataFrames and filtering."""
        df1 = ppd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
        df2 = ppd.DataFrame({"A": [3, 4], "B": ["z", "w"]})
        df3 = ppd.DataFrame({"A": [5, 6], "B": ["v", "u"]})

        # Concat all
        combined = ppd.concat([df1, df2, df3])

        # Filter
        filtered = combined[combined["A"] > 2]

        assert len(filtered) == 4


class TestComplexColumnOperations:
    """Tests for complex column manipulation workflows."""

    def test_add_compute_drop_workflow(self):
        """Test adding, computing, and dropping columns."""
        df = ppd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # Add computed column
        df["C"] = df["A"] + df["B"]

        # Add another computed column
        df["D"] = df["C"] * 2

        # Drop original columns
        result = df.drop(columns=["A", "B"])

        assert "C" in result.columns
        assert "D" in result.columns
        assert "A" not in result.columns


class TestMultiStepTransformation:
    """Tests for multi-step transformations."""

    def test_filter_group_aggregate_sort_workflow(self):
        """Test complex multi-step workflow."""
        df = ppd.DataFrame(
            {
                "category": ["A", "A", "B", "B", "C", "C", "D"],
                "subcategory": ["x", "y", "x", "y", "x", "y", "x"],
                "value": [10, 20, 15, 25, 30, 35, 5],
            }
        )

        # Step 1: Filter
        filtered = df[df["value"] > 10]

        # Step 2: Group and aggregate
        grouped = filtered.groupby("category").agg(
            [pl.col("value").sum().alias("total")]
        )

        # Step 3: Sort
        sorted_result = grouped.sort_values("total")

        assert len(sorted_result) >= 1


class TestDataValidationWorkflow:
    """Tests for data validation workflows."""

    def test_null_check_and_fill_workflow(self):
        """Test checking for nulls and filling them."""
        df = ppd.DataFrame({"A": [1, None, 3, None, 5], "B": [None, 2, 3, 4, 5]})

        # Check for nulls
        _ = df.isna()  # Just verify it works

        # Fill nulls
        filled = df.fillna(0)

        # Verify no nulls remain
        no_nulls = filled.notna()
        assert all(no_nulls["A"].tolist())

    def test_duplicate_detection_workflow(self):
        """Test detecting and removing duplicates."""
        df = ppd.DataFrame({"A": [1, 2, 2, 3, 3, 3], "B": [4, 5, 5, 6, 6, 6]})

        # Check for duplicates
        _ = df.duplicated()  # Just verify it works

        # Remove duplicates
        unique_df = df.drop_duplicates()

        assert len(unique_df) == 3


class TestColumnRenameRearrangeWorkflow:
    """Tests for column renaming and rearranging."""

    def test_rename_select_reorder_workflow(self):
        """Test renaming and reordering columns."""
        df = ppd.DataFrame(
            {"old_name1": [1, 2, 3], "old_name2": [4, 5, 6], "old_name3": [7, 8, 9]}
        )

        # Rename columns
        df = df.rename(columns={"old_name1": "A", "old_name2": "B", "old_name3": "C"})

        # Reorder columns
        df = df[["C", "A", "B"]]

        assert list(df.columns) == ["C", "A", "B"]


class TestChainedOperationsWorkflow:
    """Tests for method chaining workflows."""

    def test_method_chaining_workflow(self):
        """Test chaining multiple operations."""
        df = ppd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})

        # Chain operations
        result = df.query("A > 2") if hasattr(df, "query") else df[df["A"] > 2]

        assert len(result) == 3


class TestCopyModifyWorkflow:
    """Tests for copy and modify workflows."""

    def test_copy_and_modify_workflow(self):
        """Test copying and modifying DataFrames independently."""
        df1 = ppd.DataFrame({"A": [1, 2, 3]})

        # Copy and modify
        df2 = df1.copy()
        df2["B"] = [4, 5, 6]

        # Original should be unchanged
        assert "B" not in df1.columns
        assert "B" in df2.columns


class TestIndexResetWorkflow:
    """Tests for index reset workflows."""

    def test_set_reset_index_workflow(self):
        """Test setting and resetting index."""
        df = ppd.DataFrame({"A": ["x", "y", "z"], "B": [1, 2, 3]})

        # Set index
        df_indexed = df.set_index("A")
        assert df_indexed._index is not None

        # Reset index
        df_reset = df_indexed.reset_index()
        assert "A" in df_reset.columns


class TestCSVRoundTrip:
    """Tests for CSV read/write workflows."""

    def test_csv_write_read_workflow(self):
        """Test writing and reading CSV."""
        import os
        import tempfile

        df = ppd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # Write to CSV
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            csv_path = f.name

        try:
            df.to_csv(csv_path, index=False)

            # Read back
            df_read = ppd.read_csv(csv_path)

            assert "A" in df_read.columns
            assert len(df_read) == 3
        finally:
            os.unlink(csv_path)


class TestParquetRoundTrip:
    """Tests for Parquet read/write workflows."""

    def test_parquet_write_read_workflow(self):
        """Test writing and reading Parquet."""
        import os
        import tempfile

        df = ppd.DataFrame({"A": [1, 2, 3], "B": [4.5, 5.5, 6.5], "C": ["x", "y", "z"]})

        # Write to Parquet
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, suffix=".parquet"
        ) as f:
            parquet_path = f.name

        try:
            df.to_parquet(parquet_path)

            # Read back
            df_read = ppd.read_parquet(parquet_path)

            assert "A" in df_read.columns
            assert len(df_read) == 3
        finally:
            os.unlink(parquet_path)


class TestIoDatetimeWorkflow:
    """Integration workflow covering IO, datetime conversion, and LazyFrame."""

    def test_csv_datetime_lazy_index_roundtrip(self, tmp_path):
        csv_path = tmp_path / "events.csv"
        source = ppd.DataFrame(
            {
                "event_id": [1, 2, 3],
                "event_time": [
                    "2023-01-01 09:00:00",
                    "2023-01-02 10:15:00",
                    "2023-01-03 12:30:00",
                ],
                "event_value": [5, 15, 25],
            }
        )
        source.to_csv(csv_path, index=False)

        eager = ppd.read_csv(str(csv_path))
        eager_time = ppd.to_datetime(eager["event_time"].to_list())
        eager["event_time"] = eager_time["datetime"]
        eager = eager.set_index("event_id")

        assert eager._index == [1, 2, 3]

        lazy = ppd.scan_csv(str(csv_path), dtype={"event_id": "int64"})
        filtered_lazy = lazy.filter(pl.col("event_value") >= 15).select(
            ["event_id", "event_value"]
        )
        filtered = filtered_lazy.collect().set_index("event_id")

        joined = eager.loc[filtered.index.tolist()]
        assert joined["event_value"].to_list() == filtered["event_value"].to_list()
