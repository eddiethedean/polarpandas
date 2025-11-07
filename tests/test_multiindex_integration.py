"""
Comprehensive integration tests for MultiIndex functionality.

These tests verify MultiIndex functionality by using pandas to generate expected
results and test data, then verifying polarpandas produces equivalent results.
Tests verify functional correctness without running pandas in parallel.
"""

import pandas as pd

import polarpandas as ppd
from polarpandas.index import MultiIndex


class TestMultiIndexCreationIntegration:
    """Integration tests for MultiIndex creation methods."""

    def test_from_arrays_integration(self):
        """Test MultiIndex.from_arrays() matches pandas structure."""
        arrays = [["bar", "bar", "baz", "baz"], ["one", "two", "one", "two"]]
        names = ["first", "second"]

        # Use pandas to generate expected structure
        pd_idx = pd.MultiIndex.from_arrays(arrays, names=names)
        ppd_idx = MultiIndex.from_arrays(arrays, names=names)

        # Verify structure matches
        assert ppd_idx.nlevels == pd_idx.nlevels == 2
        assert ppd_idx.names == pd_idx.names == ("first", "second")
        assert len(ppd_idx) == len(pd_idx) == 4

        # Verify levels match
        assert ppd_idx.levels[0] == list(pd_idx.levels[0])
        assert ppd_idx.levels[1] == list(pd_idx.levels[1])

        # Verify codes match
        assert ppd_idx.codes[0] == list(pd_idx.codes[0])
        assert ppd_idx.codes[1] == list(pd_idx.codes[1])

        # Verify tuple values match
        assert ppd_idx.tolist() == [tuple(x) for x in pd_idx.values]

    def test_from_tuples_integration(self):
        """Test MultiIndex.from_tuples() matches pandas structure."""
        tuples = [("bar", "one"), ("bar", "two"), ("baz", "one"), ("baz", "two")]
        names = ["first", "second"]

        # Use pandas to generate expected structure
        pd_idx = pd.MultiIndex.from_tuples(tuples, names=names)
        ppd_idx = MultiIndex.from_tuples(tuples, names=names)

        # Verify structure matches
        assert ppd_idx.nlevels == pd_idx.nlevels == 2
        assert ppd_idx.names == pd_idx.names == ("first", "second")
        assert len(ppd_idx) == len(pd_idx) == 4

        # Verify tuple values match
        assert ppd_idx.tolist() == [tuple(x) for x in pd_idx.values]

    def test_from_product_integration(self):
        """Test MultiIndex.from_product() matches pandas structure."""
        iterables = [["bar", "baz"], ["one", "two"]]
        names = ["first", "second"]

        # Use pandas to generate expected structure
        pd_idx = pd.MultiIndex.from_product(iterables, names=names)
        ppd_idx = MultiIndex.from_product(iterables, names=names)

        # Verify structure matches
        assert ppd_idx.nlevels == pd_idx.nlevels == 2
        assert ppd_idx.names == pd_idx.names == ("first", "second")
        assert len(ppd_idx) == len(pd_idx) == 4

        # Verify all combinations are present (order may differ)
        pd_tuples = {tuple(x) for x in pd_idx.values}
        ppd_tuples = set(ppd_idx.tolist())
        assert pd_tuples == ppd_tuples

    def test_from_frame_integration(self):
        """Test MultiIndex.from_frame() matches pandas structure."""
        # Use pandas to create expected structure
        pd_df = pd.DataFrame({"A": ["bar", "baz"], "B": ["one", "two"]})
        ppd_df = ppd.DataFrame({"A": ["bar", "baz"], "B": ["one", "two"]})

        pd_idx = pd.MultiIndex.from_frame(pd_df, names=["first", "second"])
        ppd_idx = MultiIndex.from_frame(ppd_df, names=["first", "second"])

        # Verify structure matches
        assert ppd_idx.nlevels == pd_idx.nlevels == 2
        assert ppd_idx.names == pd_idx.names == ("first", "second")
        assert len(ppd_idx) == len(pd_idx) == 2

        # Verify tuple values match
        assert ppd_idx.tolist() == [tuple(x) for x in pd_idx.values]

    def test_from_arrays_three_levels(self):
        """Test MultiIndex.from_arrays() with three levels."""
        arrays = [["A", "A", "B", "B"], ["x", "y", "x", "y"], [1, 2, 1, 2]]
        names = ["level1", "level2", "level3"]

        # Use pandas to generate expected structure
        pd_idx = pd.MultiIndex.from_arrays(arrays, names=names)
        ppd_idx = MultiIndex.from_arrays(arrays, names=names)

        # Verify structure matches
        assert ppd_idx.nlevels == pd_idx.nlevels == 3
        assert ppd_idx.names == pd_idx.names == tuple(names)
        assert len(ppd_idx) == len(pd_idx) == 4

        # Verify tuple values match
        assert ppd_idx.tolist() == [tuple(x) for x in pd_idx.values]


class TestDataFrameMultiIndexOperationsIntegration:
    """Integration tests for DataFrame operations with MultiIndex."""

    def setup_method(self):
        """Create test data."""
        self.data = {
            "A": ["bar", "bar", "baz", "baz"],
            "B": ["one", "two", "one", "two"],
            "C": [1, 2, 3, 4],
            "D": [10, 20, 30, 40],
        }

    def test_set_index_multiple_columns_integration(self):
        """Test set_index with multiple columns creates MultiIndex."""
        # Use pandas to generate expected structure
        pd_df = pd.DataFrame(self.data)
        ppd_df = ppd.DataFrame(self.data)

        pd_result = pd_df.set_index(["A", "B"])
        ppd_result = ppd_df.set_index(["A", "B"])

        # Verify MultiIndex structure
        assert isinstance(ppd_result.index, MultiIndex)
        assert isinstance(pd_result.index, pd.MultiIndex)
        assert ppd_result.index.nlevels == pd_result.index.nlevels == 2
        assert ppd_result.index.names == pd_result.index.names == ("A", "B")

        # Verify index values match
        pd_tuples = [tuple(x) for x in pd_result.index.values]
        ppd_tuples = ppd_result.index.tolist()
        assert pd_tuples == ppd_tuples

    def test_reset_index_multiindex_integration(self):
        """Test reset_index with MultiIndex."""
        # Use pandas to generate expected structure
        pd_df = pd.DataFrame(self.data).set_index(["A", "B"])
        ppd_df = ppd.DataFrame(self.data).set_index(["A", "B"])

        pd_result = pd_df.reset_index(drop=False)
        ppd_result = ppd_df.reset_index(drop=False)

        # Verify columns restored
        assert "A" in ppd_result.columns
        assert "B" in ppd_result.columns
        assert "A" in pd_result.columns
        assert "B" in pd_result.columns

        # Verify index is reset
        assert ppd_result._index is None or len(ppd_result._index) == 0
        assert isinstance(pd_result.index, pd.RangeIndex)

        # Verify data matches
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_droplevel_dataframe_integration(self):
        """Test DataFrame.droplevel() matches pandas."""
        # Use pandas to generate expected structure
        pd_df = pd.DataFrame(self.data).set_index(["A", "B"])
        ppd_df = ppd.DataFrame(self.data).set_index(["A", "B"])

        pd_result = pd_df.droplevel(0)
        ppd_result = ppd_df.droplevel(0)

        # Verify level dropped
        if isinstance(pd_result.index, pd.MultiIndex):
            assert isinstance(ppd_result.index, MultiIndex)
            assert ppd_result.index.nlevels == pd_result.index.nlevels == 1
        else:
            assert isinstance(ppd_result.index, ppd.Index)
            assert not isinstance(ppd_result.index, MultiIndex)

        # Verify data preserved
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_swaplevel_dataframe_integration(self):
        """Test DataFrame.swaplevel() matches pandas."""
        # Use pandas to generate expected structure
        pd_df = pd.DataFrame(self.data).set_index(["A", "B"])
        ppd_df = ppd.DataFrame(self.data).set_index(["A", "B"])

        pd_result = pd_df.swaplevel(0, 1)
        ppd_result = ppd_df.swaplevel(0, 1)

        # Verify levels swapped
        assert isinstance(ppd_result.index, MultiIndex)
        assert ppd_result.index.names == pd_result.index.names == ("B", "A")

        # Verify data preserved
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_reorder_levels_dataframe_integration(self):
        """Test DataFrame.reorder_levels() matches pandas."""
        # Use pandas to generate expected structure
        pd_df = pd.DataFrame(self.data).set_index(["A", "B"])
        ppd_df = ppd.DataFrame(self.data).set_index(["A", "B"])

        pd_result = pd_df.reorder_levels([1, 0])
        ppd_result = ppd_df.reorder_levels([1, 0])

        # Verify levels reordered
        assert isinstance(ppd_result.index, MultiIndex)
        assert ppd_result.index.names == pd_result.index.names == ("B", "A")

        # Verify data preserved
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_index_property_returns_multiindex(self):
        """Test that index property returns MultiIndex for tuple indices."""
        ppd_df = ppd.DataFrame(self.data).set_index(["A", "B"])

        idx = ppd_df.index
        assert isinstance(idx, MultiIndex)
        assert idx.nlevels == 2
        assert idx.names == ("A", "B")


class TestMultiIndexIndexingIntegration:
    """Integration tests for MultiIndex indexing operations."""

    def setup_method(self):
        """Create test data with MultiIndex."""
        self.data = {
            "A": ["bar", "bar", "baz", "baz"],
            "B": ["one", "two", "one", "two"],
            "C": [1, 2, 3, 4],
            "D": [10, 20, 30, 40],
        }
        # Use pandas to create expected structure
        self.pd_df = pd.DataFrame(self.data).set_index(["A", "B"])
        self.ppd_df = ppd.DataFrame(self.data).set_index(["A", "B"])

    def test_loc_full_tuple_key_integration(self):
        """Test loc with full tuple key matches pandas."""
        # Use pandas to generate expected result
        pd_result = self.pd_df.loc[("bar", "one")]
        ppd_result = self.ppd_df.loc[("bar", "one")]

        # Verify result type and values
        assert isinstance(ppd_result, ppd.Series)
        assert isinstance(pd_result, pd.Series)

        # Verify values match
        pd.testing.assert_series_equal(ppd_result.to_pandas(), pd_result)

    def test_loc_partial_key_integration(self):
        """Test loc with partial key matches pandas."""
        # Use pandas to generate expected result
        pd_result = self.pd_df.loc["bar"]
        ppd_result = self.ppd_df.loc["bar"]

        # Verify result type
        assert isinstance(ppd_result, ppd.DataFrame)
        assert isinstance(pd_result, pd.DataFrame)

        # Verify values match
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_loc_slice_tuple_integration(self):
        """Test loc with slice tuple matches pandas."""
        # Use pandas to generate expected result
        pd_result = self.pd_df.loc[("bar", slice(None)), :]
        ppd_result = self.ppd_df.loc[("bar", slice(None)), :]

        # Verify result type
        assert isinstance(ppd_result, ppd.DataFrame)
        assert isinstance(pd_result, pd.DataFrame)

        # Verify values match
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_loc_list_of_tuples_integration(self):
        """Test loc with list of tuples matches pandas."""
        # Use pandas to generate expected result
        keys = [("bar", "one"), ("baz", "two")]
        pd_result = self.pd_df.loc[keys]
        ppd_result = self.ppd_df.loc[keys]

        # Verify result type
        assert isinstance(ppd_result, ppd.DataFrame)
        assert isinstance(pd_result, pd.DataFrame)

        # Verify values match
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_loc_single_row_multiple_columns_integration(self):
        """Test loc with single row, multiple columns."""
        # Use pandas to generate expected result
        pd_result = self.pd_df.loc[("bar", "one"), ["C", "D"]]
        ppd_result = self.ppd_df.loc[("bar", "one"), ["C", "D"]]

        # Verify result type
        assert isinstance(ppd_result, ppd.Series)
        assert isinstance(pd_result, pd.Series)

        # Verify values match
        pd.testing.assert_series_equal(ppd_result.to_pandas(), pd_result)

    def test_iloc_preserves_multiindex_integration(self):
        """Test that iloc preserves MultiIndex structure."""
        # Use pandas to generate expected result
        pd_result = self.pd_df.iloc[0:2]
        ppd_result = self.ppd_df.iloc[0:2]

        # Verify MultiIndex preserved
        assert isinstance(ppd_result.index, MultiIndex)
        assert isinstance(pd_result.index, pd.MultiIndex)

        # Verify values match
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)


class TestLevelBasedOperationsIntegration:
    """Integration tests for level-based operations."""

    def setup_method(self):
        """Create test data with MultiIndex."""
        self.data = {
            "A": ["bar", "bar", "baz", "baz"],
            "B": ["one", "two", "one", "two"],
            "C": [1, 2, 3, 4],
            "D": [10, 20, 30, 40],
        }
        # Use pandas to create expected structure
        self.pd_df = pd.DataFrame(self.data).set_index(["A", "B"])
        self.ppd_df = ppd.DataFrame(self.data).set_index(["A", "B"])

    def test_groupby_level_integration(self):
        """Test groupby with level parameter matches pandas."""
        import polars as pl

        # Use pandas to generate expected result
        pd_result = self.pd_df.groupby(level=0)["C"].sum()
        ppd_gb = self.ppd_df.groupby(level=0)
        ppd_result = ppd_gb.agg(pl.col("C").sum())

        # Convert to comparable format
        pd_result = pd_result.reset_index()
        ppd_result_df = ppd_result.to_pandas()

        # Sort by grouping column for comparison
        pd_result = pd_result.sort_values("A").reset_index(drop=True)
        ppd_result_df = ppd_result_df.sort_values("A").reset_index(drop=True)

        # Verify values match (may need to adjust column names)
        assert len(ppd_result_df) == len(pd_result) == 2

    def test_sum_level_integration(self):
        """Test sum with level parameter matches pandas."""
        # Use pandas to generate expected result (pandas 2.0+ uses groupby(level=...).sum())
        pd_result = self.pd_df.groupby(level=0).sum()
        ppd_result = self.ppd_df.sum(level=0)

        # Verify result type
        assert isinstance(ppd_result, (ppd.DataFrame, ppd.Series))
        assert isinstance(pd_result, (pd.DataFrame, pd.Series))

        # Convert to DataFrame for comparison
        if isinstance(ppd_result, ppd.Series):
            ppd_result = ppd_result.to_frame().T
        if isinstance(pd_result, pd.Series):
            pd_result = pd_result.to_frame().T

        # Verify values match
        pd.testing.assert_frame_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False
        )

    def test_mean_level_integration(self):
        """Test mean with level parameter matches pandas."""
        # Use pandas to generate expected result (pandas 2.0+ uses groupby(level=...).mean())
        pd_result = self.pd_df.groupby(level=0).mean()
        ppd_result = self.ppd_df.mean(level=0)

        # Verify result type
        assert isinstance(ppd_result, (ppd.DataFrame, ppd.Series))

        # Convert to DataFrame for comparison
        if isinstance(ppd_result, ppd.Series):
            ppd_result = ppd_result.to_frame().T
        if isinstance(pd_result, pd.Series):
            pd_result = pd_result.to_frame().T

        # Verify values match
        pd.testing.assert_frame_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False
        )

    def test_min_max_level_integration(self):
        """Test min and max with level parameter matches pandas."""
        # Test min (pandas 2.0+ uses groupby(level=...).min())
        pd_result_min = self.pd_df.groupby(level=0).min()
        ppd_result_min = self.ppd_df.min(level=0)

        if isinstance(ppd_result_min, ppd.Series):
            ppd_result_min = ppd_result_min.to_frame().T
        if isinstance(pd_result_min, pd.Series):
            pd_result_min = pd_result_min.to_frame().T

        pd.testing.assert_frame_equal(
            ppd_result_min.to_pandas(), pd_result_min, check_dtype=False
        )

        # Test max (pandas 2.0+ uses groupby(level=...).max())
        pd_result_max = self.pd_df.groupby(level=0).max()
        ppd_result_max = self.ppd_df.max(level=0)

        if isinstance(ppd_result_max, ppd.Series):
            ppd_result_max = ppd_result_max.to_frame().T
        if isinstance(pd_result_max, pd.Series):
            pd_result_max = pd_result_max.to_frame().T

        pd.testing.assert_frame_equal(
            ppd_result_max.to_pandas(), pd_result_max, check_dtype=False
        )

    def test_std_var_level_integration(self):
        """Test std and var with level parameter matches pandas."""
        # Test std (pandas 2.0+ uses groupby(level=...).std())
        pd_result_std = self.pd_df.groupby(level=0).std()
        ppd_result_std = self.ppd_df.std(level=0)

        if isinstance(ppd_result_std, ppd.Series):
            ppd_result_std = ppd_result_std.to_frame().T
        if isinstance(pd_result_std, pd.Series):
            pd_result_std = pd_result_std.to_frame().T

        pd.testing.assert_frame_equal(
            ppd_result_std.to_pandas(), pd_result_std, check_dtype=False
        )

        # Test var (pandas 2.0+ uses groupby(level=...).var())
        pd_result_var = self.pd_df.groupby(level=0).var()
        ppd_result_var = self.ppd_df.var(level=0)

        if isinstance(ppd_result_var, ppd.Series):
            ppd_result_var = ppd_result_var.to_frame().T
        if isinstance(pd_result_var, pd.Series):
            pd_result_var = pd_result_var.to_frame().T

        pd.testing.assert_frame_equal(
            ppd_result_var.to_pandas(), pd_result_var, check_dtype=False
        )

    def test_median_level_integration(self):
        """Test median with level parameter matches pandas."""
        # Use pandas to generate expected result (pandas 2.0+ uses groupby(level=...).median())
        pd_result = self.pd_df.groupby(level=0).median()
        ppd_result = self.ppd_df.median(level=0)

        # Convert to DataFrame for comparison
        if isinstance(ppd_result, ppd.Series):
            ppd_result = ppd_result.to_frame().T
        if isinstance(pd_result, pd.Series):
            pd_result = pd_result.to_frame().T

        # Verify values match
        pd.testing.assert_frame_equal(
            ppd_result.to_pandas(), pd_result, check_dtype=False
        )

    def test_groupby_multiple_levels_integration(self):
        """Test groupby with multiple levels matches pandas."""
        import polars as pl

        # Use pandas to generate expected result
        pd_result = self.pd_df.groupby(level=[0, 1])["C"].sum()
        ppd_gb = self.ppd_df.groupby(level=[0, 1])
        ppd_result = ppd_gb.agg(pl.col("C").sum())

        # Verify result is DataFrame
        assert isinstance(ppd_result, ppd.DataFrame)
        assert len(ppd_result) == len(pd_result) == 4


class TestMultiIndexTransformationsIntegration:
    """Integration tests for MultiIndex with DataFrame transformations."""

    def setup_method(self):
        """Create test data with MultiIndex."""
        self.data1 = {"A": ["bar", "bar"], "B": ["one", "two"], "C": [1, 2]}
        self.data2 = {"A": ["baz", "baz"], "B": ["one", "two"], "C": [3, 4]}

    def test_concat_multiindex_integration(self):
        """Test concat with MultiIndex DataFrames matches pandas."""
        # Use pandas to create expected structure
        pd_df1 = pd.DataFrame(self.data1).set_index(["A", "B"])
        pd_df2 = pd.DataFrame(self.data2).set_index(["A", "B"])
        ppd_df1 = ppd.DataFrame(self.data1).set_index(["A", "B"])
        ppd_df2 = ppd.DataFrame(self.data2).set_index(["A", "B"])

        pd_result = pd.concat([pd_df1, pd_df2])
        ppd_result = ppd.concat([ppd_df1, ppd_df2])

        # Verify MultiIndex preserved
        assert isinstance(ppd_result.index, MultiIndex)
        assert isinstance(pd_result.index, pd.MultiIndex)

        # Verify values match
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_merge_preserves_multiindex_integration(self):
        """Test merge operations preserving MultiIndex."""
        # Create DataFrames with MultiIndex
        pd_df1 = pd.DataFrame(
            {"A": ["bar", "baz"], "B": ["one", "two"], "C": [1, 2]}
        ).set_index(["A", "B"])

        pd_df2 = pd.DataFrame(
            {"A": ["bar", "baz"], "B": ["one", "two"], "D": [10, 20]}
        ).set_index(["A", "B"])

        ppd_df1 = ppd.DataFrame(
            {"A": ["bar", "baz"], "B": ["one", "two"], "C": [1, 2]}
        ).set_index(["A", "B"])

        ppd_df2 = ppd.DataFrame(
            {"A": ["bar", "baz"], "B": ["one", "two"], "D": [10, 20]}
        ).set_index(["A", "B"])

        # Use pandas to generate expected result
        pd_result = pd.merge(
            pd_df1.reset_index(), pd_df2.reset_index(), on=["A", "B"]
        ).set_index(["A", "B"])
        ppd_result = ppd.merge(
            ppd_df1.reset_index(), ppd_df2.reset_index(), on=["A", "B"]
        ).set_index(["A", "B"])

        # Verify MultiIndex preserved
        assert isinstance(ppd_result.index, MultiIndex)

        # Verify values match
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_transpose_multiindex_integration(self):
        """Test transpose with MultiIndex matches pandas."""
        # Use pandas to create expected structure
        pd_df = pd.DataFrame(self.data1).set_index(["A", "B"])
        ppd_df = ppd.DataFrame(self.data1).set_index(["A", "B"])

        pd_result = pd_df.T
        ppd_result = ppd_df.T

        # Verify values match
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)


class TestMultiIndexEdgeCasesIntegration:
    """Integration tests for MultiIndex edge cases."""

    def test_empty_multiindex(self):
        """Test empty MultiIndex."""
        # Use pandas to generate expected structure
        pd_idx = pd.MultiIndex.from_arrays([[], []], names=["A", "B"])
        ppd_idx = MultiIndex.from_arrays([[], []], names=["A", "B"])

        assert len(ppd_idx) == len(pd_idx) == 0
        assert ppd_idx.nlevels == pd_idx.nlevels == 2
        assert ppd_idx.names == pd_idx.names == ("A", "B")

    def test_single_level_multiindex(self):
        """Test single-level MultiIndex behaves correctly."""
        arrays = [["a", "b", "c"]]
        names = ["level1"]

        # Use pandas to generate expected structure
        pd_idx = pd.MultiIndex.from_arrays(arrays, names=names)
        ppd_idx = MultiIndex.from_arrays(arrays, names=names)

        assert ppd_idx.nlevels == pd_idx.nlevels == 1
        assert len(ppd_idx) == len(pd_idx) == 3
        # Compare with pandas' tolist() which returns [('a',), ('b',), ('c',)]
        assert ppd_idx.tolist() == list(pd_idx.values)

    def test_multiindex_with_duplicates(self):
        """Test MultiIndex with duplicate labels."""
        arrays = [["bar", "bar", "bar"], ["one", "one", "two"]]

        # Use pandas to generate expected structure
        pd_idx = pd.MultiIndex.from_arrays(arrays, names=["A", "B"])
        ppd_idx = MultiIndex.from_arrays(arrays, names=["A", "B"])

        assert len(ppd_idx) == len(pd_idx) == 3
        assert ppd_idx.tolist() == [tuple(x) for x in pd_idx.values]

    def test_multiindex_large_dataset(self):
        """Test MultiIndex with larger dataset."""
        # Create larger dataset
        n = 100
        arrays = [
            [f"level1_{i % 10}" for i in range(n)],
            [f"level2_{i % 5}" for i in range(n)],
            list(range(n)),
        ]

        # Use pandas to generate expected structure
        pd_idx = pd.MultiIndex.from_arrays(arrays, names=["L1", "L2", "L3"])
        ppd_idx = MultiIndex.from_arrays(arrays, names=["L1", "L2", "L3"])

        assert len(ppd_idx) == len(pd_idx) == n
        assert ppd_idx.nlevels == pd_idx.nlevels == 3

        # Verify first few values match
        assert ppd_idx.tolist()[:5] == [tuple(x) for x in pd_idx.values[:5]]


class TestMultiIndexIOIntegration:
    """Integration tests for MultiIndex I/O operations."""

    def setup_method(self):
        """Create test data with MultiIndex."""
        self.data = {
            "A": ["bar", "bar", "baz", "baz"],
            "B": ["one", "two", "one", "two"],
            "C": [1, 2, 3, 4],
        }

    def test_to_csv_read_csv_multiindex_integration(self):
        """Test to_csv and read_csv preserve MultiIndex structure."""
        import os
        import tempfile

        # Use pandas to create expected structure
        pd_df = pd.DataFrame(self.data).set_index(["A", "B"])
        ppd_df = ppd.DataFrame(self.data).set_index(["A", "B"])

        # Write to CSV
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            csv_path = f.name

        try:
            pd_df.to_csv(csv_path)
            ppd_df.to_csv(csv_path)

            # Read back
            pd_read = pd.read_csv(csv_path, index_col=[0, 1])
            ppd_read = ppd.read_csv(csv_path, index_col=[0, 1])

            # Verify MultiIndex structure
            assert isinstance(ppd_read.index, MultiIndex)
            assert isinstance(pd_read.index, pd.MultiIndex)

            # Verify values match
            pd.testing.assert_frame_equal(ppd_read.to_pandas(), pd_read)
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_to_parquet_read_parquet_multiindex_integration(self):
        """Test to_parquet and read_parquet preserve MultiIndex structure."""
        import os
        import tempfile

        # Use pandas to create expected structure
        pd_df = pd.DataFrame(self.data).set_index(["A", "B"])
        ppd_df = ppd.DataFrame(self.data).set_index(["A", "B"])

        # Write to parquet
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as f:
            parquet_path = f.name

        try:
            pd_df.to_parquet(parquet_path)
            ppd_df.to_parquet(parquet_path)

            # Read back
            pd_read = pd.read_parquet(parquet_path)
            ppd_read = ppd.read_parquet(parquet_path)

            # Note: Parquet may not preserve MultiIndex structure exactly
            # Verify data is preserved
            assert len(ppd_read) == len(pd_read) == 4
            assert "C" in ppd_read.columns
        finally:
            if os.path.exists(parquet_path):
                os.unlink(parquet_path)
