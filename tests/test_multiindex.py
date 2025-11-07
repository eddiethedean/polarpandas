"""
Comprehensive tests for MultiIndex functionality.

This test suite verifies that MultiIndex works correctly and matches pandas behavior.
"""

import polarpandas as ppd
from polarpandas.index import MultiIndex


class TestMultiIndexCreation:
    """Test MultiIndex creation methods."""

    def test_from_arrays(self):
        """Test creating MultiIndex from arrays."""
        arrays = [["bar", "bar", "baz", "baz"], ["one", "two", "one", "two"]]
        idx = MultiIndex.from_arrays(arrays, names=["first", "second"])

        assert idx.nlevels == 2
        assert idx.names == ("first", "second")
        assert len(idx) == 4
        assert idx.tolist() == [
            ("bar", "one"),
            ("bar", "two"),
            ("baz", "one"),
            ("baz", "two"),
        ]

    def test_from_tuples(self):
        """Test creating MultiIndex from tuples."""
        tuples = [("bar", "one"), ("bar", "two"), ("baz", "one"), ("baz", "two")]
        idx = MultiIndex.from_tuples(tuples, names=["first", "second"])

        assert idx.nlevels == 2
        assert idx.names == ("first", "second")
        assert len(idx) == 4

    def test_from_product(self):
        """Test creating MultiIndex from product."""
        iterables = [["bar", "baz"], ["one", "two"]]
        idx = MultiIndex.from_product(iterables, names=["first", "second"])

        assert idx.nlevels == 2
        assert len(idx) == 4
        # Product should create all combinations
        expected = [("bar", "one"), ("bar", "two"), ("baz", "one"), ("baz", "two")]
        assert set(idx.tolist()) == set(expected)

    def test_from_frame(self):
        """Test creating MultiIndex from DataFrame."""
        df = ppd.DataFrame({"A": ["bar", "baz"], "B": ["one", "two"]})
        idx = MultiIndex.from_frame(df, names=["first", "second"])

        assert idx.nlevels == 2
        assert idx.names == ("first", "second")
        assert len(idx) == 2


class TestMultiIndexMethods:
    """Test MultiIndex manipulation methods."""

    def test_get_level_values(self):
        """Test getting level values."""
        arrays = [["bar", "bar", "baz"], ["one", "two", "one"]]
        idx = MultiIndex.from_arrays(arrays, names=["first", "second"])

        level0 = idx.get_level_values(0)
        assert level0.tolist() == ["bar", "bar", "baz"]

        level1 = idx.get_level_values(1)
        assert level1.tolist() == ["one", "two", "one"]

        # Test by name
        level0_by_name = idx.get_level_values("first")
        assert level0_by_name.tolist() == ["bar", "bar", "baz"]

    def test_droplevel(self):
        """Test dropping levels."""
        arrays = [["bar", "bar", "baz"], ["one", "two", "one"]]
        idx = MultiIndex.from_arrays(arrays, names=["first", "second"])

        # Drop first level
        result = idx.droplevel(0)
        assert isinstance(result, ppd.Index)
        assert result.tolist() == ["one", "two", "one"]

        # Drop by name
        result2 = idx.droplevel("first")
        assert isinstance(result2, ppd.Index)

    def test_swaplevel(self):
        """Test swapping levels."""
        arrays = [["bar", "baz"], ["one", "two"]]
        idx = MultiIndex.from_arrays(arrays, names=["first", "second"])

        swapped = idx.swaplevel(0, 1)
        assert swapped.names == ("second", "first")
        # Values should be swapped
        assert swapped.tolist()[0][0] == "one"

    def test_reorder_levels(self):
        """Test reordering levels."""
        arrays = [["bar", "baz"], ["one", "two"]]
        idx = MultiIndex.from_arrays(arrays, names=["first", "second"])

        reordered = idx.reorder_levels([1, 0])
        assert reordered.names == ("second", "first")

    def test_set_names(self):
        """Test setting names."""
        arrays = [["bar", "baz"], ["one", "two"]]
        idx = MultiIndex.from_arrays(arrays, names=["first", "second"])

        new_idx = idx.set_names(["A", "B"])
        assert new_idx.names == ("A", "B")
        assert idx.names == ("first", "second")  # Original unchanged

    def test_get_loc(self):
        """Test getting location."""
        arrays = [["bar", "bar", "baz"], ["one", "two", "one"]]
        idx = MultiIndex.from_arrays(arrays, names=["first", "second"])

        # Full tuple key
        loc = idx.get_loc(("bar", "one"))
        assert loc == 0

        # Partial key (first level only)
        locs = idx.get_loc("bar")
        assert isinstance(locs, list)
        assert 0 in locs and 1 in locs


class TestDataFrameMultiIndex:
    """Test DataFrame with MultiIndex."""

    def test_set_index_multiple_columns(self):
        """Test setting multiple columns as index creates MultiIndex."""
        df = ppd.DataFrame(
            {"A": ["bar", "bar", "baz"], "B": ["one", "two", "one"], "C": [1, 2, 3]}
        )

        df_indexed = df.set_index(["A", "B"])
        assert isinstance(df_indexed.index, MultiIndex)
        assert df_indexed.index.nlevels == 2

    def test_index_property_returns_multiindex(self):
        """Test that index property returns MultiIndex for tuple indices."""
        df = ppd.DataFrame({"A": ["bar", "baz"], "B": ["one", "two"], "C": [1, 2]})
        df = df.set_index(["A", "B"])

        idx = df.index
        assert isinstance(idx, MultiIndex)

    def test_reset_index_multiindex(self):
        """Test reset_index with MultiIndex."""
        df = ppd.DataFrame({"A": ["bar", "baz"], "B": ["one", "two"], "C": [1, 2]})
        df = df.set_index(["A", "B"])

        reset = df.reset_index(drop=False)
        assert "A" in reset.columns
        assert "B" in reset.columns
        assert reset._index is None

    def test_droplevel_dataframe(self):
        """Test DataFrame.droplevel()."""
        df = ppd.DataFrame({"A": ["bar", "baz"], "B": ["one", "two"], "C": [1, 2]})
        df = df.set_index(["A", "B"])

        result = df.droplevel(0)
        assert isinstance(result.index, ppd.Index)
        assert result.index.tolist() == ["one", "two"]

    def test_swaplevel_dataframe(self):
        """Test DataFrame.swaplevel()."""
        df = ppd.DataFrame({"A": ["bar", "baz"], "B": ["one", "two"], "C": [1, 2]})
        df = df.set_index(["A", "B"])

        result = df.swaplevel(0, 1)
        assert isinstance(result.index, MultiIndex)
        assert result.index.names == ("B", "A")

    def test_reorder_levels_dataframe(self):
        """Test DataFrame.reorder_levels()."""
        df = ppd.DataFrame({"A": ["bar", "baz"], "B": ["one", "two"], "C": [1, 2]})
        df = df.set_index(["A", "B"])

        result = df.reorder_levels([1, 0])
        assert isinstance(result.index, MultiIndex)
        assert result.index.names == ("B", "A")

    def test_loc_with_tuple_key(self):
        """Test loc indexing with tuple keys."""
        df = ppd.DataFrame(
            {"A": ["bar", "bar", "baz"], "B": ["one", "two", "one"], "C": [1, 2, 3]}
        )
        df = df.set_index(["A", "B"])

        # Full tuple key
        result = df.loc[("bar", "one")]
        assert isinstance(result, ppd.Series)
        assert result["C"] == 1

        # Partial key
        result2 = df.loc["bar"]
        assert isinstance(result2, ppd.DataFrame)
        assert len(result2) == 2


class TestMultiIndexLevelOperations:
    """Test level-based operations with MultiIndex."""

    def test_groupby_level(self):
        """Test groupby with level parameter."""
        df = ppd.DataFrame(
            {
                "A": ["bar", "bar", "baz", "baz"],
                "B": ["one", "two", "one", "two"],
                "C": [1, 2, 3, 4],
            }
        )
        df = df.set_index(["A", "B"])

        # Group by first level
        import polars as pl

        gb = df.groupby(level=0)
        result = gb.agg(pl.col("C").sum())

        assert isinstance(result, ppd.DataFrame)
        assert len(result) == 2  # Two groups: 'bar' and 'baz'

    def test_groupby_level_by_name(self):
        """Test groupby with level name."""
        df = ppd.DataFrame(
            {
                "A": ["bar", "bar", "baz", "baz"],
                "B": ["one", "two", "one", "two"],
                "C": [1, 2, 3, 4],
            }
        )
        df = df.set_index(["A", "B"])

        # Group by level name
        import polars as pl

        gb = df.groupby(level="A")
        result = gb.agg(pl.col("C").sum())

        assert isinstance(result, ppd.DataFrame)
        assert len(result) == 2

    def test_sum_with_level(self):
        """Test sum() with level parameter."""
        df = ppd.DataFrame(
            {
                "A": ["bar", "bar", "baz", "baz"],
                "B": ["one", "two", "one", "two"],
                "C": [1, 2, 3, 4],
                "D": [10, 20, 30, 40],
            }
        )
        df = df.set_index(["A", "B"])

        # Sum by level
        result = df.sum(level=0)
        assert isinstance(result, (ppd.DataFrame, ppd.Series))

        # Should have 2 rows (one for 'bar', one for 'baz')
        if isinstance(result, ppd.DataFrame):
            assert len(result) == 2

    def test_mean_with_level(self):
        """Test mean() with level parameter."""
        df = ppd.DataFrame(
            {
                "A": ["bar", "bar", "baz", "baz"],
                "B": ["one", "two", "one", "two"],
                "C": [1, 2, 3, 4],
            }
        )
        df = df.set_index(["A", "B"])

        # Mean by level
        result = df.mean(level=0)
        assert isinstance(result, (ppd.DataFrame, ppd.Series))

        if isinstance(result, ppd.DataFrame):
            assert len(result) == 2

    def test_min_with_level(self):
        """Test min() with level parameter."""
        df = ppd.DataFrame(
            {
                "A": ["bar", "bar", "baz", "baz"],
                "B": ["one", "two", "one", "two"],
                "C": [1, 2, 3, 4],
            }
        )
        df = df.set_index(["A", "B"])

        result = df.min(level=0)
        assert isinstance(result, (ppd.DataFrame, ppd.Series))

    def test_max_with_level(self):
        """Test max() with level parameter."""
        df = ppd.DataFrame(
            {
                "A": ["bar", "bar", "baz", "baz"],
                "B": ["one", "two", "one", "two"],
                "C": [1, 2, 3, 4],
            }
        )
        df = df.set_index(["A", "B"])

        result = df.max(level=0)
        assert isinstance(result, (ppd.DataFrame, ppd.Series))

    def test_std_with_level(self):
        """Test std() with level parameter."""
        df = ppd.DataFrame(
            {
                "A": ["bar", "bar", "baz", "baz"],
                "B": ["one", "two", "one", "two"],
                "C": [1.0, 2.0, 3.0, 4.0],
            }
        )
        df = df.set_index(["A", "B"])

        result = df.std(level=0)
        assert isinstance(result, (ppd.DataFrame, ppd.Series))

    def test_median_with_level(self):
        """Test median() with level parameter."""
        df = ppd.DataFrame(
            {
                "A": ["bar", "bar", "baz", "baz"],
                "B": ["one", "two", "one", "two"],
                "C": [1, 2, 3, 4],
            }
        )
        df = df.set_index(["A", "B"])

        result = df.median(level=0)
        assert isinstance(result, (ppd.DataFrame, ppd.Series))

    def test_loc_with_partial_tuple(self):
        """Test loc with partial tuple keys."""
        df = ppd.DataFrame(
            {
                "A": ["bar", "bar", "baz", "baz"],
                "B": ["one", "two", "one", "two"],
                "C": [1, 2, 3, 4],
            }
        )
        df = df.set_index(["A", "B"])

        # Partial tuple with slice
        result = df.loc[("bar", slice(None)), :]
        assert isinstance(result, ppd.DataFrame)
        assert len(result) == 2  # Both 'bar' rows

    def test_multiindex_get_loc_with_slice(self):
        """Test MultiIndex.get_loc with slice in tuple."""
        arrays = [["bar", "bar", "baz"], ["one", "two", "one"]]
        idx = MultiIndex.from_arrays(arrays, names=["first", "second"])

        # Partial tuple with slice
        loc = idx.get_loc(("bar", slice(None)))
        assert isinstance(loc, list)
        assert len(loc) == 2  # Both 'bar' entries
