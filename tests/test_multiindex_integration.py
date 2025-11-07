"""Integration-style MultiIndex tests that do not depend on pandas."""

import polarpandas as ppd
from polarpandas.index import MultiIndex
from tests.test_helpers import assert_frame_equal, assert_index_equal


class TestMultiIndexCreationIntegration:
    """Validate MultiIndex constructors."""

    def test_from_arrays_integration(self) -> None:
        arrays = [["bar", "bar", "baz", "baz"], ["one", "two", "one", "two"]]
        idx = MultiIndex.from_arrays(arrays, names=["first", "second"])

        assert idx.nlevels == 2
        assert idx.names == ("first", "second")
        assert_index_equal(
            idx, [("bar", "one"), ("bar", "two"), ("baz", "one"), ("baz", "two")]
        )

    def test_from_tuples_integration(self) -> None:
        tuples = [("bar", "one"), ("bar", "two"), ("baz", "one"), ("baz", "two")]
        idx = MultiIndex.from_tuples(tuples, names=["first", "second"])

        assert idx.nlevels == 2
        assert idx.names == ("first", "second")
        assert_index_equal(idx, tuples)

    def test_from_product_integration(self) -> None:
        iterables = [["bar", "baz"], ["one", "two"]]
        idx = MultiIndex.from_product(iterables, names=["first", "second"])

        expected = {("bar", "one"), ("bar", "two"), ("baz", "one"), ("baz", "two")}
        assert set(idx.tolist()) == expected
        assert idx.names == ("first", "second")

    def test_from_frame_integration(self) -> None:
        df = ppd.DataFrame({"A": ["bar", "baz"], "B": ["one", "two"]})
        idx = MultiIndex.from_frame(df, names=["first", "second"])

        assert idx.nlevels == 2
        assert idx.names == ("first", "second")
        assert_index_equal(idx, [("bar", "one"), ("baz", "two")])

    def test_from_arrays_three_levels(self) -> None:
        arrays = [["A", "A", "B", "B"], ["x", "y", "x", "y"], [1, 2, 1, 2]]
        idx = MultiIndex.from_arrays(arrays, names=["level1", "level2", "level3"])

        assert idx.nlevels == 3
        assert idx.names == ("level1", "level2", "level3")
        assert_index_equal(
            idx, [("A", "x", 1), ("A", "y", 2), ("B", "x", 1), ("B", "y", 2)]
        )


class TestDataFrameMultiIndexOperationsIntegration:
    """Ensure DataFrame operations respect MultiIndex semantics."""

    def setup_method(self) -> None:
        self.data = {
            "A": ["bar", "bar", "baz", "baz"],
            "B": ["one", "two", "one", "two"],
            "C": [1, 2, 3, 4],
            "D": [10, 20, 30, 40],
        }
        self.df = ppd.DataFrame(self.data)

    def test_set_index_multiple_columns_integration(self) -> None:
        result = self.df.set_index(["A", "B"])
        assert isinstance(result.index, MultiIndex)
        assert_index_equal(
            result.index,
            [("bar", "one"), ("bar", "two"), ("baz", "one"), ("baz", "two")],
        )
        assert list(result.columns) == ["C", "D"]

    def test_reset_index_multiindex_integration(self) -> None:
        df = self.df.set_index(["A", "B"])
        result = df.reset_index(drop=False)
        assert list(result.columns) == ["A", "B", "C", "D"]
        assert result._index is None or len(result._index) == 0

    def test_droplevel_dataframe_integration(self) -> None:
        df = self.df.set_index(["A", "B"])
        result = df.droplevel(0)
        assert_index_equal(result.index, ["one", "two", "one", "two"])
        assert list(result.columns) == ["C", "D"]

    def test_swaplevel_dataframe_integration(self) -> None:
        df = self.df.set_index(["A", "B"])
        result = df.swaplevel(0, 1)
        assert isinstance(result.index, MultiIndex)
        assert result.index.names == ("B", "A")
        assert_index_equal(
            result.index,
            [("one", "bar"), ("two", "bar"), ("one", "baz"), ("two", "baz")],
        )

    def test_reorder_levels_dataframe_integration(self) -> None:
        df = self.df.set_index(["A", "B"])
        result = df.reorder_levels([1, 0])
        assert result.index.names == ("B", "A")
        assert_index_equal(
            result.index,
            [("one", "bar"), ("two", "bar"), ("one", "baz"), ("two", "baz")],
        )

    def test_index_property_returns_multiindex(self) -> None:
        df = self.df.set_index(["A", "B"])
        assert isinstance(df.index, MultiIndex)


class TestMultiIndexIndexingIntegration:
    """Check MultiIndex properties on DataFrames."""

    def setup_method(self) -> None:
        self.df = ppd.DataFrame(
            {
                "A": ["bar", "bar", "baz", "baz"],
                "B": ["one", "two", "one", "two"],
                "C": [1, 2, 3, 4],
            }
        ).set_index(["A", "B"])

    def test_index_metadata(self) -> None:
        idx = self.df.index
        assert isinstance(idx, MultiIndex)
        assert idx.names == ("A", "B")
        assert_index_equal(
            idx, [("bar", "one"), ("bar", "two"), ("baz", "one"), ("baz", "two")]
        )

    def test_index_levels(self) -> None:
        levels = self.df.index.levels
        assert levels[0] == ["bar", "baz"]
        assert levels[1] == ["one", "two"]

    def test_reset_index_round_trip(self) -> None:
        reset = self.df.reset_index()
        assert list(reset.columns) == ["A", "B", "C"]
        assert_frame_equal(
            reset,
            {
                "A": ["bar", "bar", "baz", "baz"],
                "B": ["one", "two", "one", "two"],
                "C": [1, 2, 3, 4],
            },
        )

    def test_reorder_levels_changes_order(self) -> None:
        swapped = self.df.reorder_levels([1, 0])
        assert swapped.index.names == ("B", "A")
        assert_index_equal(
            swapped.index,
            [("one", "bar"), ("two", "bar"), ("one", "baz"), ("two", "baz")],
        )
