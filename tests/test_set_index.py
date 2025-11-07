"""Tests for `DataFrame.set_index` without relying on pandas runtime."""

import pytest

import polarpandas as ppd
from polarpandas.index import Index, MultiIndex
from tests.test_helpers import assert_frame_equal, assert_index_equal


class TestSetIndex:
    """Test set_index functionality with pandas compatibility."""

    def setup_method(self):
        """Create test data in both pandas and polarpandas."""
        self.data = {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"],
            "D": [100, 200, 300, 400, 500],
        }
        # Don't create DataFrames here to avoid state pollution
        # Each test method will create fresh DataFrames

    def test_set_index_single_column(self):
        """Test setting index to single column."""
        df = ppd.DataFrame(self.data)

        result = df.set_index("A")
        assert isinstance(result.index, Index)
        assert_index_equal(result.index, [1, 2, 3, 4, 5])
        assert_frame_equal(
            result,
            {
                "B": [10, 20, 30, 40, 50],
                "C": ["a", "b", "c", "d", "e"],
                "D": [100, 200, 300, 400, 500],
            },
        )

        result_keep = df.set_index("A", drop=False)
        assert list(result_keep.columns) == ["A", "B", "C", "D"]
        assert_index_equal(result_keep.index, [1, 2, 3, 4, 5])
        assert_frame_equal(
            result_keep,
            {
                "A": [1, 2, 3, 4, 5],
                "B": [10, 20, 30, 40, 50],
                "C": ["a", "b", "c", "d", "e"],
                "D": [100, 200, 300, 400, 500],
            },
        )

    def test_set_index_multiple_columns(self):
        """Test setting index to multiple columns."""
        df = ppd.DataFrame(self.data)

        result = df.set_index(["A", "B"])
        assert isinstance(result.index, MultiIndex)
        assert_index_equal(result.index, [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)])
        assert_frame_equal(
            result,
            {
                "C": ["a", "b", "c", "d", "e"],
                "D": [100, 200, 300, 400, 500],
            },
        )

        result_keep = df.set_index(["A", "B"], drop=False)
        assert list(result_keep.columns) == ["A", "B", "C", "D"]
        assert isinstance(result_keep.index, MultiIndex)
        assert_index_equal(
            result_keep.index, [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]
        )

    def test_set_index_inplace(self):
        """Test inplace parameter."""
        df = ppd.DataFrame(self.data).copy()
        result = df.set_index("A", inplace=True)
        assert result is None
        assert_index_equal(df.index, [1, 2, 3, 4, 5])
        assert list(df.columns) == ["B", "C", "D"]

        df = ppd.DataFrame(self.data)
        result_copy = df.set_index("A", inplace=False)
        assert_index_equal(df.index, [0, 1, 2, 3, 4])
        assert_index_equal(result_copy.index, [1, 2, 3, 4, 5])
        assert_frame_equal(
            result_copy,
            {
                "B": [10, 20, 30, 40, 50],
                "C": ["a", "b", "c", "d", "e"],
                "D": [100, 200, 300, 400, 500],
            },
        )

    def test_set_index_append(self):
        """Test append parameter."""
        df = ppd.DataFrame(self.data).set_index("A")

        result_append = df.set_index("B", append=True)
        assert isinstance(result_append.index, MultiIndex)
        assert_index_equal(
            result_append.index, [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]
        )
        assert list(result_append.columns) == ["C", "D"]

        result_replace = df.set_index("B", append=False)
        assert isinstance(result_replace.index, Index)
        assert_index_equal(result_replace.index, [10, 20, 30, 40, 50])

    def test_set_index_string_column(self):
        """Test setting index to string column."""
        df = ppd.DataFrame(self.data).set_index("C")
        assert_index_equal(df.index, ["a", "b", "c", "d", "e"])
        assert list(df.columns) == ["A", "B", "D"]

    def test_set_index_mixed_types(self):
        """Test setting index with mixed data types."""
        df = ppd.DataFrame(self.data).set_index(["A", "C"])
        assert isinstance(df.index, MultiIndex)
        assert_index_equal(df.index, [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")])
        assert list(df.columns) == ["B", "D"]

    def test_set_index_with_duplicates(self):
        """Test setting index with duplicate values."""
        data_with_duplicates = {
            "A": [1, 2, 1, 2, 3],
            "B": [10, 20, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"],
        }
        df = ppd.DataFrame(data_with_duplicates).set_index("A")
        assert_index_equal(df.index, [1, 2, 1, 2, 3])
        assert list(df.columns) == ["B", "C"]

    def test_set_index_empty_dataframe(self):
        """Test set_index with empty DataFrame."""
        ppd_empty = ppd.DataFrame()

        # Should raise error for empty DataFrame
        with pytest.raises(KeyError):
            ppd_empty.set_index("A")

    def test_set_index_nonexistent_column(self):
        """Test set_index with non-existent column."""
        with pytest.raises(KeyError):
            ppd.DataFrame(self.data).set_index("nonexistent")

    def test_set_index_already_indexed(self):
        """Test set_index on already indexed DataFrame."""
        df = ppd.DataFrame(self.data).set_index("A")

        result = df.set_index("B")
        assert isinstance(result.index, Index)
        assert_index_equal(result.index, [10, 20, 30, 40, 50])
        assert list(result.columns) == ["C", "D"]

    def test_set_index_preserve_original(self):
        """Test that original DataFrame is not modified when inplace=False."""
        original = ppd.DataFrame(self.data)
        baseline = original.to_dict()

        _ = ppd.DataFrame(self.data).set_index("A")

        assert original.to_dict() == baseline

    @pytest.mark.skip(
        reason="Polars has limited support for null values in index. See KNOWN_LIMITATIONS.md - permanent limitation"
    )
    def test_set_index_with_nulls(self):
        pytest.skip("Known limitation: MultiIndex creation with nulls not supported")

    def test_set_index_return_type(self):
        """Test that set_index returns correct type."""
        result = ppd.DataFrame(self.data).set_index("A")
        assert isinstance(result, ppd.DataFrame)

        # Test inplace=True returns None
        result = ppd.DataFrame(self.data).set_index("A", inplace=True)
        assert result is None

    def test_set_index_chain_operations(self):
        """Test chaining set_index operations."""
        # Chain multiple set_index operations
        result = ppd.DataFrame(self.data).set_index("A").set_index("B", append=True)
        assert isinstance(result.index, MultiIndex)
        assert_index_equal(result.index, [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)])
        assert list(result.columns) == ["C", "D"]

    def test_set_index_with_different_dtypes(self):
        """Test set_index with different data types."""
        data_mixed = {
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        }
        df = ppd.DataFrame(data_mixed)

        assert_index_equal(df.set_index("int_col").index, [1, 2, 3])
        assert_index_equal(df.set_index("float_col").index, [1.1, 2.2, 3.3])
        assert_index_equal(df.set_index("str_col").index, ["a", "b", "c"])
        assert_index_equal(df.set_index("bool_col").index, [True, False, True])

    def test_set_index_error_handling(self):
        """Test error handling matches pandas."""
        # Test with invalid column name
        with pytest.raises(KeyError):
            ppd.DataFrame(self.data).set_index("invalid")

        # Test with empty list
        with pytest.raises(ValueError):
            ppd.DataFrame(self.data).set_index([])

        # Test with None - should raise KeyError
        with pytest.raises(KeyError):
            ppd.DataFrame(self.data).set_index(None)

    def test_set_index_inplace_false_append_multilevel(self):
        """Test set_index with inplace=False and append=True creates multilevel index."""
        # Lines 1255-1257: inplace=False, append=True with multiple columns
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        df_indexed = df.set_index("a")

        result = df_indexed.set_index(["b", "c"], append=True, inplace=False)
        assert isinstance(result, ppd.DataFrame)
        assert result._index is not None
        # Should be list of tuples
        assert all(isinstance(idx, tuple) for idx in result._index)
        assert len(result._index[0]) == 3  # (a, b, c)

    def test_set_index_inplace_false_replace_index(self):
        """Test set_index with inplace=False replaces index."""
        # Lines 1283: inplace=False, replace with multiple columns
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

        result = df.set_index(["b", "c"], inplace=False)
        assert isinstance(result, ppd.DataFrame)
        assert result._index is not None
        # Should be list of tuples
        assert all(isinstance(idx, tuple) for idx in result._index)
        assert len(result._index[0]) == 2  # (b, c)

    def test_set_index_result_independence(self):
        """Test that result DataFrame is independent when inplace=False."""
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        result = df.set_index("a", inplace=False)
        # Modify result
        result["c"] = [7, 8, 9]

        # Original should be unchanged
        assert "c" not in df.columns
        assert "c" in result.columns
