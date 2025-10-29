"""
Test DataFrame mutable operations including setitem, delitem, and inplace operations.
"""

import polars as pl
import pytest

from polarpandas import DataFrame


class TestDataFrameSetItem:
    """Test __setitem__ for column assignment."""

    def test_setitem_new_column(self):
        """Test adding a new column with __setitem__."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df["c"] = [7, 8, 9]
        assert "c" in df.columns
        assert df["c"].to_list() == [7, 8, 9]

    def test_setitem_existing_column(self):
        """Test replacing an existing column with __setitem__."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df["a"] = [10, 20, 30]
        assert df["a"].to_list() == [10, 20, 30]

    def test_setitem_mutates_in_place(self):
        """Test that __setitem__ mutates the DataFrame in place."""
        df = DataFrame({"a": [1, 2, 3]})
        original_id = id(df)
        df["b"] = [4, 5, 6]
        assert id(df) == original_id
        assert "b" in df.columns

    def test_setitem_with_series(self):
        """Test __setitem__ with a Polars Series."""
        df = DataFrame({"a": [1, 2, 3]})
        df["b"] = pl.Series("b", [4, 5, 6])
        assert "b" in df.columns
        assert df["b"].to_list() == [4, 5, 6]


class TestDataFrameDelItem:
    """Test __delitem__ for column deletion."""

    def test_delitem_existing_column(self):
        """Test deleting an existing column."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        del df["b"]
        assert "b" not in df.columns
        assert df.columns == ["a", "c"]

    def test_delitem_mutates_in_place(self):
        """Test that __delitem__ mutates the DataFrame in place."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        original_id = id(df)
        del df["b"]
        assert id(df) == original_id
        assert "b" not in df.columns

    def test_delitem_nonexistent_column_raises(self):
        """Test that deleting a nonexistent column raises an error."""
        df = DataFrame({"a": [1, 2, 3]})
        with pytest.raises(
            pl.exceptions.ColumnNotFoundError
        ):  # Polars will raise an error
            del df["nonexistent"]


class TestDataFrameDrop:
    """Test drop() method with inplace parameter."""

    def test_drop_columns_not_inplace(self):
        """Test dropping columns without inplace."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        result = df.drop(["b"], inplace=False)
        assert isinstance(result, DataFrame)
        assert "b" in df.columns  # Original unchanged
        assert "b" not in result.columns  # Result changed

    def test_drop_columns_inplace(self):
        """Test dropping columns with inplace=True."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        original_id = id(df)
        result = df.drop(["b"], inplace=True)
        assert result is None
        assert id(df) == original_id
        assert "b" not in df.columns


class TestDataFrameRename:
    """Test rename() method with inplace parameter."""

    def test_rename_not_inplace(self):
        """Test renaming columns without inplace."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.rename({"a": "x", "b": "y"}, inplace=False)
        assert isinstance(result, DataFrame)
        assert df.columns == ["a", "b"]  # Original unchanged
        assert result.columns == ["x", "y"]  # Result changed

    def test_rename_inplace(self):
        """Test renaming columns with inplace=True."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        original_id = id(df)
        result = df.rename({"a": "x", "b": "y"}, inplace=True)
        assert result is None
        assert id(df) == original_id
        assert df.columns == ["x", "y"]


class TestDataFrameSortValues:
    """Test sort_values() method with inplace parameter."""

    def test_sort_values_not_inplace(self):
        """Test sorting without inplace."""
        df = DataFrame({"a": [3, 1, 2], "b": [6, 4, 5]})
        result = df.sort_values("a", inplace=False)
        assert isinstance(result, DataFrame)
        assert df["a"].to_list() == [3, 1, 2]  # Original unchanged
        assert result["a"].to_list() == [1, 2, 3]  # Result sorted

    def test_sort_values_inplace(self):
        """Test sorting with inplace=True."""
        df = DataFrame({"a": [3, 1, 2], "b": [6, 4, 5]})
        original_id = id(df)
        result = df.sort_values("a", inplace=True)
        assert result is None
        assert id(df) == original_id
        assert df["a"].to_list() == [1, 2, 3]


class TestDataFrameFillna:
    """Test fillna() method with inplace parameter."""

    def test_fillna_not_inplace(self):
        """Test filling NA values without inplace."""
        df = DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
        result = df.fillna(0, inplace=False)
        assert isinstance(result, DataFrame)
        assert df["a"].to_list()[1] is None  # Original unchanged
        assert result["a"].to_list() == [1, 0, 3]  # Result filled

    def test_fillna_inplace(self):
        """Test filling NA values with inplace=True."""
        df = DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
        original_id = id(df)
        result = df.fillna(0, inplace=True)
        assert result is None
        assert id(df) == original_id
        assert df["a"].to_list() == [1, 0, 3]


class TestDataFrameDropna:
    """Test dropna() method with inplace parameter."""

    def test_dropna_not_inplace(self):
        """Test dropping NA rows without inplace."""
        df = DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        result = df.dropna(inplace=False)
        assert isinstance(result, DataFrame)
        assert len(df) == 3  # Original unchanged
        assert len(result) == 2  # Result has NA dropped

    def test_dropna_inplace(self):
        """Test dropping NA rows with inplace=True."""
        df = DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        original_id = id(df)
        result = df.dropna(inplace=True)
        assert result is None
        assert id(df) == original_id
        assert len(df) == 2
