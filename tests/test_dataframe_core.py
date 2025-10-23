"""
Test core DataFrame functionality including initialization, delegation, and basic operations.
"""

import polars as pl
import pytest

from polarpandas import DataFrame


class TestDataFrameInitialization:
    """Test DataFrame initialization from various sources."""

    def test_init_from_dict(self):
        """Test creating DataFrame from dictionary."""
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        df = DataFrame(data)
        assert isinstance(df, DataFrame)
        assert hasattr(df, "_df")
        assert isinstance(df._df, pl.DataFrame)
        assert df._df.columns == ["a", "b"]

    def test_init_from_list_of_dicts(self):
        """Test creating DataFrame from list of dictionaries."""
        data = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
        df = DataFrame(data)
        assert isinstance(df, DataFrame)
        assert isinstance(df._df, pl.DataFrame)

    def test_init_from_polars_dataframe(self):
        """Test creating DataFrame from existing Polars DataFrame."""
        pl_df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        df = DataFrame(pl_df)
        assert isinstance(df, DataFrame)
        assert isinstance(df._df, pl.DataFrame)
        assert df._df.columns == ["x", "y"]

    def test_init_empty(self):
        """Test creating empty DataFrame."""
        df = DataFrame()
        assert isinstance(df, DataFrame)
        assert isinstance(df._df, pl.DataFrame)


class TestDataFrameDelegation:
    """Test that DataFrame properly delegates to underlying Polars DataFrame."""

    def test_access_columns(self):
        """Test accessing columns attribute."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        columns = df.columns
        assert columns == ["a", "b"]

    def test_access_shape(self):
        """Test accessing shape attribute."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        shape = df.shape
        assert shape == (3, 2)

    def test_call_polars_method(self):
        """Test calling a Polars method through delegation."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # Test that we can call Polars methods
        height = df.height
        assert height == 3
        width = df.width
        assert width == 2

    def test_access_nonexistent_attribute(self):
        """Test that accessing nonexistent attribute raises AttributeError."""
        df = DataFrame({"a": [1, 2, 3]})
        with pytest.raises(AttributeError):
            _ = df.nonexistent_attribute


class TestDataFrameRepresentation:
    """Test DataFrame string representations."""

    def test_repr(self):
        """Test __repr__ returns a string."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        repr_str = repr(df)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0

    def test_str(self):
        """Test __str__ returns a string."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        str_repr = str(df)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0

    def test_repr_contains_data(self):
        """Test that repr contains information about the data."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        repr_str = repr(df)
        # Should contain column names
        assert "a" in repr_str
        assert "b" in repr_str


class TestDataFrameCopy:
    """Test DataFrame copying."""

    def test_internal_df_attribute(self):
        """Test that _df attribute is accessible."""
        df = DataFrame({"a": [1, 2, 3]})
        assert hasattr(df, "_df")
        assert isinstance(df._df, pl.DataFrame)

    def test_multiple_instances_independent(self):
        """Test that multiple DataFrame instances are independent."""
        df1 = DataFrame({"a": [1, 2, 3]})
        df2 = DataFrame({"b": [4, 5, 6]})
        assert df1._df is not df2._df
        assert df1.columns != df2.columns
