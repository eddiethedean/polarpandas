"""
Test DataFrame properties and accessors.
"""

from polarpandas import DataFrame


class TestDataFrameProperties:
    """Test DataFrame properties."""

    def test_shape_property(self):
        """Test shape property returns (rows, cols)."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert df.shape == (3, 2)

    def test_columns_property(self):
        """Test columns property returns list of column names."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        assert df.columns == ["a", "b", "c"]

    def test_dtypes_property(self):
        """Test dtypes property returns column dtypes."""
        df = DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        dtypes = df.dtypes
        assert dtypes is not None
        assert len(dtypes) == 2

    def test_empty_property_false(self):
        """Test empty property returns False for non-empty DataFrame."""
        df = DataFrame({"a": [1, 2, 3]})
        assert df.empty is False

    def test_empty_property_true(self):
        """Test empty property returns True for empty DataFrame."""
        df = DataFrame()
        assert df.empty is True

    def test_values_property(self):
        """Test values property returns numpy array."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        values = df.values
        assert values is not None
        # Check it's array-like (Polars might return different type)


class TestDataFrameMethods:
    """Test DataFrame basic methods."""

    def test_head_default(self):
        """Test head() returns first 5 rows by default."""
        df = DataFrame({"a": range(10), "b": range(10, 20)})
        result = df.head()
        assert isinstance(result, DataFrame)
        assert len(result) == 5

    def test_head_with_n(self):
        """Test head(n) returns first n rows."""
        df = DataFrame({"a": range(10), "b": range(10, 20)})
        result = df.head(3)
        assert isinstance(result, DataFrame)
        assert len(result) == 3

    def test_tail_default(self):
        """Test tail() returns last 5 rows by default."""
        df = DataFrame({"a": range(10), "b": range(10, 20)})
        result = df.tail()
        assert isinstance(result, DataFrame)
        assert len(result) == 5

    def test_tail_with_n(self):
        """Test tail(n) returns last n rows."""
        df = DataFrame({"a": range(10), "b": range(10, 20)})
        result = df.tail(3)
        assert isinstance(result, DataFrame)
        assert len(result) == 3

    def test_copy(self):
        """Test copy() creates an independent copy."""
        df1 = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = df1.copy()

        assert isinstance(df2, DataFrame)
        assert df1._df is not df2._df  # Different underlying DataFrames

        # Modify df2, df1 should remain unchanged
        df2["c"] = [7, 8, 9]
        assert "c" in df2.columns
        assert "c" not in df1.columns


class TestDataFrameIndexProperty:
    """Test DataFrame index property."""

    def test_index_property_default(self):
        """Test default index is created."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # For now, we might just delegate to Polars or create a simple index
        # This test will be more detailed once we implement proper index support
        index = df.index
        assert index is not None

    def test_index_property_after_operations(self):
        """Test index after DataFrame operations."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = df.head(2)
        index = df2.index
        assert index is not None


class TestDataFrameLocIloc:
    """Test loc and iloc accessors."""

    def test_loc_exists(self):
        """Test that loc accessor exists."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert hasattr(df, "loc")
        # For now, just check it exists
        # Detailed tests will come when we implement the accessor

    def test_iloc_exists(self):
        """Test that iloc accessor exists."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert hasattr(df, "iloc")
        # For now, just check it exists
        # Detailed tests will come when we implement the accessor
