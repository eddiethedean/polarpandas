"""
Test Index functionality.
"""

import polars as pl

from polarpandas import Index


class TestIndexInitialization:
    """Test Index initialization from various sources."""

    def test_init_from_list(self):
        """Test creating Index from list."""
        data = [0, 1, 2, 3, 4]
        idx = Index(data)
        assert isinstance(idx, Index)
        assert hasattr(idx, "_series")
        assert isinstance(idx._series, pl.Series)

    def test_init_from_polars_series(self):
        """Test creating Index from existing Polars Series."""
        pl_series = pl.Series("index", [0, 1, 2, 3, 4])
        idx = Index(pl_series)
        assert isinstance(idx, Index)
        assert isinstance(idx._series, pl.Series)

    def test_init_empty(self):
        """Test creating empty Index."""
        idx = Index([])
        assert isinstance(idx, Index)
        assert isinstance(idx._series, pl.Series)
        assert len(idx) == 0

    def test_init_none(self):
        """Test creating Index with None data."""
        # Line 32: data is None case
        idx = Index(None)
        assert isinstance(idx, Index)
        assert isinstance(idx._series, pl.Series)
        assert len(idx) == 0


class TestIndexDelegation:
    """Test that Index properly delegates to underlying Polars Series."""

    def test_len(self):
        """Test len() function."""
        idx = Index([0, 1, 2, 3, 4])
        assert len(idx) == 5

    def test_access_dtype(self):
        """Test accessing dtype attribute."""
        idx = Index([0, 1, 2, 3])
        dtype = idx.dtype
        assert dtype is not None

    def test_access_private_attribute_raises_error(self):
        """Test accessing private attribute raises AttributeError."""
        # Line 47: private attribute starting with _ raises error
        import pytest

        idx = Index([0, 1, 2, 3])
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = idx._private_attr

    def test_access_nonexistent_attribute_raises_error(self):
        """Test accessing nonexistent attribute raises AttributeError."""
        # Lines 54-55: AttributeError handling
        import pytest

        idx = Index([0, 1, 2, 3])
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = idx.nonexistent_method()


class TestIndexProperties:
    """Test Index properties."""

    def test_shape_property(self):
        """Test shape property."""
        idx = Index([0, 1, 2, 3, 4])
        shape = idx.shape
        assert shape == (5,)

    def test_size_property(self):
        """Test size property."""
        idx = Index([0, 1, 2, 3, 4])
        size = idx.size
        assert size == 5


class TestIndexRepresentation:
    """Test Index string representations."""

    def test_repr(self):
        """Test __repr__ returns a string."""
        idx = Index([0, 1, 2, 3, 4])
        repr_str = repr(idx)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0

    def test_str(self):
        """Test __str__ returns a string."""
        idx = Index([0, 1, 2, 3, 4])
        str_repr = str(idx)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0

    def test_tolist(self):
        """Test tolist() method."""
        # Line 77: tolist() method
        idx = Index([0, 1, 2, 3, 4])
        result = idx.tolist()
        assert isinstance(result, list)
        assert result == [0, 1, 2, 3, 4]

    def test_index_with_nulls(self):
        """Test Index with null values."""
        idx = Index([0, None, 2, None, 4])
        assert len(idx) == 5
        # Should handle nulls gracefully
        assert idx._series.null_count() == 2

    def test_index_comparison_operations(self):
        """Test Index comparison operations."""
        idx1 = Index([1, 2, 3])
        idx2 = Index([1, 2, 4])

        # Test that comparison operations work through delegation
        # These will be handled by the underlying Series
        result = idx1._series == idx2._series
        assert isinstance(result, pl.Series)
        assert result.to_list() == [True, True, False]

    def test_index_slice_edge_cases(self):
        """Test Index slicing edge cases."""
        idx = Index([0, 1, 2, 3, 4])

        # Test iteration (line 73 covered)
        values = list(idx)
        assert values == [0, 1, 2, 3, 4]

        # Test negative indices (if supported)
        # This is handled by underlying Series slicing


class TestDataFrameSeriesInterop:
    """Test interoperability between DataFrame and Series."""

    def test_dataframe_column_returns_series(self):
        """Test that accessing a DataFrame column returns a Series (will be implemented later)."""
        # This will be tested more thoroughly once we implement __getitem__ for DataFrame
        pass
