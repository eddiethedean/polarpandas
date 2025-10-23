"""
Test Index functionality.
"""
import pytest
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


class TestDataFrameSeriesInterop:
    """Test interoperability between DataFrame and Series."""
    
    def test_dataframe_column_returns_series(self):
        """Test that accessing a DataFrame column returns a Series (will be implemented later)."""
        # This will be tested more thoroughly once we implement __getitem__ for DataFrame
        pass

