"""
Test Series functionality.
"""

import polars as pl

from polarpandas import Series


class TestSeriesInitialization:
    """Test Series initialization from various sources."""

    def test_init_from_list(self):
        """Test creating Series from list."""
        data = [1, 2, 3, 4, 5]
        s = Series(data)
        assert isinstance(s, Series)
        assert hasattr(s, "_series")
        assert isinstance(s._series, pl.Series)

    def test_init_from_list_with_name(self):
        """Test creating Series from list with name."""
        data = [1, 2, 3, 4, 5]
        s = Series(data, name="test")
        assert isinstance(s, Series)
        assert s.name == "test"

    def test_init_from_polars_series(self):
        """Test creating Series from existing Polars Series."""
        pl_series = pl.Series("values", [1, 2, 3, 4, 5])
        s = Series(pl_series)
        assert isinstance(s, Series)
        assert isinstance(s._series, pl.Series)
        assert s.name == "values"

    def test_init_empty(self):
        """Test creating empty Series."""
        s = Series([])
        assert isinstance(s, Series)
        assert isinstance(s._series, pl.Series)
        assert len(s) == 0


class TestSeriesDelegation:
    """Test that Series properly delegates to underlying Polars Series."""

    def test_access_dtype(self):
        """Test accessing dtype attribute."""
        s = Series([1, 2, 3])
        dtype = s.dtype
        assert dtype is not None

    def test_len(self):
        """Test len() function."""
        s = Series([1, 2, 3, 4, 5])
        assert len(s) == 5

    def test_call_polars_method(self):
        """Test calling a Polars method through delegation."""
        s = Series([1, 2, 3, 4, 5])
        # Test that we can call Polars methods
        mean_val = s.mean()
        assert mean_val == 3.0


class TestSeriesProperties:
    """Test Series properties."""

    def test_name_property(self):
        """Test name property."""
        s = Series([1, 2, 3], name="test_series")
        assert s.name == "test_series"

    def test_shape_property(self):
        """Test shape property."""
        s = Series([1, 2, 3, 4, 5])
        shape = s.shape
        assert shape == (5,)

    def test_size_property(self):
        """Test size property."""
        s = Series([1, 2, 3, 4, 5])
        size = s.size
        assert size == 5


class TestSeriesRepresentation:
    """Test Series string representations."""

    def test_repr(self):
        """Test __repr__ returns a string."""
        s = Series([1, 2, 3, 4, 5])
        repr_str = repr(s)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0

    def test_str(self):
        """Test __str__ returns a string."""
        s = Series([1, 2, 3, 4, 5])
        str_repr = str(s)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0
