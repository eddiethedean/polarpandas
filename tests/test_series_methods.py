"""
Test additional Series methods.
"""

from polarpandas import Series


class TestSeriesArithmetic:
    """Test Series arithmetic operations."""

    def test_series_addition(self):
        """Test Series addition."""
        s1 = Series([1, 2, 3])
        s2 = Series([4, 5, 6])
        result = s1 + s2
        # Result should be a Polars Series
        assert result is not None

    def test_series_scalar_addition(self):
        """Test Series scalar addition."""
        s = Series([1, 2, 3])
        result = s + 10
        # Result should work
        assert result is not None


class TestSeriesMethods:
    """Test Series-specific methods."""

    def test_unique(self):
        """Test unique() method."""
        s = Series([1, 2, 2, 3, 3, 3])
        result = s.unique()
        assert result is not None
        assert len(result) == 3

    def test_value_counts(self):
        """Test value_counts() method."""
        s = Series([1, 2, 2, 3, 3, 3])
        result = s.value_counts()
        assert result is not None

    def test_to_list(self):
        """Test to_list() method."""
        s = Series([1, 2, 3, 4, 5])
        result = s.to_list()
        assert result == [1, 2, 3, 4, 5]
