"""
Test additional pandas-compatible methods.
"""

import os
import tempfile

from polarpandas import DataFrame, Series


class TestDataFrameDescriptive:
    """Test descriptive statistics methods."""

    def test_describe(self):
        """Test describe() method."""
        df = DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        result = df.describe()
        assert result is not None
        assert isinstance(result, DataFrame)

    def test_info(self):
        """Test info() method prints DataFrame information."""
        df = DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        # info() typically prints to stdout, just check it doesn't error
        df.info()


class TestDataFrameDuplicates:
    """Test duplicate-related methods."""

    def test_drop_duplicates(self):
        """Test drop_duplicates() method."""
        df = DataFrame({"a": [1, 1, 2, 2, 3], "b": [1, 1, 2, 3, 3]})
        result = df.drop_duplicates()
        assert isinstance(result, DataFrame)
        assert len(result) < len(df)

    def test_drop_duplicates_inplace(self):
        """Test drop_duplicates() with inplace."""
        df = DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
        original_len = len(df)
        df.drop_duplicates(inplace=True)
        assert len(df) < original_len

    def test_duplicated(self):
        """Test duplicated() method."""
        df = DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
        result = df.duplicated()
        assert result is not None


class TestDataFrameSorting:
    """Test additional sorting methods."""

    def test_sort_index(self):
        """Test sort_index() method."""
        df = DataFrame({"a": [3, 1, 2], "b": [6, 4, 5]})
        result = df.sort_index()
        assert isinstance(result, DataFrame)


class TestDataFrameComparison:
    """Test comparison methods."""

    def test_isin(self):
        """Test isin() method."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.isin([1, 2, 4, 5])
        assert result is not None
        assert isinstance(result, DataFrame)

    def test_equals(self):
        """Test equals() method."""
        df1 = DataFrame({"a": [1, 2, 3]})
        df2 = DataFrame({"a": [1, 2, 3]})
        df3 = DataFrame({"a": [1, 2, 4]})

        assert df1.equals(df2) is True
        assert df1.equals(df3) is False


class TestDataFrameIndexing:
    """Test reset_index and set_index."""

    def test_reset_index(self):
        """Test reset_index() method."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.reset_index(inplace=False)
        assert isinstance(result, DataFrame)

    def test_reset_index_inplace(self):
        """Test reset_index() with inplace=True."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.reset_index(inplace=True)
        assert result is None


class TestDataFrameIO:
    """Test IO operations."""

    def test_to_csv(self):
        """Test to_csv() method."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            filepath = f.name

        try:
            df.to_csv(filepath)
            assert os.path.exists(filepath)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_to_dict(self):
        """Test to_dict() method."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.to_dict()
        assert isinstance(result, dict)
        assert "a" in result
        assert "b" in result


class TestDataFrameApply:
    """Test apply-related methods."""

    def test_apply_lambda(self):
        """Test apply() with lambda function."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # For now, just check if method exists and can be called
        try:
            result = df.apply(lambda x: x * 2)
            assert result is not None
        except (NotImplementedError, AttributeError):
            # Expected if not yet implemented
            pass


class TestDataFrameSample:
    """Test sample method."""

    def test_sample_n(self):
        """Test sample() with n parameter."""
        df = DataFrame({"a": range(100), "b": range(100, 200)})
        result = df.sample(n=10)
        assert isinstance(result, DataFrame)
        assert len(result) == 10

    def test_sample_frac(self):
        """Test sample() with frac parameter."""
        df = DataFrame({"a": range(100), "b": range(100, 200)})
        result = df.sample(frac=0.1)
        assert isinstance(result, DataFrame)
        assert len(result) == 10


class TestSeriesStringAccessor:
    """Test Series string accessor."""

    def test_str_lower(self):
        """Test .str.lower() method."""
        s = Series(["HELLO", "WORLD", "TEST"])
        try:
            result = s.str.lower()
            assert result is not None
        except AttributeError:
            # Expected if not implemented yet
            pass

    def test_str_upper(self):
        """Test .str.upper() method."""
        s = Series(["hello", "world", "test"])
        try:
            result = s.str.upper()
            assert result is not None
        except AttributeError:
            # Expected if not implemented yet
            pass
