"""
Test set_index() method with pandas compatibility.

All tests compare polarpandas output against actual pandas output
to ensure 100% compatibility.
"""

import pandas as pd
import pytest

import polarpandas as ppd


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
        self.pd_df = pd.DataFrame(self.data)
        self.ppd_df = ppd.DataFrame(self.data)

    def test_set_index_single_column(self):
        """Test setting index to single column."""
        # Test with drop=True (default)
        pd_result = self.pd_df.set_index("A")
        ppd_result = self.ppd_df.set_index("A")
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

        # Test with drop=False
        pd_result = self.pd_df.set_index("A", drop=False)
        ppd_result = self.ppd_df.set_index("A", drop=False)
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_set_index_multiple_columns(self):
        """Test setting index to multiple columns."""
        # Test with drop=True (default)
        pd_result = self.pd_df.set_index(["A", "B"])
        ppd_result = self.ppd_df.set_index(["A", "B"])
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

        # Test with drop=False
        pd_result = self.pd_df.set_index(["A", "B"], drop=False)
        ppd_result = self.ppd_df.set_index(["A", "B"], drop=False)
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_set_index_inplace(self):
        """Test inplace parameter."""
        # Test inplace=True
        pd_df_copy = self.pd_df.copy()
        ppd_df_copy = self.ppd_df.copy()

        pd_result = pd_df_copy.set_index("A", inplace=True)
        ppd_result = ppd_df_copy.set_index("A", inplace=True)

        assert pd_result is None
        assert ppd_result is None
        pd.testing.assert_frame_equal(ppd_df_copy.to_pandas(), pd_df_copy)

        # Test inplace=False (default)
        pd_df_copy = self.pd_df.copy()
        ppd_df_copy = self.ppd_df.copy()

        pd_result = pd_df_copy.set_index("A", inplace=False)
        ppd_result = ppd_df_copy.set_index("A", inplace=False)

        assert pd_result is not None
        assert ppd_result is not None
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_set_index_append(self):
        """Test append parameter."""
        # First set an index
        pd_df_indexed = self.pd_df.set_index("A")
        ppd_df_indexed = self.ppd_df.set_index("A")

        # Test append=True
        pd_result = pd_df_indexed.set_index("B", append=True)
        ppd_result = ppd_df_indexed.set_index("B", append=True)
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

        # Test append=False
        pd_result = pd_df_indexed.set_index("B", append=False)
        ppd_result = ppd_df_indexed.set_index("B", append=False)
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_set_index_string_column(self):
        """Test setting index to string column."""
        pd_result = self.pd_df.set_index("C")
        ppd_result = self.ppd_df.set_index("C")
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_set_index_mixed_types(self):
        """Test setting index with mixed data types."""
        pd_result = self.pd_df.set_index(["A", "C"])
        ppd_result = self.ppd_df.set_index(["A", "C"])
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_set_index_with_duplicates(self):
        """Test setting index with duplicate values."""
        data_with_duplicates = {
            "A": [1, 2, 1, 2, 3],
            "B": [10, 20, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"],
        }
        pd_df = pd.DataFrame(data_with_duplicates)
        ppd_df = ppd.DataFrame(data_with_duplicates)

        pd_result = pd_df.set_index("A")
        ppd_result = ppd_df.set_index("A")
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_set_index_empty_dataframe(self):
        """Test set_index with empty DataFrame."""
        pd_empty = pd.DataFrame()
        ppd_empty = ppd.DataFrame()

        # Should raise error for empty DataFrame
        with pytest.raises(KeyError):
            pd_empty.set_index("A")
        with pytest.raises(KeyError):
            ppd_empty.set_index("A")

    def test_set_index_nonexistent_column(self):
        """Test set_index with non-existent column."""
        with pytest.raises(KeyError):
            self.pd_df.set_index("nonexistent")
        with pytest.raises(KeyError):
            self.ppd_df.set_index("nonexistent")

    def test_set_index_already_indexed(self):
        """Test set_index on already indexed DataFrame."""
        # Set initial index
        pd_df_indexed = self.pd_df.set_index("A")
        ppd_df_indexed = self.ppd_df.set_index("A")

        # Set new index
        pd_result = pd_df_indexed.set_index("B")
        ppd_result = ppd_df_indexed.set_index("B")
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_set_index_preserve_original(self):
        """Test that original DataFrame is not modified when inplace=False."""
        original_pd = self.pd_df.copy()
        original_ppd = self.ppd_df.copy()

        # Set index without inplace
        self.pd_df.set_index("A")
        self.ppd_df.set_index("A")

        # Original should be unchanged
        pd.testing.assert_frame_equal(original_pd, self.pd_df)
        pd.testing.assert_frame_equal(original_ppd.to_pandas(), self.ppd_df.to_pandas())

    @pytest.mark.skip(
        reason="Polars has limited support for null values in index - permanent limitation"
    )
    def test_set_index_with_nulls(self):
        """Test set_index with null values."""
        data_with_nulls = {
            "A": [1, None, 3, 4, 5],
            "B": [10, 20, None, 40, 50],
            "C": ["a", "b", "c", None, "e"],
        }
        pd_df = pd.DataFrame(data_with_nulls)
        ppd_df = ppd.DataFrame(data_with_nulls)

        # Test with single column containing nulls
        pd_result = pd_df.set_index("A")
        ppd_result = ppd_df.set_index("A")
        pd.testing.assert_frame_equal(
            ppd_result.to_pandas(), pd_result, check_index_type=False
        )

        # Skip multi-column test due to fundamental MultiIndex limitation
        # Polarpandas cannot create proper MultiIndex structures
        pytest.skip("Known limitation: MultiIndex creation with nulls not supported")

    def test_set_index_return_type(self):
        """Test that set_index returns correct type."""
        result = self.ppd_df.set_index("A")
        assert isinstance(result, ppd.DataFrame)

        # Test inplace=True returns None
        result = self.ppd_df.set_index("A", inplace=True)
        assert result is None

    def test_set_index_chain_operations(self):
        """Test chaining set_index operations."""
        # Chain multiple set_index operations
        pd_result = self.pd_df.set_index("A").set_index("B", append=True)
        ppd_result = self.ppd_df.set_index("A").set_index("B", append=True)
        pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_set_index_with_different_dtypes(self):
        """Test set_index with different data types."""
        data_mixed = {
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        }
        pd_df = pd.DataFrame(data_mixed)
        ppd_df = ppd.DataFrame(data_mixed)

        # Test each column type
        for col in data_mixed.keys():
            pd_result = pd_df.set_index(col)
            ppd_result = ppd_df.set_index(col)
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

    def test_set_index_error_handling(self):
        """Test error handling matches pandas."""
        # Test with invalid column name
        with pytest.raises(KeyError):
            self.pd_df.set_index("invalid")
        with pytest.raises(KeyError):
            self.ppd_df.set_index("invalid")

        # Test with empty list
        with pytest.raises(ValueError):
            self.pd_df.set_index([])
        with pytest.raises(ValueError):
            self.ppd_df.set_index([])

        # Test with None - pandas raises KeyError, not TypeError
        with pytest.raises(KeyError):
            self.pd_df.set_index(None)
        with pytest.raises(KeyError):
            self.ppd_df.set_index(None)
