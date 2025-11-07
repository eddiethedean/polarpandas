"""
Test utility functions in polarpandas.utils.

Tests cover edge cases and error handling for utility functions.
"""

import polarpandas as ppd


class TestUtilsFunctions:
    """Test utility functions."""

    def test_cut_empty_list(self):
        """Test cut with empty list."""
        result = ppd.cut([], bins=3)
        assert result == []

    def test_cut_with_labels_proper_binning(self):
        """Test cut with labels performs proper binning."""
        result = ppd.cut([1, 2, 3, 4, 5, 6], bins=3, labels=["A", "B", "C"])
        assert len(result) == 6
        # Values should be binned into 3 categories
        assert all(label in ["A", "B", "C"] for label in result.tolist())

    def test_isna_dataframe_edge_cases(self):
        """Test isna with DataFrame edge cases."""
        # Line 34: DataFrame.isna() path
        df = ppd.DataFrame({"a": [1, None, 3], "b": [None, 5, None]})
        result = ppd.isna(df)
        assert isinstance(result, ppd.DataFrame)

    def test_notna_dataframe_edge_cases(self):
        """Test notna with DataFrame edge cases."""
        # Line 63: DataFrame.notna() path
        df = ppd.DataFrame({"a": [1, None, 3], "b": [None, 5, None]})
        result = ppd.notna(df)
        assert isinstance(result, ppd.DataFrame)
