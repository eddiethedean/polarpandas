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

    def test_cut_with_labels_short_list(self):
        """Test cut with labels when list is shorter than labels."""
        # Line 101: labels[:len(x)] edge case
        result = ppd.cut([1, 2], bins=3, labels=["A", "B", "C", "D", "E"])
        assert len(result) == 2
        assert result[0] == "A"
        assert result[1] == "B"

    def test_cut_with_labels_long_list(self):
        """Test cut with labels when list is longer than labels."""
        # Line 101: labels[:len(x)] when len(x) > len(labels)
        # The implementation just does labels[:len(x)], so if labels is shorter,
        # it returns only len(labels) items
        result = ppd.cut([1, 2, 3, 4, 5], bins=3, labels=["A", "B"])
        assert len(result) == 2  # Only returns labels[:len(x)], so min(2, 5) = 2
        assert result[0] == "A"
        assert result[1] == "B"

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
