"""
Test DataFrame internal state and bug fixes.

These tests are specifically designed to catch bugs related to:
1. DataFrame._df being a LazyFrame instead of DataFrame
2. Indexer classes using wrong references
3. Undefined variables in set_index
4. Rolling operations using wrong references
"""

import polars as pl

import polarpandas as ppd


class TestDataFrameInternalState:
    """Test that DataFrame internal state is correct."""

    def test_df_is_always_dataframe_not_lazyframe(self):
        """Test that _df attribute is always a DataFrame, never a LazyFrame."""
        # Test with various DataFrame operations
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert isinstance(df._df, pl.DataFrame)
        assert not hasattr(df._df, "collect")

        # Test after set_index with drop=True and all columns
        df_all_columns = ppd.DataFrame({"a": [1, 2, 3]})
        df_all_columns.set_index("a", drop=True, inplace=True)
        assert isinstance(df_all_columns._df, pl.DataFrame)
        assert not hasattr(df_all_columns._df, "collect")

        # Test after various operations that might return LazyFrame
        df_filtered = df[df["a"] > 1]
        assert isinstance(df_filtered._df, pl.DataFrame)
        assert not hasattr(df_filtered._df, "collect")

        df_selected = df[["a"]]
        assert isinstance(df_selected._df, pl.DataFrame)

        # Test after operations that preserve DataFrame
        df_sorted = df.sort_values("a")
        assert isinstance(df_sorted._df, pl.DataFrame)

    def test_df_is_dataframe_after_set_index_drop_all_columns(self):
        """Test that set_index with drop=True using all columns returns DataFrame."""
        # This would have failed with the bug: self._df = pl.DataFrame().lazy()
        df = ppd.DataFrame({"a": [1, 2, 3]})
        df.set_index("a", drop=True, inplace=True)
        assert isinstance(df._df, pl.DataFrame)
        # Should be able to use DataFrame methods without collect()
        assert hasattr(df._df, "height")
        assert hasattr(df._df, "columns")

    def test_set_index_append_inplace_false_single_column(self):
        """Test set_index with append=True, inplace=False, single column (bug fix)."""
        # This would have failed with undefined 'materialized' variable
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df_indexed = df.set_index("a")

        # Now append with inplace=False
        result = df_indexed.set_index("b", append=True, inplace=False)
        assert isinstance(result, ppd.DataFrame)
        assert isinstance(result._df, pl.DataFrame)
        assert result._index is not None
        # Should be list of tuples
        assert all(isinstance(idx, tuple) for idx in result._index)
        assert len(result._index[0]) == 2  # (a, b)

    def test_set_index_append_inplace_false_multiple_columns(self):
        """Test set_index with append=True, inplace=False, multiple columns (bug fix)."""
        # This would have failed with undefined 'materialized' variable
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        df_indexed = df.set_index("a")

        # Now append multiple columns with inplace=False
        result = df_indexed.set_index(["b", "c"], append=True, inplace=False)
        assert isinstance(result, ppd.DataFrame)
        assert isinstance(result._df, pl.DataFrame)
        assert result._index is not None
        # Should be list of tuples
        assert all(isinstance(idx, tuple) for idx in result._index)
        assert len(result._index[0]) == 3  # (a, b, c)

    def test_set_index_replace_inplace_false_multiple_columns(self):
        """Test set_index replace with inplace=False, multiple columns (bug fix)."""
        # This would have failed with undefined 'materialized' variable
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

        result = df.set_index(["b", "c"], inplace=False)
        assert isinstance(result, ppd.DataFrame)
        assert isinstance(result._df, pl.DataFrame)
        assert result._index is not None
        # Should be list of tuples
        assert all(isinstance(idx, tuple) for idx in result._index)
        assert len(result._index[0]) == 2  # (b, c)


class TestIndexerClasses:
    """Test that indexer classes use correct references."""

    def test_loc_uses_correct_reference(self):
        """Test that loc indexer uses self._df._df, not self._df."""
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=["x", "y", "z"])

        # Single cell access
        value = df.loc["x", "a"]
        assert value == 1

        # Single row access
        row = df.loc["x"]
        assert isinstance(row, ppd.Series)

    def test_iloc_uses_correct_reference(self):
        """Test that iloc indexer uses self._df._df, not self._df."""
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Single cell access
        value = df.iloc[0, 0]
        assert value == 1

        # Slice access
        subset = df.iloc[0:2]
        assert isinstance(subset, ppd.DataFrame)
        assert len(subset) == 2

        # List access
        subset = df.iloc[[0, 2]]
        assert isinstance(subset, ppd.DataFrame)
        assert len(subset) == 2

    def test_at_uses_correct_reference(self):
        """Test that at indexer uses self._df._df, not self._df."""
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=["x", "y", "z"])

        # Access with labels
        value = df.at["x", "a"]
        assert value == 1

        # Without index, use integer position
        df2 = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        value = df2.at[0, "a"]
        assert value == 1

    def test_iat_uses_correct_reference(self):
        """Test that iat indexer uses self._df._df, not self._df."""
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Integer position access
        value = df.iat[0, 0]
        assert value == 1

        value = df.iat[2, 1]
        assert value == 6

    def test_loc_get_rows_cols(self):
        """Test loc._get_rows_cols uses correct reference."""
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=["x", "y", "z"])

        # Single cell
        value = df.loc["x", "a"]
        assert value == 1

        # Single row, single column
        value = df.loc["y", "b"]
        assert value == 5

        # Single row, multiple columns
        row = df.loc["x", ["a", "b"]]
        assert isinstance(row, ppd.Series)

    def test_iloc_get_rows_cols(self):
        """Test iloc._get_rows_cols uses correct reference."""
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Single cell
        value = df.iloc[0, 0]
        assert value == 1

        # Single row, single column by name
        value = df.iloc[1, 1]
        assert value == 5

        # Multiple rows, multiple columns - tests _get_rows_cols with correct reference
        subset = df.iloc[[0, 1], ["a", "b"]]
        assert isinstance(subset, ppd.DataFrame)
        assert len(subset) == 2


class TestRollingOperations:
    """Test rolling operations use correct references."""

    def test_rolling_mean_uses_correct_reference(self):
        """Test that rolling.mean() uses self._df._df, not self._df."""
        df = ppd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

        result = df.rolling(3).mean()
        assert isinstance(result, ppd.DataFrame)
        assert isinstance(result._df, pl.DataFrame)
        assert len(result) == 5
        assert "a" in result.columns
        assert "b" in result.columns

    def test_rolling_sum_uses_correct_reference(self):
        """Test that rolling.sum() uses self._df._df, not self._df."""
        df = ppd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

        result = df.rolling(3).sum()
        assert isinstance(result, ppd.DataFrame)
        assert isinstance(result._df, pl.DataFrame)
        assert len(result) == 5

    def test_rolling_std_uses_correct_reference(self):
        """Test that rolling.std() uses self._df._df, not self._df."""
        df = ppd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

        result = df.rolling(3).std()
        assert isinstance(result, ppd.DataFrame)
        assert isinstance(result._df, pl.DataFrame)
        assert len(result) == 5

    def test_rolling_max_uses_correct_reference(self):
        """Test that rolling.max() uses self._df._df, not self._df."""
        df = ppd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

        result = df.rolling(3).max()
        assert isinstance(result, ppd.DataFrame)
        assert isinstance(result._df, pl.DataFrame)
        assert len(result) == 5

    def test_rolling_min_uses_correct_reference(self):
        """Test that rolling.min() uses self._df._df, not self._df."""
        df = ppd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

        result = df.rolling(3).min()
        assert isinstance(result, ppd.DataFrame)
        assert isinstance(result._df, pl.DataFrame)
        assert len(result) == 5


class TestNlargestNsmallest:
    """Test nlargest and nsmallest after removing materialized variables."""

    def test_nlargest_preserves_dataframe(self):
        """Test that nlargest returns DataFrame after bug fix."""
        df = ppd.DataFrame({"a": [5, 1, 4, 2, 3], "b": [10, 20, 30, 40, 50]})

        result = df.nlargest(3, "a")
        assert isinstance(result, ppd.DataFrame)
        assert isinstance(result._df, pl.DataFrame)
        assert len(result) == 3
        # Should be sorted descending
        assert result["a"].values[0] == 5

    def test_nsmallest_preserves_dataframe(self):
        """Test that nsmallest returns DataFrame after bug fix."""
        df = ppd.DataFrame({"a": [5, 1, 4, 2, 3], "b": [10, 20, 30, 40, 50]})

        result = df.nsmallest(3, "a")
        assert isinstance(result, ppd.DataFrame)
        assert isinstance(result._df, pl.DataFrame)
        assert len(result) == 3
        # Should be sorted ascending
        assert result["a"].values[0] == 1

    def test_nlargest_with_index(self):
        """Test nlargest with custom index preserves DataFrame."""
        df = ppd.DataFrame(
            {"a": [5, 1, 4, 2, 3], "b": [10, 20, 30, 40, 50]},
            index=["x", "y", "z", "w", "v"],
        )

        result = df.nlargest(2, "a")
        assert isinstance(result, ppd.DataFrame)
        assert isinstance(result._df, pl.DataFrame)
        assert result._index is not None


class TestDataFrameOperationsAfterCleanup:
    """Test various operations that relied on materialized variables."""

    def test_transpose_preserves_dataframe(self):
        """Test transpose doesn't use materialized variable incorrectly."""
        df = ppd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = df.transpose()
        assert isinstance(result, ppd.DataFrame)
        assert isinstance(result._df, pl.DataFrame)

    def test_to_csv_preserves_dataframe(self):
        """Test to_csv doesn't use materialized variable incorrectly."""
        import os
        import tempfile

        df = ppd.DataFrame({"a": [1, 2], "b": [3, 4]})

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_path = f.name

        try:
            df.to_csv(temp_path, index=False)
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_to_parquet_preserves_dataframe(self):
        """Test to_parquet doesn't use materialized variable incorrectly."""
        import os
        import tempfile

        df = ppd.DataFrame({"a": [1, 2], "b": [3, 4]})

        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as f:
            temp_path = f.name

        try:
            df.to_parquet(temp_path)
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_to_json_preserves_dataframe(self):
        """Test to_json doesn't use materialized variable incorrectly."""
        df = ppd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = df.to_json()
        assert isinstance(result, str)
        assert "a" in result or '"a"' in result

    def test_sample_preserves_dataframe(self):
        """Test sample doesn't use materialized variable incorrectly."""
        df = ppd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})
        result = df.sample(3)
        assert isinstance(result, ppd.DataFrame)
        assert isinstance(result._df, pl.DataFrame)
        assert len(result) == 3

    def test_pivot_preserves_dataframe(self):
        """Test pivot doesn't use materialized variable incorrectly."""
        df = ppd.DataFrame(
            {"a": ["A", "B", "A", "B"], "b": [1, 2, 3, 4], "c": [10, 20, 30, 40]}
        )
        result = df.pivot(index="a", columns="b", values="c")
        assert isinstance(result, ppd.DataFrame)
        assert isinstance(result._df, pl.DataFrame)

    def test_get_dummies_preserves_dataframe(self):
        """Test get_dummies doesn't use materialized variable incorrectly."""
        df = ppd.DataFrame({"category": ["A", "B", "A", "B"]})
        result = df.get_dummies()
        assert isinstance(result, ppd.DataFrame)
        assert isinstance(result._df, pl.DataFrame)
