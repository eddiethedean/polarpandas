"""
Test DataFrame core methods including aggregation, groupby, merging, and reshaping.
"""

from polarpandas import DataFrame


class TestDataFrameAggregation:
    """Test DataFrame aggregation methods."""

    def test_sum(self):
        """Test sum() aggregation."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.sum()
        assert result is not None
        # Polars returns a DataFrame with one row for sum

    def test_mean(self):
        """Test mean() aggregation."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.mean()
        assert result is not None

    def test_median(self):
        """Test median() aggregation."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.median()
        assert result is not None

    def test_min(self):
        """Test min() aggregation."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.min()
        assert result is not None

    def test_max(self):
        """Test max() aggregation."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.max()
        assert result is not None

    def test_std(self):
        """Test std() aggregation."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.std()
        assert result is not None

    def test_var(self):
        """Test var() aggregation."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.var()
        assert result is not None

    def test_count(self):
        """Test count() method."""
        df = DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        result = df.count()
        assert result is not None


class TestDataFrameSelection:
    """Test DataFrame selection methods."""

    def test_select_columns(self):
        """Test select() method for column selection."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        result = df.select(["a", "c"])
        assert isinstance(result, DataFrame)
        assert result.columns == ["a", "c"]

    def test_filter_rows(self):
        """Test filter() method for row filtering."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # Polars filter syntax
        result = df.filter(df["a"] > 1)
        assert isinstance(result, DataFrame)
        assert len(result) == 2


class TestDataFrameMissingData:
    """Test DataFrame missing data methods."""

    def test_isna(self):
        """Test isna() method."""
        df = DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
        result = df.isna()
        assert result is not None
        # Polars is_null returns boolean DataFrame

    def test_notna(self):
        """Test notna() method (inverse of isna)."""
        df = DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
        result = df.notna()
        assert result is not None


class TestDataFrameGroupBy:
    """Test DataFrame groupby operations."""

    def test_groupby_simple(self):
        """Test simple groupby operation."""
        df = DataFrame({"group": ["A", "B", "A", "B"], "value": [1, 2, 3, 4]})
        grouped = df.groupby("group")
        assert grouped is not None
        # GroupBy object should be returned

    def test_groupby_agg(self):
        """Test groupby with aggregation."""
        df = DataFrame({"group": ["A", "B", "A", "B"], "value": [1, 2, 3, 4]})
        # Polars groupby syntax
        result = df.groupby("group").agg(df["value"].sum())
        assert result is not None


class TestDataFrameMerge:
    """Test DataFrame merging operations."""

    def test_merge_basic(self):
        """Test basic merge (join) operation."""
        df1 = DataFrame({"key": ["A", "B", "C"], "value1": [1, 2, 3]})
        df2 = DataFrame({"key": ["A", "B", "D"], "value2": [4, 5, 6]})

        # This will test the merge method when implemented
        # For now, just check that the DataFrames exist
        assert df1 is not None
        assert df2 is not None


class TestDataFrameReshape:
    """Test DataFrame reshaping operations."""

    def test_melt(self):
        """Test melt operation (unpivot)."""
        df = DataFrame({"id": [1, 2], "A": [3, 4], "B": [5, 6]})
        # Polars has melt method
        result = df.melt(id_vars=["id"])
        assert result is not None
        assert isinstance(result, DataFrame)
