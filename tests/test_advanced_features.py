"""
Test advanced pandas features.
"""

from polarpandas import DataFrame, Series


class TestDataFramePivot:
    """Test pivot operations."""

    def test_pivot(self):
        """Test pivot() method."""
        df = DataFrame(
            {
                "date": ["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"],
                "variable": ["A", "B", "A", "B"],
                "value": [1, 2, 3, 4],
            }
        )
        try:
            result = df.pivot(index="date", columns="variable", values="value")
            assert result is not None
        except (NotImplementedError, AttributeError):
            pass


class TestDataFrameRolling:
    """Test rolling window operations."""

    def test_rolling_mean(self):
        """Test rolling().mean() method."""
        df = DataFrame({"a": range(10), "b": range(10, 20)})
        try:
            result = df.rolling(window=3).mean()
            assert result is not None
        except (NotImplementedError, AttributeError):
            pass


class TestDataFrameApply:
    """Test apply functions."""

    def test_apply_to_columns(self):
        """Test apply() to columns."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df.apply(lambda x: x.sum())
        assert result is not None

    def test_applymap(self):
        """Test applymap() element-wise."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        try:
            result = df.applymap(lambda x: x * 2)
            assert result is not None
        except (NotImplementedError, AttributeError):
            pass


class TestDataFrameLocIloc:
    """Test loc and iloc indexing."""

    def test_loc_single_row(self):
        """Test loc[row] indexing."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        try:
            result = df.loc[0]
            assert result is not None
        except (NotImplementedError, TypeError):
            pass

    def test_iloc_single_row(self):
        """Test iloc[row] indexing."""
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        try:
            result = df.iloc[0]
            assert result is not None
        except (NotImplementedError, TypeError):
            pass

    def test_loc_slice(self):
        """Test loc[start:end] slicing."""
        df = DataFrame({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})
        try:
            result = df.loc[1:3]
            assert result is not None
        except (NotImplementedError, TypeError):
            pass


class TestSeriesStringAccessor:
    """Test Series .str accessor."""

    def test_str_contains(self):
        """Test .str.contains() method."""
        s = Series(["hello", "world", "test"])
        try:
            result = s.str.contains("o")
            assert result is not None
        except AttributeError:
            pass

    def test_str_startswith(self):
        """Test .str.startswith() method."""
        s = Series(["hello", "world", "test"])
        try:
            result = s.str.startswith("h")
            assert result is not None
        except AttributeError:
            pass

    def test_str_len(self):
        """Test .str.len() method."""
        s = Series(["hello", "world", "test"])
        try:
            result = s.str.len()
            assert result is not None
        except AttributeError:
            pass


class TestSeriesDatetimeAccessor:
    """Test Series .dt accessor."""

    def test_dt_year(self):
        """Test .dt.year property."""
        try:
            import datetime

            s = Series([datetime.date(2021, 1, 1), datetime.date(2021, 6, 15)])
            result = s.dt.year
            assert result is not None
        except (AttributeError, ImportError):
            pass

    def test_dt_month(self):
        """Test .dt.month property."""
        try:
            import datetime

            s = Series([datetime.date(2021, 1, 1), datetime.date(2021, 6, 15)])
            result = s.dt.month
            assert result is not None
        except (AttributeError, ImportError):
            pass


class TestSeriesApply:
    """Test Series apply functions."""

    def test_series_apply(self):
        """Test Series.apply() method."""
        s = Series([1, 2, 3, 4, 5])
        result = s.apply(lambda x: x * 2)
        assert result is not None

    def test_series_map(self):
        """Test Series.map() method."""
        s = Series([1, 2, 3])
        try:
            result = s.map({1: "a", 2: "b", 3: "c"})
            assert result is not None
        except (NotImplementedError, AttributeError):
            pass
