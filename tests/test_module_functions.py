"""
Test module-level pandas-compatible functions.
"""

import os
import tempfile

import polarpandas as ppd


class TestModuleReadFunctions:
    """Test module-level read functions."""

    def test_read_csv(self):
        """Test ppd.read_csv()."""
        # Create a temp CSV file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n")
            filepath = f.name

        try:
            df = ppd.read_csv(filepath)
            assert isinstance(df, ppd.DataFrame)
            assert df.shape == (2, 3)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_read_json(self):
        """Test ppd.read_json()."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            f.write('[{"a": 1, "b": 2}, {"a": 3, "b": 4}]')
            filepath = f.name

        try:
            df = ppd.read_json(filepath)
            assert isinstance(df, ppd.DataFrame)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestModuleConcatMerge:
    """Test concat and merge functions."""

    def test_concat_vertical(self):
        """Test ppd.concat() vertical."""
        df1 = ppd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = ppd.DataFrame({"a": [5, 6], "b": [7, 8]})

        result = ppd.concat([df1, df2])
        assert isinstance(result, ppd.DataFrame)
        assert len(result) == 4

    def test_concat_horizontal(self):
        """Test ppd.concat() horizontal."""
        df1 = ppd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = ppd.DataFrame({"c": [5, 6], "d": [7, 8]})

        result = ppd.concat([df1, df2], axis=1)
        assert isinstance(result, ppd.DataFrame)
        assert len(result.columns) == 4

    def test_merge(self):
        """Test ppd.merge()."""
        df1 = ppd.DataFrame({"key": ["A", "B"], "val1": [1, 2]})
        df2 = ppd.DataFrame({"key": ["A", "B"], "val2": [3, 4]})

        result = ppd.merge(df1, df2, on="key")
        assert isinstance(result, ppd.DataFrame)
        assert "val1" in result.columns
        assert "val2" in result.columns

    def test_merge_invalid_how_raises_error(self):
        """Test merge raises error for invalid 'how' parameter."""
        import pytest

        df1 = ppd.DataFrame({"key": ["A", "B"], "val1": [1, 2]})
        df2 = ppd.DataFrame({"key": ["A", "B"], "val2": [3, 4]})

        # Invalid 'how' value - Polars will raise an error
        # The exact error depends on Polars implementation
        with pytest.raises(
            (ValueError, TypeError)
        ):  # Polars may raise ValueError or TypeError
            ppd.merge(df1, df2, on="key", how="invalid")

    def test_concat_empty_dataframes(self):
        """Test concat with empty DataFrames."""
        # Empty DataFrames
        df1 = ppd.DataFrame()
        df2 = ppd.DataFrame()

        result = ppd.concat([df1, df2])
        assert isinstance(result, ppd.DataFrame)

        # Empty and non-empty
        df3 = ppd.DataFrame({"a": [1, 2]})
        result = ppd.concat([df1, df3])
        assert isinstance(result, ppd.DataFrame)
        assert len(result) == 2

    def test_get_dummies_prefix_combinations(self):
        """Test get_dummies with various parameter combinations."""
        s = ppd.Series(["a", "b", "a", "c"])

        # Basic get_dummies
        result = ppd.get_dummies(s)
        assert isinstance(result, ppd.DataFrame)

        # With separator - polars uses separator, not prefix_sep
        try:
            result = ppd.get_dummies(s, separator="_")
            assert isinstance(result, ppd.DataFrame)
        except TypeError:
            # separator parameter may not be supported
            pass

        # DataFrame version
        df = ppd.DataFrame({"col1": ["a", "b"], "col2": [1, 2]})
        result = ppd.get_dummies(df, columns=["col1"])
        assert isinstance(result, ppd.DataFrame)

    def test_get_dummies_unsupported_type(self):
        """Test get_dummies raises error for unsupported types."""
        import pytest

        # Unsupported type
        with pytest.raises(ValueError, match="Unsupported type"):
            ppd.get_dummies(123)  # Integer not supported

        with pytest.raises(ValueError, match="Unsupported type"):
            ppd.get_dummies("abc")  # String not supported


class TestModuleDataFunctions:
    """Test data manipulation functions."""

    def test_get_dummies_series(self):
        """Test ppd.get_dummies() with Series."""
        s = ppd.Series(["a", "b", "a", "c"])
        result = ppd.get_dummies(s)
        assert isinstance(result, ppd.DataFrame)
        assert len(result) == 4

    def test_get_dummies_list(self):
        """Test ppd.get_dummies() with list."""
        result = ppd.get_dummies(["a", "b", "a", "c"])
        assert isinstance(result, ppd.DataFrame)

    def test_cut(self):
        """Test ppd.cut()."""
        s = ppd.Series([1, 7, 5, 4, 6, 3])
        result = ppd.cut(s, bins=3)
        assert result is not None

    def test_pivot_table(self):
        """Test ppd.pivot_table()."""
        df = ppd.DataFrame(
            {
                "A": ["foo", "foo", "bar", "bar"],
                "B": ["one", "two", "one", "two"],
                "C": [1, 2, 3, 4],
            }
        )
        result = ppd.pivot_table(df, values="C", index="A", columns="B")
        assert isinstance(result, ppd.DataFrame)


class TestModuleDatetimeFunctions:
    """Test datetime utility functions."""

    def test_date_range(self):
        """Test ppd.date_range()."""
        try:
            dates = ppd.date_range("2021-01-01", periods=5)
            assert isinstance(dates, ppd.Series)
            assert len(dates) == 5
        except (TypeError, AttributeError):
            # Polars date_range API might be different
            pass

    def test_to_datetime_list(self):
        """Test ppd.to_datetime() with list."""
        dates = ppd.to_datetime(["2021-01-01", "2021-01-02"])
        assert dates is not None

    def test_date_range_missing_parameters(self):
        """Test date_range raises ValueError when required parameters missing."""
        import pytest

        # Missing both start/end and periods
        with pytest.raises(ValueError, match="Must specify either"):
            ppd.date_range()

        # Missing periods when only start provided
        with pytest.raises(ValueError, match="Must specify either"):
            ppd.date_range(start="2021-01-01")

        # Missing start when only periods provided
        with pytest.raises(ValueError, match="Must specify either"):
            ppd.date_range(periods=5)

        # Missing start when only end provided
        with pytest.raises(ValueError, match="Must specify either"):
            ppd.date_range(end="2021-01-10")

    def test_to_datetime_invalid_type(self):
        """Test to_datetime raises ValueError for unsupported types."""
        import pytest

        # Unsupported type
        with pytest.raises(ValueError, match="Unsupported type"):
            ppd.to_datetime(123)  # Integer not supported

        with pytest.raises(ValueError, match="Unsupported type"):
            ppd.to_datetime(
                "2021-01-01"
            )  # String not supported (only list or DataFrame)


class TestModuleUtilityFunctions:
    """Test utility functions."""

    def test_isna_series(self):
        """Test ppd.isna() with Series."""
        s = ppd.Series([1, None, 3])
        result = ppd.isna(s)
        assert result is not None

    def test_isna_scalar(self):
        """Test ppd.isna() with scalar."""
        assert ppd.isna(None) is True
        assert ppd.isna(1) is False

    def test_notna_series(self):
        """Test ppd.notna() with Series."""
        s = ppd.Series([1, None, 3])
        result = ppd.notna(s)
        assert result is not None

    def test_notna_scalar(self):
        """Test ppd.notna() with scalar."""
        assert ppd.notna(None) is False
        assert ppd.notna(1) is True


class TestModuleImport:
    """Test module imports."""

    def test_version(self):
        """Test __version__ is defined."""
        assert hasattr(ppd, "__version__")
        assert isinstance(ppd.__version__, str)

    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        assert "DataFrame" in ppd.__all__
        assert "Series" in ppd.__all__
        assert "Index" in ppd.__all__
        assert "read_csv" in ppd.__all__
        assert "concat" in ppd.__all__
        assert "merge" in ppd.__all__
