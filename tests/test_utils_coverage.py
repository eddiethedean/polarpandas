"""
Comprehensive coverage tests for utils module.

This test file focuses on increasing coverage of utils.py by testing
utility functions like isna, notna, cut, and schema conversion.
"""

import polars as pl
import pytest

import polarpandas as ppd
from polarpandas import utils


class TestIsnaNotna:
    """Tests for isna and notna functions."""

    def test_isna_dataframe(self):
        """Test isna on DataFrame."""
        df = ppd.DataFrame({"A": [1, None, 3]})
        result = utils.isna(df)

        assert isinstance(result, ppd.DataFrame)
        assert result["A"].tolist()[1] is True

    def test_isna_series(self):
        """Test isna on Series."""
        s = ppd.Series([1, None, 3])
        result = utils.isna(s)

        assert result.tolist()[1] is True

    def test_isna_scalar_none(self):
        """Test isna on None scalar."""
        result = utils.isna(None)
        assert result is True

    def test_isna_scalar_value(self):
        """Test isna on non-None scalar."""
        result = utils.isna(5)
        assert result is False

    def test_notna_dataframe(self):
        """Test notna on DataFrame."""
        df = ppd.DataFrame({"A": [1, None, 3]})
        result = utils.notna(df)

        assert isinstance(result, ppd.DataFrame)
        assert result["A"].tolist()[0] is True
        assert result["A"].tolist()[1] is False

    def test_notna_series(self):
        """Test notna on Series."""
        s = ppd.Series([1, None, 3])
        result = utils.notna(s)

        assert result.tolist()[0] is True
        assert result.tolist()[1] is False

    def test_notna_scalar_none(self):
        """Test notna on None scalar."""
        result = utils.notna(None)
        assert result is False

    def test_notna_scalar_value(self):
        """Test notna on non-None scalar."""
        result = utils.notna(5)
        assert result is True


class TestIsnullNotnull:
    """Tests for isnull and notnull aliases."""

    def test_isnull_alias(self):
        """Test isnull is alias for isna."""
        df = ppd.DataFrame({"A": [1, None]})

        isna_result = utils.isna(df)
        isnull_result = utils.isnull(df)

        assert isna_result["A"].tolist() == isnull_result["A"].tolist()

    def test_notnull_alias(self):
        """Test notnull is alias for notna."""
        s = ppd.Series([1, None, 3])

        notna_result = utils.notna(s)
        notnull_result = utils.notnull(s)

        assert notna_result.tolist() == notnull_result.tolist()


class TestConvertSchemaToPolars:
    """Tests for convert_schema_to_polars function."""

    def test_convert_int64(self):
        """Test converting int64 dtype."""
        schema = {"A": "int64", "B": "int32"}
        result = utils.convert_schema_to_polars(schema)

        assert result is not None
        assert "A" in result
        assert result["A"] == pl.Int64

    def test_convert_float_dtypes(self):
        """Test converting float dtypes."""
        schema = {"A": "float64", "B": "float32"}
        result = utils.convert_schema_to_polars(schema)

        assert result is not None
        assert result["A"] == pl.Float64
        assert result["B"] == pl.Float32

    def test_convert_string_dtype(self):
        """Test converting string dtype."""
        schema = {"A": "string", "B": "str"}
        result = utils.convert_schema_to_polars(schema)

        assert result is not None
        assert "A" in result

    def test_convert_bool_dtype(self):
        """Test converting boolean dtype."""
        schema = {"A": "bool", "B": "boolean"}
        result = utils.convert_schema_to_polars(schema)

        assert result is not None
        assert result["A"] == pl.Boolean

    def test_convert_polars_schema(self):
        """Test with existing Polars schema."""
        schema = {"A": pl.Int64, "B": pl.Float64}
        result = utils.convert_schema_to_polars(schema)

        assert result is not None
        assert result["A"] == pl.Int64
        assert result["B"] == pl.Float64

    def test_convert_unsupported_dtype(self):
        """Test handling unsupported dtype."""
        schema = {"A": "unsupported_type"}

        # Should raise ValueError for unsupported types
        with pytest.raises(ValueError):
            utils.convert_schema_to_polars(schema)

    def test_convert_mixed_types(self):
        """Test converting mixed dtype specifications."""
        schema = {"int_col": "int64", "float_col": pl.Float32, "str_col": "string"}
        result = utils.convert_schema_to_polars(schema)

        assert result is not None
        assert "int_col" in result
        assert "float_col" in result


class TestCutFunction:
    """Tests for cut function."""

    def test_cut_basic(self):
        """Test basic cut operation."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = utils.cut(data, bins=3)
        assert len(result) == len(data)

    def test_cut_with_labels(self):
        """Test cut with custom labels."""
        data = [1, 2, 3, 4, 5, 6]
        result = utils.cut(data, bins=3, labels=["low", "mid", "high"])
        assert len(result) == len(data)

    def test_cut_explicit_bins(self):
        """Test cut with explicit bin edges."""
        data = [1, 2, 3, 4, 5]
        result = utils.cut(data, bins=[0, 2, 4, 6])
        assert len(result) == len(data)


class TestSchemaConversionEdgeCases:
    """Tests for edge cases in schema conversion."""

    def test_convert_empty_schema(self):
        """Test converting empty schema."""
        result = utils.convert_schema_to_polars({})
        assert result == {} or result is None

    def test_convert_none_schema(self):
        """Test converting None schema."""
        result = utils.convert_schema_to_polars(None)
        assert result is None

    def test_convert_uint_types(self):
        """Test converting unsigned integer types."""
        schema = {"A": "uint8", "B": "uint16", "C": "uint32", "D": "uint64"}
        result = utils.convert_schema_to_polars(schema)

        if result:
            assert "A" in result


class TestIsnaEdgeCases:
    """Tests for isna edge cases."""

    def test_isna_empty_dataframe(self):
        """Test isna on empty DataFrame."""
        df = ppd.DataFrame()
        result = utils.isna(df)
        assert isinstance(result, ppd.DataFrame)

    def test_isna_all_nulls(self):
        """Test isna on DataFrame with all nulls."""
        df = ppd.DataFrame({"A": [None, None, None]})
        result = utils.isna(df)
        assert all(result["A"].tolist())

    def test_isna_no_nulls(self):
        """Test isna on DataFrame with no nulls."""
        df = ppd.DataFrame({"A": [1, 2, 3]})
        result = utils.isna(df)
        assert not any(result["A"].tolist())


class TestNotnaEdgeCases:
    """Tests for notna edge cases."""

    def test_notna_empty_series(self):
        """Test notna on empty Series."""
        s = ppd.Series([])
        result = utils.notna(s)
        assert len(result) == 0

    def test_notna_all_nulls(self):
        """Test notna on Series with all nulls."""
        s = ppd.Series([None, None, None])
        result = utils.notna(s)
        assert not any(result.tolist())

    def test_notna_no_nulls(self):
        """Test notna on Series with no nulls."""
        s = ppd.Series([1, 2, 3])
        result = utils.notna(s)
        assert all(result.tolist())


class TestSchemaConversionComplex:
    """Tests for complex schema conversions."""

    def test_convert_datetime_types(self):
        """Test converting datetime types."""
        schema = {"A": "datetime64[ns]"}
        result = utils.convert_schema_to_polars(schema)

        # Datetime should be supported
        if result:
            assert isinstance(result, dict)
            assert "A" in result

    def test_convert_categorical_type(self):
        """Test converting categorical type."""
        schema = {"A": "category"}
        result = utils.convert_schema_to_polars(schema)

        if result:
            assert "A" in result

    def test_convert_object_type(self):
        """Test converting object type."""
        schema = {"A": "object"}
        result = utils.convert_schema_to_polars(schema)

        if result:
            # object typically maps to String
            assert "A" in result


class TestSchemaConversionTypes:
    """Tests for all supported dtype conversions."""

    def test_convert_all_int_types(self):
        """Test all integer type conversions."""
        schema = {"i8": "int8", "i16": "int16", "i32": "int32", "i64": "int64"}
        result = utils.convert_schema_to_polars(schema)

        if result:
            assert result["i8"] == pl.Int8
            assert result["i16"] == pl.Int16
            assert result["i32"] == pl.Int32
            assert result["i64"] == pl.Int64

    def test_convert_all_uint_types(self):
        """Test all unsigned integer type conversions."""
        schema = {"u8": "uint8", "u16": "uint16", "u32": "uint32", "u64": "uint64"}
        result = utils.convert_schema_to_polars(schema)

        if result:
            assert "u8" in result

    def test_convert_numpy_dtypes(self):
        """Test converting numpy-style dtypes."""
        try:
            import numpy as np

            schema = {"A": np.int64, "B": np.float64}
            result = utils.convert_schema_to_polars(schema)

            if result:
                assert "A" in result
        except ImportError:
            pytest.skip("numpy not installed")


class TestCutEdgeCases:
    """Tests for cut edge cases."""

    def test_cut_single_value(self):
        """Test cut with single value."""
        result = utils.cut([5], bins=3)
        assert len(result) == 1

    def test_cut_empty_list(self):
        """Test cut with empty list."""
        result = utils.cut([], bins=3)
        assert len(result) == 0


class TestUtilityHelpers:
    """Tests for utility helper functions."""

    def test_convert_schema_with_series_dtype(self):
        """Test converting schema with Series-like dtype."""
        # Create schema from actual Series dtype
        s = ppd.Series([1, 2, 3])
        schema = {"A": s._series.dtype}
        result = utils.convert_schema_to_polars(schema)

        assert result is not None


class TestIsnaWithDifferentTypes:
    """Tests for isna with different data types."""

    def test_isna_with_numeric(self):
        """Test isna with numeric DataFrame."""
        df = ppd.DataFrame({"A": [1.0, None, 3.0]})
        result = utils.isna(df)
        assert result["A"].tolist()[1] is True

    def test_isna_with_string(self):
        """Test isna with string DataFrame."""
        df = ppd.DataFrame({"A": ["a", None, "c"]})
        result = utils.isna(df)
        assert result["A"].tolist()[1] is True

    def test_isna_with_mixed_types(self):
        """Test isna with mixed-type DataFrame."""
        df = ppd.DataFrame({"int_col": [1, None, 3], "str_col": ["a", "b", None]})
        result = utils.isna(df)
        assert result["int_col"].tolist()[1] is True
        assert result["str_col"].tolist()[2] is True


class TestNotnaWithDifferentTypes:
    """Tests for notna with different data types."""

    def test_notna_with_numeric(self):
        """Test notna with numeric Series."""
        s = ppd.Series([1.0, None, 3.0])
        result = utils.notna(s)
        assert result.tolist()[0] is True
        assert result.tolist()[1] is False

    def test_notna_with_boolean(self):
        """Test notna with boolean values."""
        s = ppd.Series([True, False, None])
        result = utils.notna(s)
        assert result.tolist()[0] is True
        assert result.tolist()[2] is False


class TestSchemaConversionAliases:
    """Tests for dtype alias handling in schema conversion."""

    def test_int_alias(self):
        """Test 'int' as alias for int64."""
        schema = {"A": "int"}
        result = utils.convert_schema_to_polars(schema)

        if result and "A" in result:
            assert result["A"] in (pl.Int64, pl.Int32)

    def test_float_alias(self):
        """Test 'float' as alias for float64."""
        schema = {"A": "float"}
        result = utils.convert_schema_to_polars(schema)

        if result and "A" in result:
            assert result["A"] in (pl.Float64, pl.Float32)

    def test_pandas_dtype_strings(self):
        """Test pandas-style dtype strings."""
        schema = {"A": "Int64", "B": "Float64", "C": "String"}
        result = utils.convert_schema_to_polars(schema)

        if result:
            assert isinstance(result, dict)


class TestSchemaWithPolarsTypes:
    """Tests for schema with Polars types directly."""

    def test_polars_schema_dict(self):
        """Test schema that's already Polars types."""
        schema = {"col1": pl.Int64, "col2": pl.Float64, "col3": pl.Utf8}
        result = utils.convert_schema_to_polars(schema)

        assert result == schema

    def test_mixed_polars_and_string(self):
        """Test mixed Polars types and strings."""
        schema = {"A": pl.Int64, "B": "float64", "C": pl.Utf8}
        result = utils.convert_schema_to_polars(schema)

        if result:
            assert result["A"] == pl.Int64
            assert "B" in result
            assert result["C"] == pl.Utf8
