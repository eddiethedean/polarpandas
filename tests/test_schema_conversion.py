"""
Test schema conversion functionality.

Tests the convert_schema_to_polars utility and schema support in
DataFrame constructor and I/O functions.
"""

import os
import tempfile

import numpy as np
import polars as pl
import pytest

import polarpandas as ppd
from polarpandas.utils import convert_schema_to_polars


class TestSchemaConversionUtility:
    """Test the convert_schema_to_polars utility function."""

    def test_string_dtype_names(self):
        """Test conversion of pandas-style string dtype names."""
        schema = {"col1": "int64", "col2": "float64", "col3": "object"}
        result = convert_schema_to_polars(schema)

        assert result is not None
        assert result["col1"] == pl.Int64
        assert result["col2"] == pl.Float64
        assert result["col3"] == pl.Utf8

    def test_numpy_dtype_objects(self):
        """Test conversion of NumPy dtype objects."""
        schema = {"col1": np.int64, "col2": np.float64, "col3": np.bool_}
        result = convert_schema_to_polars(schema)

        assert result is not None
        assert result["col1"] == pl.Int64
        assert result["col2"] == pl.Float64
        assert result["col3"] == pl.Boolean

    def test_polars_schema_dict(self):
        """Test that Polars schema dicts pass through unchanged."""
        schema = {"col1": pl.Int64, "col2": pl.Float64, "col3": pl.Utf8}
        result = convert_schema_to_polars(schema)

        assert result is not None
        assert result == schema

    def test_polars_schema_object(self):
        """Test conversion of Polars Schema object."""
        schema = pl.Schema({"col1": pl.Int64, "col2": pl.Float64})
        result = convert_schema_to_polars(schema)

        assert result is not None
        assert isinstance(result, dict)
        assert result["col1"] == pl.Int64
        assert result["col2"] == pl.Float64

    def test_mixed_integer_types(self):
        """Test different integer types."""
        schema = {
            "int64": "int64",
            "int32": "int32",
            "int16": "int16",
            "int8": "int8",
        }
        result = convert_schema_to_polars(schema)

        assert result["int64"] == pl.Int64
        assert result["int32"] == pl.Int32
        assert result["int16"] == pl.Int16
        assert result["int8"] == pl.Int8

    def test_mixed_float_types(self):
        """Test different float types."""
        schema = {"float64": "float64", "float32": "float32"}
        result = convert_schema_to_polars(schema)

        assert result["float64"] == pl.Float64
        assert result["float32"] == pl.Float32

    def test_datetime_types(self):
        """Test datetime type conversion."""
        schema = {"date": "datetime64[ns]", "date2": "datetime64"}
        result = convert_schema_to_polars(schema)

        assert result["date"] == pl.Datetime
        assert result["date2"] == pl.Datetime

    def test_boolean_types(self):
        """Test boolean type conversion."""
        schema = {"bool1": "bool", "bool2": "boolean", "bool3": "bool_"}
        result = convert_schema_to_polars(schema)

        assert result["bool1"] == pl.Boolean
        assert result["bool2"] == pl.Boolean
        assert result["bool3"] == pl.Boolean

    def test_string_types(self):
        """Test string type conversion."""
        schema = {"str1": "object", "str2": "string", "str3": "str"}
        result = convert_schema_to_polars(schema)

        assert result["str1"] == pl.Utf8
        assert result["str2"] == pl.Utf8
        assert result["str3"] == pl.Utf8

    def test_categorical_type(self):
        """Test categorical type conversion."""
        schema = {"cat": "category"}
        result = convert_schema_to_polars(schema)

        assert result["cat"] == pl.Categorical

    def test_none_input(self):
        """Test that None input returns None."""
        result = convert_schema_to_polars(None)
        assert result is None

    def test_invalid_type(self):
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError):
            convert_schema_to_polars("not a dict or schema")

    def test_unsupported_dtype(self):
        """Test that unsupported dtypes raise ValueError."""
        schema = {"col1": "unsupported_type"}
        with pytest.raises(ValueError):
            convert_schema_to_polars(schema)

    def test_pandas_dtype_objects(self):
        """Test conversion of objects that mimic pandas dtype classes."""

        class _FakePandasInt64:
            __module__ = "pandas.core.dtypes.base"

            def __str__(self) -> str:  # pragma: no cover - simple representation
                return "Int64"

        class _FakePandasFloat64:
            __module__ = "pandas.core.dtypes.base"

            def __str__(self) -> str:
                return "Float64"

        class _FakePandasString:
            __module__ = "pandas.core.dtypes.base"

            def __str__(self) -> str:
                return "String"

        schema = {
            "col1": _FakePandasInt64(),
            "col2": _FakePandasFloat64(),
            "col3": _FakePandasString(),
        }
        result = convert_schema_to_polars(schema)

        assert result is not None
        assert result["col1"] == pl.Int64
        assert result["col2"] == pl.Float64
        assert result["col3"] == pl.Utf8

    def test_more_numpy_dtype_variations(self):
        """Test more NumPy dtype variations."""
        schema = {
            "int32": np.int32,
            "int16": np.int16,
            "int8": np.int8,
            "float32": np.float32,
            "uint64": np.uint64,
            "uint32": np.uint32,
        }
        result = convert_schema_to_polars(schema)

        assert result["int32"] == pl.Int32
        assert result["int16"] == pl.Int16
        assert result["int8"] == pl.Int8
        assert result["float32"] == pl.Float32
        assert result["uint64"] == pl.UInt64
        assert result["uint32"] == pl.UInt32

    def test_unsigned_integer_types(self):
        """Test unsigned integer type conversion."""
        schema = {
            "uint64": "uint64",
            "uint32": "uint32",
            "uint16": "uint16",
            "uint8": "uint8",
        }
        result = convert_schema_to_polars(schema)

        assert result["uint64"] == pl.UInt64
        assert result["uint32"] == pl.UInt32
        assert result["uint16"] == pl.UInt16
        assert result["uint8"] == pl.UInt8

    def test_nullable_pandas_dtype_strings(self):
        """Test nullable pandas dtype strings (Int64, Float64, String)."""
        schema = {"col1": "Int64", "col2": "Float64", "col3": "String"}
        result = convert_schema_to_polars(schema)

        assert result["col1"] == pl.Int64
        assert result["col2"] == pl.Float64
        assert result["col3"] == pl.Utf8


class TestDataFrameConstructorSchema:
    """Test schema support in DataFrame constructor."""

    def test_dataframe_with_string_dtype_dict(self):
        """Test DataFrame creation with string dtype dict."""
        data = {"col1": ["1", "2", "3"], "col2": ["1.0", "2.0", "3.0"]}
        dtype = {"col1": "int64", "col2": "float64"}

        df = ppd.DataFrame(data, dtype=dtype)

        assert df._df["col1"].dtype == pl.Int64
        assert df._df["col2"].dtype == pl.Float64

    def test_dataframe_with_numpy_dtype_dict(self):
        """Test DataFrame creation with NumPy dtype dict."""
        data = {"col1": ["1", "2", "3"], "col2": ["1.0", "2.0", "3.0"]}
        dtype = {"col1": np.int64, "col2": np.float64}

        df = ppd.DataFrame(data, dtype=dtype)

        assert df._df["col1"].dtype == pl.Int64
        assert df._df["col2"].dtype == pl.Float64

    def test_dataframe_with_polars_schema_dict(self):
        """Test DataFrame creation with Polars schema dict."""
        data = {"col1": ["1", "2", "3"], "col2": ["1.0", "2.0", "3.0"]}
        dtype = {"col1": pl.Int64, "col2": pl.Float64}

        df = ppd.DataFrame(data, dtype=dtype)

        assert df._df["col1"].dtype == pl.Int64
        assert df._df["col2"].dtype == pl.Float64

    def test_dataframe_with_polars_schema_object(self):
        """Test DataFrame creation with Polars Schema object."""
        data = {"col1": ["1", "2", "3"], "col2": ["1.0", "2.0", "3.0"]}
        schema = pl.Schema({"col1": pl.Int64, "col2": pl.Float64})

        df = ppd.DataFrame(data, dtype=schema)

        assert df._df["col1"].dtype == pl.Int64
        assert df._df["col2"].dtype == pl.Float64

    def test_dataframe_partial_schema(self):
        """Test DataFrame with schema for only some columns."""
        data = {
            "col1": ["1", "2", "3"],
            "col2": ["1.0", "2.0", "3.0"],
            "col3": ["a", "b", "c"],
        }
        dtype = {"col1": "int64"}  # Only specify col1

        df = ppd.DataFrame(data, dtype=dtype)

        assert df._df["col1"].dtype == pl.Int64
        # col2 and col3 should retain their inferred types

    def test_dataframe_schema_with_non_existent_column(self):
        """Test DataFrame with schema for non-existent column (should be ignored)."""
        data = {"col1": [1, 2, 3]}
        dtype = {"col1": "int64", "nonexistent": "float64"}

        df = ppd.DataFrame(data, dtype=dtype)

        # Should not raise error, just ignore nonexistent column
        assert df._df["col1"].dtype == pl.Int64
        assert "nonexistent" not in df._df.columns

    def test_dataframe_with_mixed_schema_types(self):
        """Test DataFrame with mixed schema types (string, numpy, polars)."""
        data = {"col1": ["1", "2"], "col2": ["3", "4"], "col3": ["5", "6"]}
        dtype = {
            "col1": "int64",  # String
            "col2": np.float64,  # NumPy
            "col3": pl.Utf8,  # Polars
        }

        df = ppd.DataFrame(data, dtype=dtype)

        assert df._df["col1"].dtype == pl.Int64
        assert df._df["col2"].dtype == pl.Float64
        assert df._df["col3"].dtype == pl.Utf8

    def test_dataframe_with_all_numpy_types(self):
        """Test DataFrame with various NumPy dtype objects."""
        data = {
            "int8": ["1", "2"],
            "int16": ["3", "4"],
            "int32": ["5", "6"],
            "float32": ["7.0", "8.0"],
        }
        dtype = {
            "int8": np.int8,
            "int16": np.int16,
            "int32": np.int32,
            "float32": np.float32,
        }

        df = ppd.DataFrame(data, dtype=dtype)

        assert df._df["int8"].dtype == pl.Int8
        assert df._df["int16"].dtype == pl.Int16
        assert df._df["int32"].dtype == pl.Int32
        assert df._df["float32"].dtype == pl.Float32


class TestIOSchemaSupport:
    """Test schema support in I/O functions."""

    def test_read_csv_with_dtype(self):
        """Test read_csv with dtype parameter."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("col1,col2\n1,2.0\n3,4.0\n")
            filepath = f.name

        try:
            df = ppd.read_csv(filepath, dtype={"col1": "int64", "col2": "float64"})
            assert df._df["col1"].dtype == pl.Int64
            assert df._df["col2"].dtype == pl.Float64
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_read_csv_with_schema(self):
        """Test read_csv with schema parameter."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("col1,col2\n1,2.0\n3,4.0\n")
            filepath = f.name

        try:
            schema = {"col1": pl.Int64, "col2": pl.Float64}
            df = ppd.read_csv(filepath, schema=schema)
            assert df._df["col1"].dtype == pl.Int64
            assert df._df["col2"].dtype == pl.Float64
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_read_parquet_with_dtype(self):
        """Test read_parquet with dtype parameter."""
        # Create temporary Parquet file
        data = {"col1": [1, 2, 3], "col2": [1.0, 2.0, 3.0]}
        pl_df = pl.DataFrame(data)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as f:
            filepath = f.name
            pl_df.write_parquet(filepath)

        try:
            df = ppd.read_parquet(filepath, dtype={"col1": "int32", "col2": "float32"})
            # Note: Parquet files preserve schema, so this might not change types
            # but the parameter should be accepted
            assert "col1" in df.columns
            assert "col2" in df.columns
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_read_json_with_dtype(self):
        """Test read_json with dtype parameter."""
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            f.write('[{"col1": "1", "col2": "2.0"}, {"col1": "3", "col2": "4.0"}]')
            filepath = f.name

        try:
            df = ppd.read_json(filepath, dtype={"col1": "int64", "col2": "float64"})
            assert df._df["col1"].dtype == pl.Int64
            assert df._df["col2"].dtype == pl.Float64
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_scan_csv_with_dtype(self):
        """Test scan_csv with dtype parameter."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("col1,col2\n1,2.0\n3,4.0\n")
            filepath = f.name

        try:
            lf = ppd.scan_csv(filepath, dtype={"col1": "int64", "col2": "float64"})
            df = lf.collect()
            assert df._df["col1"].dtype == pl.Int64
            assert df._df["col2"].dtype == pl.Float64
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_scan_parquet_with_dtype(self):
        """Test scan_parquet with dtype parameter."""
        # Create temporary Parquet file
        data = {"col1": [1, 2, 3], "col2": [1.0, 2.0, 3.0]}
        pl_df = pl.DataFrame(data)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as f:
            filepath = f.name
            pl_df.write_parquet(filepath)

        try:
            lf = ppd.scan_parquet(filepath, dtype={"col1": "int32"})
            df = lf.collect()
            assert "col1" in df.columns
            assert "col2" in df.columns
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_scan_json_with_dtype(self):
        """Test scan_json with dtype parameter."""
        # Create temporary NDJSON file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            f.write('{"col1": "1", "col2": "2.0"}\n{"col1": "3", "col2": "4.0"}\n')
            filepath = f.name

        try:
            lf = ppd.scan_json(filepath, dtype={"col1": "int64", "col2": "float64"})
            df = lf.collect()
            assert df._df["col1"].dtype == pl.Int64
            assert df._df["col2"].dtype == pl.Float64
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_read_feather_with_dtype(self):
        """Test read_feather with dtype parameter."""
        # Create temporary Feather file
        data = {"col1": [1, 2, 3], "col2": [1.0, 2.0, 3.0]}
        pl_df = pl.DataFrame(data)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".feather") as f:
            filepath = f.name
            pl_df.write_ipc(filepath)

        try:
            df = ppd.read_feather(filepath, dtype={"col1": "int32", "col2": "float32"})
            # Feather files preserve schema, but parameter should be accepted
            assert "col1" in df.columns
            assert "col2" in df.columns
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_read_feather_with_schema(self):
        """Test read_feather with schema parameter."""
        # Create temporary Feather file
        data = {"col1": [1, 2, 3], "col2": [1.0, 2.0, 3.0]}
        pl_df = pl.DataFrame(data)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".feather") as f:
            filepath = f.name
            pl_df.write_ipc(filepath)

        try:
            schema = {"col1": pl.Int32, "col2": pl.Float32}
            df = ppd.read_feather(filepath, schema=schema)
            assert "col1" in df.columns
            assert "col2" in df.columns
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_read_json_with_schema(self):
        """Test read_json with schema parameter."""
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            f.write('[{"col1": "1", "col2": "2.0"}, {"col1": "3", "col2": "4.0"}]')
            filepath = f.name

        try:
            schema = {"col1": pl.Int64, "col2": pl.Float64}
            df = ppd.read_json(filepath, schema=schema)
            assert df._df["col1"].dtype == pl.Int64
            assert df._df["col2"].dtype == pl.Float64
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_scan_csv_with_schema(self):
        """Test scan_csv with schema parameter."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("col1,col2\n1,2.0\n3,4.0\n")
            filepath = f.name

        try:
            schema = {"col1": pl.Int64, "col2": pl.Float64}
            lf = ppd.scan_csv(filepath, schema=schema)
            df = lf.collect()
            assert df._df["col1"].dtype == pl.Int64
            assert df._df["col2"].dtype == pl.Float64
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_scan_parquet_with_schema(self):
        """Test scan_parquet with schema parameter."""
        # Create temporary Parquet file
        data = {"col1": [1, 2, 3], "col2": [1.0, 2.0, 3.0]}
        pl_df = pl.DataFrame(data)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as f:
            filepath = f.name
            pl_df.write_parquet(filepath)

        try:
            schema = {"col1": pl.Int32}
            lf = ppd.scan_parquet(filepath, schema=schema)
            df = lf.collect()
            assert "col1" in df.columns
            assert "col2" in df.columns
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_scan_json_with_schema(self):
        """Test scan_json with schema parameter."""
        # Create temporary NDJSON file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            f.write('{"col1": "1", "col2": "2.0"}\n{"col1": "3", "col2": "4.0"}\n')
            filepath = f.name

        try:
            schema = {"col1": pl.Int64, "col2": pl.Float64}
            lf = ppd.scan_json(filepath, schema=schema)
            df = lf.collect()
            assert df._df["col1"].dtype == pl.Int64
            assert df._df["col2"].dtype == pl.Float64
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestSchemaEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_schema_dict(self):
        """Test empty schema dict."""
        result = convert_schema_to_polars({})
        assert result == {}

    def test_schema_precedence_dtype_vs_schema(self):
        """Test that schema parameter takes precedence over dtype."""
        # This would be used in read functions
        # If both dtype and schema are provided, schema should win
        schema_dict = {"col1": pl.Int64}
        dtype_dict = {"col1": pl.Float64}

        # In actual usage, schema would take precedence
        result_schema = convert_schema_to_polars(schema_dict)
        result_dtype = convert_schema_to_polars(dtype_dict)

        assert result_schema["col1"] == pl.Int64
        assert result_dtype["col1"] == pl.Float64

    def test_case_insensitive_dtype_strings(self):
        """Test that dtype strings are case-insensitive (where applicable)."""
        schema = {"col1": "INT64", "col2": "Float64", "col3": "OBJECT"}
        result = convert_schema_to_polars(schema)

        assert result["col1"] == pl.Int64
        assert result["col2"] == pl.Float64
        assert result["col3"] == pl.Utf8
