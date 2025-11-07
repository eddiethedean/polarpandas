"""I/O coverage tests that do not rely on pandas runtime."""

import csv
import json
import os
import pickle
import tempfile
from datetime import datetime

import polars as pl
import pytest

try:
    from pandas.errors import EmptyDataError
except ImportError:  # pragma: no cover - pandas may be optional in some environments

    class EmptyDataError(RuntimeError):
        """Fallback EmptyDataError when pandas is unavailable."""

        pass


import polarpandas as ppd
from tests.test_helpers import assert_frame_equal


class TestCSVIO:
    """Basic CSV read/write scenarios."""

    def _write_csv(self, rows, header, sep=","):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            writer = csv.writer(tmp, delimiter=sep)
            if header is not None:
                writer.writerow(header)
            writer.writerows(rows)
            tmp.flush()
            return tmp.name

    def test_read_csv_basic(self):
        path = self._write_csv(
            rows=[[1, 10, "a"], [2, 20, "b"], [3, 30, "c"]],
            header=["A", "B", "C"],
        )
        try:
            result = ppd.read_csv(path)
            assert_frame_equal(
                result, {"A": [1, 2, 3], "B": [10, 20, 30], "C": ["a", "b", "c"]}
            )
        finally:
            os.unlink(path)

    def test_read_csv_with_index(self):
        path = self._write_csv(
            rows=[[0, 1, 10], [1, 2, 20], [2, 3, 30]],
            header=["idx", "A", "B"],
        )
        try:
            result = ppd.read_csv(path, index_col=0)
            assert list(result.columns) == ["A", "B"]
            assert result.index.tolist() == [0, 1, 2]
        finally:
            os.unlink(path)

    def test_read_csv_with_separator(self):
        path = self._write_csv(
            rows=[[1, 10], [2, 20], [3, 30]],
            header=["A", "B"],
            sep=";",
        )
        try:
            result = ppd.read_csv(path, sep=";")
            assert_frame_equal(result, {"A": [1, 2, 3], "B": [10, 20, 30]})
        finally:
            os.unlink(path)

    def test_read_csv_skiprows(self):
        path = self._write_csv(
            rows=[["# comment", "ignore"], ["A", "B"], [1, 10], [2, 20]],
            header=None,
        )
        try:
            result = ppd.read_csv(path, skiprows=1)
            assert_frame_equal(result, {"A": [1, 2], "B": [10, 20]})
        finally:
            os.unlink(path)

    def test_read_csv_nrows(self):
        path = self._write_csv(
            rows=[[1, 10], [2, 20], [3, 30]],
            header=["A", "B"],
        )
        try:
            result = ppd.read_csv(path, nrows=2)
            assert_frame_equal(result, {"A": [1, 2], "B": [10, 20]})
        finally:
            os.unlink(path)

    def test_read_csv_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            path = tmp.name
        try:
            with pytest.raises(EmptyDataError):
                ppd.read_csv(path)
        finally:
            os.unlink(path)

    def test_to_csv_round_trip(self):
        df = ppd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            path = tmp.name
        try:
            df.to_csv(path, index=False)
            reloaded = ppd.read_csv(path)
            assert_frame_equal(reloaded, {"A": [1, 2], "B": ["x", "y"]})
        finally:
            os.unlink(path)

    def test_read_table_with_custom_delimiter(self):
        path = self._write_csv(
            rows=[[1, 10], [2, 20], [3, 30]],
            header=["A", "B"],
            sep="|",
        )
        try:
            result = ppd.read_table(path, sep="|")
            assert_frame_equal(result, {"A": [1, 2, 3], "B": [10, 20, 30]})
        finally:
            os.unlink(path)

    def test_scan_csv_with_dtype_override(self):
        path = self._write_csv(rows=[[1], [2]], header=["A"])
        try:
            lazy = ppd.scan_csv(path, dtype={"A": "str"})
            df = lazy.collect()
            assert df._df.dtypes == [pl.Utf8]
            assert df.to_dict() == {"A": ["1", "2"]}
        finally:
            os.unlink(path)


class TestIoFallbacks:
    """Optional dependency fallbacks."""

    def test_read_clipboard_without_pandas(self, monkeypatch):
        import builtins

        import polarpandas.io as io_mod

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("pandas unavailable for clipboard test")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        with pytest.raises(NotImplementedError):
            io_mod.read_clipboard()

    def test_read_html_without_pandas(self, monkeypatch):
        import builtins

        import polarpandas.io as io_mod

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("pandas unavailable for html test")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        with pytest.raises(NotImplementedError):
            io_mod.read_html("http://example.com")

    def test_read_pickle_round_trip(self):
        df = ppd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as tmp:
            pickle.dump(df, tmp)
            path = tmp.name
        try:
            loaded = ppd.read_pickle(path)
            assert isinstance(loaded, ppd.DataFrame)
            assert_frame_equal(loaded, {"a": [1, 2], "b": ["x", "y"]})
        finally:
            os.unlink(path)

    def test_read_hdf_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ppd.read_hdf("dummy.h5", key="table")

    def test_read_iceberg_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ppd.read_iceberg("s3://bucket/table")


class TestJSONIO:
    """JSON read/write helpers."""

    def test_read_json_basic(self):
        payload = [{"A": 1, "B": "x"}, {"A": 2, "B": "y"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(payload, tmp)
            path = tmp.name
        try:
            result = ppd.read_json(path)
            assert_frame_equal(result, {"A": [1, 2], "B": ["x", "y"]})
        finally:
            os.unlink(path)

    def test_to_json_round_trip(self):
        df = ppd.DataFrame({"A": [datetime(2023, 1, 1), datetime(2023, 1, 2)]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            path = tmp.name
        try:
            df.to_json(path)
            loaded = ppd.read_json(path)
            assert_frame_equal(
                loaded, {"A": ["2023-01-01 00:00:00", "2023-01-02 00:00:00"]}
            )
        finally:
            os.unlink(path)

    def test_json_methods_preserve_original(self):
        df = ppd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
        baseline = df.to_dict()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            path = tmp.name
        try:
            df.to_json(path)
            assert df.to_dict() == baseline
        finally:
            os.unlink(path)

    def test_scan_json_with_dtype(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write('{"value": "1"}\n{"value": "2"}\n')
            path = tmp.name
        try:
            lazy = ppd.scan_json(path, dtype={"value": "int64"})
            df = lazy.collect()
            assert df._df.dtypes == [pl.Int64]
            assert df.to_dict() == {"value": [1, 2]}
        finally:
            os.unlink(path)


class TestScanFunctions:
    """Spot-check lazy scan helpers."""

    def test_scan_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp.write("A,B\n1,10\n2,20\n")
            tmp.flush()
            path = tmp.name
        try:
            lazy = ppd.scan_csv(path)
            df = lazy.collect()
            assert_frame_equal(df, {"A": [1, 2], "B": [10, 20]})
        finally:
            os.unlink(path)

    def test_scan_parquet(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            path = tmp.name
        try:
            pl.DataFrame({"A": [1, 2]}).write_parquet(path)
            lazy = ppd.scan_parquet(path)
            df = lazy.collect()
            assert_frame_equal(df, {"A": [1, 2]})
        finally:
            os.unlink(path)

    def test_scan_parquet_with_schema_cast(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            path = tmp.name
        try:
            pl.DataFrame({"A": [1, 2], "B": [10.5, 20.5]}).write_parquet(path)
            lazy = ppd.scan_parquet(path, dtype={"A": "float64"})
            df = lazy.collect()
            assert df._df.dtypes[0] == pl.Float64
            assert df.to_dict()["A"] == [1.0, 2.0]
        finally:
            os.unlink(path)

    def test_unimplemented_readers_raise(self):
        with pytest.raises(NotImplementedError):
            ppd.read_excel("dummy.xlsx")
        with pytest.raises(NotImplementedError):
            ppd.read_excel("dummy.xlsx", engine="openpyxl")
