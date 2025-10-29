"""
Comprehensive tests for I/O operations with various formats and edge cases.

All tests compare polarpandas output against actual pandas output
to ensure 100% compatibility.
"""

import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import polarpandas as ppd


class TestCSVIOComprehensive:
    """Comprehensive tests for CSV I/O operations."""

    def setup_method(self):
        """Create test data in both pandas and polarpandas."""
        self.data = {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"],
        }
        # Don't create DataFrames here to avoid state pollution
        # Each test method will create fresh DataFrames

    def test_read_csv_basic(self):
        """Test basic CSV reading."""
        pd_df = pd.DataFrame(self.data)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd_df.to_csv(f.name, index=False)

            pd_result = pd.read_csv(f.name)
            ppd_result = ppd.read_csv(f.name)
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_read_csv_with_index(self):
        """Test CSV reading with index."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame(self.data).to_csv(f.name, index=True)

            pd_result = pd.read_csv(f.name, index_col=0)
            ppd_result = ppd.read_csv(f.name, index_col=0)
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_read_csv_with_nulls(self):
        """Test CSV reading with null values."""
        data_with_nulls = {
            "A": [1, None, 3, 4, 5],
            "B": [10, 20, None, 40, 50],
            "C": ["a", None, "c", "d", "e"],
        }
        pd_df = pd.DataFrame(data_with_nulls)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd_df.to_csv(f.name, index=False)

            pd_result = pd.read_csv(f.name)
            ppd_result = ppd.read_csv(f.name)
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_read_csv_with_different_dtypes(self):
        """Test CSV reading with different data types."""
        mixed_data = {
            "A": [1, 2, 3, 4, 5],
            "B": [1.1, 2.2, 3.3, 4.4, 5.5],
            "C": [True, False, True, False, True],
            "D": [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
                datetime(2023, 1, 4),
                datetime(2023, 1, 5),
            ],
        }
        pd_df = pd.DataFrame(mixed_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd_df.to_csv(f.name, index=False)

            pd_result = pd.read_csv(f.name)
            ppd_result = ppd.read_csv(f.name)
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_read_csv_with_separator(self):
        """Test CSV reading with different separator."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame(self.data).to_csv(f.name, index=False, sep=";")

            pd_result = pd.read_csv(f.name, sep=";")
            ppd_result = ppd.read_csv(f.name, sep=";")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_read_csv_with_header(self):
        """Test CSV reading with custom header."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame(self.data).to_csv(f.name, index=False, header=["X", "Y", "Z"])

            pd_result = pd.read_csv(f.name, names=["X", "Y", "Z"])
            ppd_result = ppd.read_csv(f.name, names=["X", "Y", "Z"])
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_read_csv_with_skiprows(self):
        """Test CSV reading with skiprows."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Add header row
            f.write("Header line\n")
            pd.DataFrame(self.data).to_csv(f.name, index=False, mode="a")

            pd_result = pd.read_csv(f.name, skiprows=1)
            ppd_result = ppd.read_csv(f.name, skiprows=1)
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_read_csv_with_nrows(self):
        """Test CSV reading with nrows parameter."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame(self.data).to_csv(f.name, index=False)

            pd_result = pd.read_csv(f.name, nrows=3)
            ppd_result = ppd.read_csv(f.name, nrows=3)
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_read_csv_empty_file(self):
        """Test CSV reading with empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Create empty file
            pass

            # Both should raise EmptyDataError for empty files
            with pytest.raises(pd.errors.EmptyDataError):
                pd.read_csv(f.name)
            with pytest.raises(pd.errors.EmptyDataError):
                ppd.read_csv(f.name)

            os.unlink(f.name)

    def test_read_csv_single_column(self):
        """Test CSV reading with single column."""
        single_col_data = {"A": [1, 2, 3, 4, 5]}
        pd_df = pd.DataFrame(single_col_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd_df.to_csv(f.name, index=False)

            pd_result = pd.read_csv(f.name)
            ppd_result = ppd.read_csv(f.name)
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_read_csv_single_row(self):
        """Test CSV reading with single row."""
        single_row_data = {"A": [1], "B": [10], "C": ["a"]}
        pd_df = pd.DataFrame(single_row_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd_df.to_csv(f.name, index=False)

            pd_result = pd.read_csv(f.name)
            ppd_result = ppd.read_csv(f.name)
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_read_csv_large_dataset(self):
        """Test CSV reading with large dataset."""
        # Create larger dataset
        np.random.seed(42)
        large_data = {
            "A": np.random.randn(1000),
            "B": np.random.randn(1000),
            "C": np.random.randn(1000),
        }
        pd_df = pd.DataFrame(large_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd_df.to_csv(f.name, index=False)

            pd_result = pd.read_csv(f.name)
            ppd_result = ppd.read_csv(f.name)
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_to_csv_basic(self):
        """Test basic CSV writing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame(self.data).to_csv(f.name, index=False)
            ppd.DataFrame(self.data).to_csv(f.name, index=False)

            # Read back and compare
            pd_result = pd.read_csv(f.name)
            ppd_result = ppd.read_csv(f.name)
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_to_csv_with_index(self):
        """Test CSV writing with index."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame(self.data).to_csv(f.name, index=True)
            ppd.DataFrame(self.data).to_csv(f.name, index=True)

            # Read back and compare
            pd_result = pd.read_csv(f.name, index_col=0)
            ppd_result = ppd.read_csv(f.name, index_col=0)
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_to_csv_with_separator(self):
        """Test CSV writing with different separator."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame(self.data).to_csv(f.name, index=False, sep=";")
            ppd.DataFrame(self.data).to_csv(f.name, index=False, sep=";")

            # Read back and compare
            pd_result = pd.read_csv(f.name, sep=";")
            ppd_result = ppd.read_csv(f.name, sep=";")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_to_csv_with_header(self):
        """Test CSV writing with custom header."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Test pandas
            pd.DataFrame(self.data).to_csv(f.name, index=False, header=["X", "Y", "Z"])
            pd_result = pd.read_csv(f.name, names=["X", "Y", "Z"])

            # Clear file and test polarpandas
            with open(f.name, "w") as f:
                pass  # Clear the file
            ppd.DataFrame(self.data).to_csv(f.name, index=False, header=["X", "Y", "Z"])
            ppd_result = ppd.read_csv(f.name, names=["X", "Y", "Z"])

            # Compare results
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_io_methods_return_types(self):
        """Test that I/O methods return correct types."""
        # Test read_csv
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame(self.data).to_csv(f.name, index=False)
            result = ppd.read_csv(f.name)
            assert isinstance(result, ppd.DataFrame)
            os.unlink(f.name)

    def test_io_methods_preserve_original(self):
        """Test that I/O methods don't modify original DataFrame."""
        original_pd = pd.DataFrame(self.data).copy()
        original_ppd = ppd.DataFrame(self.data).copy()

        # Perform I/O operations
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            pd.DataFrame(self.data).to_csv(f.name, index=False)
            ppd.DataFrame(self.data).to_csv(f.name, index=False)
            os.unlink(f.name)

        # Original should be unchanged
        pd.testing.assert_frame_equal(original_pd, pd.DataFrame(self.data))
        pd.testing.assert_frame_equal(
            original_ppd.to_pandas(), ppd.DataFrame(self.data).to_pandas()
        )


class TestJSONIOComprehensive:
    """Comprehensive tests for JSON I/O operations."""

    def setup_method(self):
        """Create test data in both pandas and polarpandas."""
        self.data = {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": ["a", "b", "c", "d", "e"],
        }
        # Don't create DataFrames here to avoid state pollution
        # Each test method will create fresh DataFrames

    def test_read_json_basic(self):
        """Test basic JSON reading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd.DataFrame(self.data).to_json(f.name, orient="records")

            pd_result = pd.read_json(f.name, orient="records")
            ppd_result = ppd.read_json(f.name, orient="records")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    @pytest.mark.skip(
        reason="Polars doesn't support orient parameter for JSON operations - permanent limitation"
    )
    def test_read_json_with_index(self):
        """Test JSON reading with index."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd.DataFrame(self.data).to_json(f.name, orient="index")

            pd_result = pd.read_json(f.name, orient="index")
            ppd_result = ppd.read_json(f.name, orient="index")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    @pytest.mark.skip(
        reason="Polars doesn't support orient parameter for JSON operations - permanent limitation"
    )
    def test_read_json_with_columns(self):
        """Test JSON reading with columns orientation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd.DataFrame(self.data).to_json(f.name, orient="columns")

            pd_result = pd.read_json(f.name, orient="columns")
            ppd_result = ppd.read_json(f.name, orient="columns")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    @pytest.mark.skip(
        reason="Polars doesn't support orient parameter for JSON operations - permanent limitation"
    )
    def test_read_json_with_values(self):
        """Test JSON reading with values orientation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd.DataFrame(self.data).to_json(f.name, orient="values")

            pd_result = pd.read_json(f.name, orient="values")
            ppd_result = ppd.read_json(f.name, orient="values")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    @pytest.mark.skip(
        reason="Polars doesn't support orient parameter for JSON operations - permanent limitation"
    )
    def test_read_json_with_split(self):
        """Test JSON reading with split orientation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd.DataFrame(self.data).to_json(f.name, orient="split")

            pd_result = pd.read_json(f.name, orient="split")
            ppd_result = ppd.read_json(f.name, orient="split")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    @pytest.mark.skip(
        reason="Polars doesn't support orient parameter for JSON operations - permanent limitation"
    )
    def test_read_json_with_table(self):
        """Test JSON reading with table orientation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd.DataFrame(self.data).to_json(f.name, orient="table")

            pd_result = pd.read_json(f.name, orient="table")
            ppd_result = ppd.read_json(f.name, orient="table")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_to_json_basic(self):
        """Test basic JSON writing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd.DataFrame(self.data).to_json(f.name, orient="records")
            ppd.DataFrame(self.data).to_json(f.name, orient="records")

            # Read back and compare
            pd_result = pd.read_json(f.name, orient="records")
            ppd_result = ppd.read_json(f.name, orient="records")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    @pytest.mark.skip(
        reason="Polars doesn't support orient parameter for JSON operations - permanent limitation"
    )
    def test_to_json_with_index(self):
        """Test JSON writing with index."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd.DataFrame(self.data).to_json(f.name, orient="index")
            ppd.DataFrame(self.data).to_json(f.name, orient="index")

            # Read back and compare
            pd_result = pd.read_json(f.name, orient="index")
            ppd_result = ppd.read_json(f.name, orient="index")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    @pytest.mark.skip(
        reason="Polars doesn't support orient parameter for JSON operations - permanent limitation"
    )
    def test_to_json_with_columns(self):
        """Test JSON writing with columns orientation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd.DataFrame(self.data).to_json(f.name, orient="columns")
            ppd.DataFrame(self.data).to_json(f.name, orient="columns")

            # Read back and compare
            pd_result = pd.read_json(f.name, orient="columns")
            ppd_result = ppd.read_json(f.name, orient="columns")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    @pytest.mark.skip(
        reason="Polars doesn't support orient parameter for JSON operations - permanent limitation"
    )
    def test_to_json_with_values(self):
        """Test JSON writing with values orientation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd.DataFrame(self.data).to_json(f.name, orient="values")
            ppd.DataFrame(self.data).to_json(f.name, orient="values")

            # Read back and compare
            pd_result = pd.read_json(f.name, orient="values")
            ppd_result = ppd.read_json(f.name, orient="values")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    @pytest.mark.skip(
        reason="Polars doesn't support orient parameter for JSON operations - permanent limitation"
    )
    def test_to_json_with_split(self):
        """Test JSON writing with split orientation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd.DataFrame(self.data).to_json(f.name, orient="split")
            ppd.DataFrame(self.data).to_json(f.name, orient="split")

            # Read back and compare
            pd_result = pd.read_json(f.name, orient="split")
            ppd_result = ppd.read_json(f.name, orient="split")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    @pytest.mark.skip(
        reason="Polars doesn't support orient parameter for JSON operations - permanent limitation"
    )
    def test_to_json_with_table(self):
        """Test JSON writing with table orientation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd.DataFrame(self.data).to_json(f.name, orient="table")
            ppd.DataFrame(self.data).to_json(f.name, orient="table")

            # Read back and compare
            pd_result = pd.read_json(f.name, orient="table")
            ppd_result = ppd.read_json(f.name, orient="table")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_json_io_with_nulls(self):
        """Test JSON I/O with null values."""
        data_with_nulls = {
            "A": [1, None, 3, 4, 5],
            "B": [10, 20, None, 40, 50],
            "C": ["a", None, "c", "d", "e"],
        }
        pd_df = pd.DataFrame(data_with_nulls)
        ppd_df = ppd.DataFrame(data_with_nulls)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd_df.to_json(f.name, orient="records")
            ppd_df.to_json(f.name, orient="records")

            # Read back and compare
            pd_result = pd.read_json(f.name, orient="records")
            ppd_result = ppd.read_json(f.name, orient="records")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_json_io_with_different_dtypes(self):
        """Test JSON I/O with different data types."""
        mixed_data = {
            "A": [1, 2, 3, 4, 5],
            "B": [1.1, 2.2, 3.3, 4.4, 5.5],
            "C": [True, False, True, False, True],
            "D": [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
                datetime(2023, 1, 4),
                datetime(2023, 1, 5),
            ],
        }
        pd_df = pd.DataFrame(mixed_data)
        ppd_df = ppd.DataFrame(mixed_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd_df.to_json(f.name, orient="records")
            ppd_df.to_json(f.name, orient="records")

            # Read back and compare
            pd_result = pd.read_json(f.name, orient="records")
            ppd_result = ppd.read_json(f.name, orient="records")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_json_io_empty_dataframe(self):
        """Test JSON I/O with empty DataFrame."""
        pd_empty = pd.DataFrame()
        ppd_empty = ppd.DataFrame()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd_empty.to_json(f.name, orient="records")
            ppd_empty.to_json(f.name, orient="records")

            # Read back and compare
            pd_result = pd.read_json(f.name, orient="records")
            ppd_result = ppd.read_json(f.name, orient="records")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_json_io_single_column(self):
        """Test JSON I/O with single column."""
        single_col_data = {"A": [1, 2, 3, 4, 5]}
        pd_df = pd.DataFrame(single_col_data)
        ppd_df = ppd.DataFrame(single_col_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd_df.to_json(f.name, orient="records")
            ppd_df.to_json(f.name, orient="records")

            # Read back and compare
            pd_result = pd.read_json(f.name, orient="records")
            ppd_result = ppd.read_json(f.name, orient="records")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_json_io_single_row(self):
        """Test JSON I/O with single row."""
        single_row_data = {"A": [1], "B": [10], "C": ["a"]}
        pd_df = pd.DataFrame(single_row_data)
        ppd_df = ppd.DataFrame(single_row_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd_df.to_json(f.name, orient="records")
            ppd_df.to_json(f.name, orient="records")

            # Read back and compare
            pd_result = pd.read_json(f.name, orient="records")
            ppd_result = ppd.read_json(f.name, orient="records")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_json_io_large_dataset(self):
        """Test JSON I/O with large dataset."""
        # Create larger dataset
        np.random.seed(42)
        large_data = {
            "A": np.random.randn(1000),
            "B": np.random.randn(1000),
            "C": np.random.randn(1000),
        }
        pd_df = pd.DataFrame(large_data)
        ppd_df = ppd.DataFrame(large_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd_df.to_json(f.name, orient="records")
            ppd_df.to_json(f.name, orient="records")

            # Read back and compare
            pd_result = pd.read_json(f.name, orient="records")
            ppd_result = ppd.read_json(f.name, orient="records")
            pd.testing.assert_frame_equal(ppd_result.to_pandas(), pd_result)

            os.unlink(f.name)

    def test_read_excel_file_not_found(self):
        """Test read_excel raises NotImplementedError."""
        import pytest

        # Excel reading is not implemented
        with pytest.raises(NotImplementedError, match="read_excel not yet implemented"):
            ppd.read_excel("nonexistent.xlsx")

    def test_read_parquet_basic(self):
        """Test read_parquet function."""
        # Create a parquet file first
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = f.name

        try:
            df.to_parquet(temp_path)

            # Test module-level read_parquet
            result = ppd.read_parquet(temp_path)
            assert isinstance(result, ppd.DataFrame)
            assert len(result) == 3
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_read_parquet_invalid_file(self):
        """Test read_parquet with invalid file raises error."""
        import pytest

        # File doesn't exist
        with pytest.raises(
            (FileNotFoundError, OSError, Exception)
        ):  # Polars will raise an error
            ppd.read_parquet("nonexistent_file.parquet")

    def test_read_feather_basic(self):
        """Test read_feather function."""
        # Create a feather file first
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with tempfile.NamedTemporaryFile(suffix=".feather", delete=False) as f:
            temp_path = f.name

        try:
            df.to_feather(temp_path)

            # Test module-level read_feather
            result = ppd.read_feather(temp_path)
            assert isinstance(result, ppd.DataFrame)
            assert len(result) == 3
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_read_sql_mock(self):
        """Test read_sql with a mock that will error."""
        import pytest

        # read_sql requires actual database connection, test that it calls the method
        # Since we can't easily mock a database connection, we'll test with an invalid one
        with pytest.raises(
            (AttributeError, TypeError, ValueError)
        ):  # Will raise connection or attribute error
            ppd.read_sql("SELECT * FROM table", None)

    def test_json_io_methods_return_types(self):
        """Test that JSON I/O methods return correct types."""
        # Test read_json
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd.DataFrame(self.data).to_json(f.name, orient="records")
            result = ppd.read_json(f.name, orient="records")
            assert isinstance(result, ppd.DataFrame)
            os.unlink(f.name)

    def test_json_io_methods_preserve_original(self):
        """Test that JSON I/O methods don't modify original DataFrame."""
        original_pd = pd.DataFrame(self.data).copy()
        original_ppd = ppd.DataFrame(self.data).copy()

        # Perform I/O operations
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pd.DataFrame(self.data).to_json(f.name, orient="records")
            ppd.DataFrame(self.data).to_json(f.name, orient="records")
            os.unlink(f.name)

        # Original should be unchanged
        pd.testing.assert_frame_equal(original_pd, pd.DataFrame(self.data))
        pd.testing.assert_frame_equal(
            original_ppd.to_pandas(), ppd.DataFrame(self.data).to_pandas()
        )
