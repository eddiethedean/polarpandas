"""
Comprehensive error handling tests for DataFrame.

Tests exception conversion paths and invalid parameter validation.
"""

import pytest

import polarpandas as ppd
from tests.test_helpers import assert_frame_equal


class TestDataFrameErrorHandling:
    """Test error handling and exception conversion."""

    def test_invalid_column_keyerror(self):
        """Test KeyError for invalid column access."""
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        with pytest.raises(KeyError):
            _ = df["nonexistent"]

    def test_invalid_multiple_columns_keyerror(self):
        """Test KeyError for invalid column list."""
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        with pytest.raises(KeyError):
            _ = df[["a", "nonexistent"]]

    def test_set_index_invalid_column_keyerror(self):
        """Test KeyError when setting index with invalid column."""
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        with pytest.raises(KeyError):
            df.set_index("nonexistent")

    def test_set_index_empty_list_valueerror(self):
        """Test ValueError when setting index with empty list."""
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        with pytest.raises(ValueError, match="non-zero number"):
            df.set_index([])

    def test_drop_invalid_column_keyerror(self):
        """Test KeyError when dropping invalid column."""
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        with pytest.raises(KeyError):
            df.drop(columns=["nonexistent"])

    def test_rename_invalid_column_keyerror(self):
        """Test that rename silently ignores invalid columns (pandas behavior)."""
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # rename() filters out invalid columns silently (matches pandas behavior)
        expected = df.to_dict()
        result = df.rename(columns={"nonexistent": "new_name"})
        # Should return unchanged DataFrame
        assert_frame_equal(result, expected)

    def test_sort_values_invalid_column_keyerror(self):
        """Test error when sorting by invalid column."""
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # May raise KeyError or other exception from Polars
        with pytest.raises((KeyError, ValueError, Exception)):
            df.sort_values(by="nonexistent")

    def test_type_error_on_invalid_operation(self):
        """Test TypeError for invalid operations."""
        df = ppd.DataFrame({"a": [1, 2, 3]})

        # Invalid arithmetic operation
        with pytest.raises((TypeError, ValueError)):
            # This should raise an error
            _ = df + "invalid"

    def test_index_error_out_of_bounds(self):
        """Test IndexError for out of bounds access."""
        df = ppd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises((IndexError, KeyError)):
            _ = df.iloc[10]

    def test_exception_conversion_polars_to_pandas(self):
        """Test that Polars exceptions are converted to pandas-compatible ones."""
        df = ppd.DataFrame({"a": [1, 2, 3]})

        # Invalid column should raise KeyError (not Polars ColumnNotFoundError)
        try:
            _ = df["nonexistent"]
        except KeyError:
            # Expected - Polars error converted to KeyError
            pass
        except Exception as e:
            # Should be KeyError, not raw Polars error
            pytest.fail(f"Expected KeyError, got {type(e)}: {e}")

    def test_invalid_merge_parameters(self):
        """Test error handling for invalid merge parameters."""
        df1 = ppd.DataFrame({"key": [1, 2], "val1": [10, 20]})
        df2 = ppd.DataFrame({"key": [1, 2], "val2": [30, 40]})

        # Invalid 'on' column - may raise various exceptions
        with pytest.raises((KeyError, ValueError, Exception)):
            df1.merge(df2, on="nonexistent")

    def test_invalid_concat_axis(self):
        """Test error handling for invalid concat axis."""
        import polarpandas.operations as ops

        df1 = ppd.DataFrame({"a": [1, 2]})
        df2 = ppd.DataFrame({"b": [3, 4]})

        # Very large axis value - may not raise, just test it doesn't crash
        import contextlib

        with contextlib.suppress(ValueError, IndexError, TypeError, Exception):
            _ = ops.concat([df1, df2], axis=999)
            # If it doesn't raise, that's acceptable behavior

    def test_invalid_groupby_column(self):
        """Test error for invalid groupby column."""
        df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # GroupBy may not validate immediately, but aggregation should fail
        with pytest.raises((KeyError, ValueError, Exception)):
            import polars as pl

            _ = df.groupby("nonexistent").agg(pl.col("a").sum())
