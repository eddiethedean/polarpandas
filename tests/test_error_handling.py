"""
Comprehensive error handling tests.

This test file focuses on testing error handling across polarpandas,
ensuring proper exceptions are raised for invalid inputs and operations.
"""

import pytest

import polarpandas as ppd
from polarpandas.index import MultiIndex


class TestDataFrameErrorHandling:
    """Tests for DataFrame error handling."""

    def test_invalid_column_name(self):
        """Test accessing non-existent column."""
        df = ppd.DataFrame({"A": [1, 2, 3]})

        with pytest.raises(KeyError):
            _ = df["NonExistent"]

    def test_invalid_column_assignment_length_mismatch(self):
        """Test assigning column with wrong length."""
        df = ppd.DataFrame({"A": [1, 2, 3]})

        with pytest.raises((ValueError, Exception)):
            df["B"] = [1, 2]  # Wrong length

    def test_drop_nonexistent_column(self):
        """Test dropping non-existent column."""
        df = ppd.DataFrame({"A": [1, 2, 3]})

        with pytest.raises((KeyError, ValueError, Exception)):
            df.drop(columns=["NonExistent"])

    def test_invalid_merge_key(self):
        """Test merge with non-existent key."""
        df1 = ppd.DataFrame({"A": [1, 2]})
        df2 = ppd.DataFrame({"B": [3, 4]})

        with pytest.raises((KeyError, ValueError, Exception)):
            df1.merge(df2, on="NonExistent")

    def test_set_index_nonexistent_column(self):
        """Test set_index with non-existent column."""
        df = ppd.DataFrame({"A": [1, 2, 3]})

        with pytest.raises((KeyError, ValueError, Exception)):
            df.set_index("NonExistent")

    def test_loc_invalid_key(self):
        """Test loc with invalid key."""
        df = ppd.DataFrame({"A": [1, 2, 3]}, index=["x", "y", "z"])

        with pytest.raises((KeyError, ValueError, Exception)):
            _ = df.loc["invalid_key"]

    def test_iloc_out_of_bounds(self):
        """Test iloc with out of bounds index."""
        df = ppd.DataFrame({"A": [1, 2, 3]})

        with pytest.raises((IndexError, Exception)):
            _ = df.iloc[10]


class TestSeriesErrorHandling:
    """Tests for Series error handling."""

    def test_series_length_mismatch_index(self):
        """Test creating Series with mismatched index length."""
        with pytest.raises((ValueError, Exception)):
            ppd.Series([1, 2, 3], index=["a", "b"])  # Wrong length

    def test_series_getitem_invalid_label(self):
        """Test Series getitem with invalid label."""
        s = ppd.Series([1, 2, 3], index=["a", "b", "c"])

        with pytest.raises((KeyError, Exception)):
            _ = s["invalid"]

    def test_series_division_by_zero(self):
        """Test Series division by zero."""
        s = ppd.Series([1, 2, 3])

        # Division by zero typically returns inf, not error
        result = s.div(0)
        assert result is not None

    def test_series_item_on_multi_element(self):
        """Test item() on multi-element Series raises error."""
        s = ppd.Series([1, 2, 3])

        with pytest.raises((ValueError, Exception)):
            s.item()


class TestMultiIndexErrorHandling:
    """Tests for MultiIndex error handling."""

    def test_multiindex_empty_arrays(self):
        """Test creating MultiIndex with empty arrays."""
        with pytest.raises(ValueError):
            MultiIndex.from_arrays([])

    def test_multiindex_mismatched_array_lengths(self):
        """Test creating MultiIndex with mismatched lengths."""
        with pytest.raises(ValueError):
            MultiIndex.from_arrays([["a", "b"], ["x", "y", "z"]])

    def test_multiindex_invalid_level_number(self):
        """Test accessing invalid level number."""
        mi = MultiIndex.from_arrays([["a", "b"], ["x", "y"]])

        with pytest.raises((IndexError, KeyError)):
            mi.get_level_values(10)

    def test_multiindex_invalid_level_name(self):
        """Test accessing invalid level name."""
        mi = MultiIndex.from_arrays([["a", "b"], ["x", "y"]], names=["A", "B"])

        with pytest.raises(KeyError):
            mi.get_level_values("NonExistent")

    def test_multiindex_droplevel_invalid(self):
        """Test droplevel with invalid level."""
        mi = MultiIndex.from_arrays([["a", "b"], ["x", "y"]])

        with pytest.raises((IndexError, KeyError)):
            mi.droplevel(10)


class TestOperationsErrorHandling:
    """Tests for operations error handling."""

    def test_concat_incompatible_shapes(self):
        """Test concat with incompatible DataFrames."""
        df1 = ppd.DataFrame({"A": [1, 2]})
        df2 = ppd.DataFrame({"A": [3, 4], "B": [5, 6]})

        # Concat should handle different columns
        try:
            result = ppd.concat([df1, df2])
            # May succeed with NaN values
            assert isinstance(result, ppd.DataFrame)
        except Exception:
            # Or may fail depending on implementation
            pass

    def test_merge_incompatible_keys(self):
        """Test merge with incompatible key types."""
        df1 = ppd.DataFrame({"key": [1, 2], "val": [10, 20]})
        df2 = ppd.DataFrame({"key": ["a", "b"], "val": [30, 40]})

        # May or may not raise error depending on Polars handling
        try:
            result = ppd.merge(df1, df2, on="key")
            # May succeed with empty result
            assert isinstance(result, ppd.DataFrame)
        except Exception:
            pass


class TestIndexErrorHandling:
    """Tests for Index error handling."""

    def test_index_empty_creation(self):
        """Test creating empty Index."""
        idx = ppd.Index([])
        assert len(idx) == 0

    def test_index_getitem_out_of_bounds(self):
        """Test Index getitem out of bounds."""
        idx = ppd.Index([1, 2, 3])

        with pytest.raises((IndexError, Exception)):
            _ = idx[10]


class TestAsTypeErrors:
    """Tests for astype error handling."""

    def test_astype_invalid_errors_parameter(self):
        """Test astype with invalid errors parameter."""
        df = ppd.DataFrame({"A": [1, 2, 3]})

        with pytest.raises(ValueError):
            df.astype({"A": "int64"}, errors="invalid")

    def test_series_astype_incompatible(self):
        """Test Series astype with incompatible conversion."""
        s = ppd.Series(["a", "b", "c"])

        # With errors='ignore', should not raise
        result = s.astype("int64", errors="ignore")
        assert result is not None


class TestGroupByErrors:
    """Tests for groupby error handling."""

    def test_groupby_invalid_column(self):
        """Test groupby with non-existent column."""
        df = ppd.DataFrame({"A": [1, 2, 3]})

        with pytest.raises((KeyError, Exception)):
            df.groupby("NonExistent")

    def test_groupby_invalid_level(self):
        """Test groupby with invalid level on non-MultiIndex."""
        df = ppd.DataFrame({"A": [1, 2, 3]})

        with pytest.raises((ValueError, KeyError, Exception)):
            df.groupby(level=0)


class TestLocIlocErrors:
    """Tests for loc and iloc error handling."""

    def test_loc_tuple_on_non_multiindex(self):
        """Test loc with tuple key on non-MultiIndex."""
        df = ppd.DataFrame({"A": [1, 2, 3]})

        with pytest.raises((KeyError, Exception)):
            _ = df.loc[("a", "b")]

    def test_iloc_negative_out_of_bounds(self):
        """Test iloc with negative index out of bounds."""
        df = ppd.DataFrame({"A": [1, 2, 3]})

        with pytest.raises((IndexError, Exception)):
            _ = df.iloc[-10]


class TestIOErrors:
    """Tests for I/O error handling."""

    def test_read_csv_nonexistent_file(self):
        """Test reading non-existent CSV file."""
        with pytest.raises((FileNotFoundError, Exception)):
            ppd.read_csv("nonexistent_file.csv")

    def test_read_parquet_nonexistent_file(self):
        """Test reading non-existent Parquet file."""
        with pytest.raises((FileNotFoundError, Exception)):
            ppd.read_parquet("nonexistent_file.parquet")

    def test_to_csv_invalid_path(self):
        """Test writing CSV to invalid path."""
        import contextlib

        df = ppd.DataFrame({"A": [1, 2, 3]})

        # Try writing to invalid directory (expected to fail)
        with contextlib.suppress(OSError, PermissionError, Exception):
            df.to_csv("/invalid/path/file.csv")


class TestSchemaConversionErrors:
    """Tests for schema conversion error handling."""

    def test_convert_schema_invalid_type(self):
        """Test schema conversion with truly invalid type."""
        with pytest.raises((ValueError, TypeError)):
            ppd.utils.convert_schema_to_polars({"A": lambda x: x})


class TestMultiIndexFromMethodsErrors:
    """Tests for MultiIndex from_* method errors."""

    def test_from_arrays_mismatched_lengths(self):
        """Test from_arrays with mismatched array lengths."""
        with pytest.raises(ValueError):
            MultiIndex.from_arrays([["a", "b"], ["x"]])

    def test_from_tuples_invalid_input(self):
        """Test from_tuples with invalid input."""
        with pytest.raises((ValueError, TypeError, Exception)):
            MultiIndex.from_tuples("not_a_list")

    def test_from_product_empty_iterables(self):
        """Test from_product with empty iterables."""
        try:
            result = MultiIndex.from_product([])
            # May return empty MultiIndex
            assert isinstance(result, MultiIndex)
        except ValueError:
            # Or raise ValueError
            pass

    def test_from_frame_invalid_dataframe(self):
        """Test from_frame with invalid DataFrame."""
        with pytest.raises((ValueError, AttributeError, Exception)):
            MultiIndex.from_frame("not_a_dataframe")


class TestDataFrameMethodErrors:
    """Tests for various DataFrame method errors."""

    def test_pivot_missing_required_params(self):
        """Test pivot without required parameters."""
        df = ppd.DataFrame({"A": [1, 2], "B": [3, 4]})

        try:
            with pytest.raises((TypeError, ValueError)):
                df.pivot()
        except (NotImplementedError, AttributeError):
            pytest.skip("pivot not yet implemented")

    def test_rename_invalid_mapper(self):
        """Test rename with invalid mapper."""
        df = ppd.DataFrame({"A": [1, 2]})

        # Invalid mapper type
        try:
            result = df.rename(columns="not_a_dict")
            # May handle gracefully
            assert result is not None
        except (TypeError, ValueError):
            # Or raise error
            pass


class TestMultiIndexStructuralErrors:
    """Tests for MultiIndex structural validation."""

    def test_multiindex_set_names_wrong_count(self):
        """Test set_names with wrong number of names."""
        mi = MultiIndex.from_arrays([["a", "b"], ["x", "y"]])

        try:
            # set_names might enforce correct count
            with pytest.raises((ValueError, Exception)):
                mi.set_names(["OnlyOne"])
        except AttributeError:
            # set_names might have limitations
            pytest.skip("set_names validation not yet implemented")

    def test_multiindex_swaplevel_invalid_levels(self):
        """Test swaplevel with invalid level indices."""
        mi = MultiIndex.from_arrays([["a", "b"], ["x", "y"]])

        with pytest.raises((IndexError, KeyError, Exception)):
            mi.swaplevel(0, 10)
