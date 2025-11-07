"""
Test Index functionality.
"""

import polars as pl

import polarpandas as ppd
from polarpandas import Index
from polarpandas._index_manager import IndexManager


class TestIndexInitialization:
    """Test Index initialization from various sources."""

    def test_init_from_list(self):
        """Test creating Index from list."""
        data = [0, 1, 2, 3, 4]
        idx = Index(data)
        assert isinstance(idx, Index)
        assert hasattr(idx, "_series")
        assert isinstance(idx._series, pl.Series)

    def test_init_from_polars_series(self):
        """Test creating Index from existing Polars Series."""
        pl_series = pl.Series("index", [0, 1, 2, 3, 4])
        idx = Index(pl_series)
        assert isinstance(idx, Index)
        assert isinstance(idx._series, pl.Series)

    def test_init_empty(self):
        """Test creating empty Index."""
        idx = Index([])
        assert isinstance(idx, Index)
        assert isinstance(idx._series, pl.Series)
        assert len(idx) == 0

    def test_init_none(self):
        """Test creating Index with None data."""
        # Line 32: data is None case
        idx = Index(None)
        assert isinstance(idx, Index)
        assert isinstance(idx._series, pl.Series)
        assert len(idx) == 0


class TestIndexManagerUtilities:
    """Tests for internal IndexManager helper methods."""

    def test_preserve_index_copies_index_metadata(self):
        source = ppd.DataFrame({"a": [1, 2]}, index=["row-1", "row-2"])
        source._index_name = "row_label"

        result_pl = source._df.select(["a"])
        preserved = IndexManager.preserve_index(source, result_pl)

        assert preserved._index == source._index
        assert preserved._index_name == "row_label"

    def test_preserve_index_without_name(self):
        source = ppd.DataFrame({"a": [1, 2]}, index=["x", "y"])
        source._index_name = "row_label"

        result_pl = source._df.with_columns((pl.col("a") * 2).alias("a"))
        preserved = IndexManager.preserve_index(source, result_pl, preserve_name=False)

        assert preserved._index == source._index
        assert preserved._index_name is None

    def test_preserve_index_inplace_resets_mismatched_length(self):
        df = ppd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
        filtered = df._df.filter(pl.col("a") > 2)

        IndexManager.preserve_index_inplace(df, filtered)
        assert df._index is None
        assert df._index_name is None

    def test_extract_index_for_rows_bounds_clipped(self):
        df = ppd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])

        extracted = IndexManager.extract_index_for_rows(df, [0, 2, 10])
        assert extracted == ["x", "z"]

    def test_create_index_from_columns_multi_level(self):
        polars_df = pl.DataFrame({"city": ["NY", "SF"], "year": [2020, 2021]})

        index_values, index_name = IndexManager.create_index_from_columns(
            polars_df, ["city", "year"]
        )
        assert index_values == [("NY", 2020), ("SF", 2021)]
        assert index_name is None

    def test_validate_index_length_mismatch(self):
        assert IndexManager.validate_index_length(["x"], data_length=2) is False


class TestIndexDelegation:
    """Test that Index properly delegates to underlying Polars Series."""

    def test_len(self):
        """Test len() function."""
        idx = Index([0, 1, 2, 3, 4])
        assert len(idx) == 5

    def test_access_dtype(self):
        """Test accessing dtype attribute."""
        idx = Index([0, 1, 2, 3])
        dtype = idx.dtype
        assert dtype is not None

    def test_access_private_attribute_raises_error(self):
        """Test accessing private attribute raises AttributeError."""
        # Line 47: private attribute starting with _ raises error
        import pytest

        idx = Index([0, 1, 2, 3])
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = idx._private_attr

    def test_access_nonexistent_attribute_raises_error(self):
        """Test accessing nonexistent attribute raises AttributeError."""
        # Lines 54-55: AttributeError handling
        import pytest

        idx = Index([0, 1, 2, 3])
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = idx.nonexistent_method()


class TestIndexProperties:
    """Test Index properties."""

    def test_shape_property(self):
        """Test shape property."""
        idx = Index([0, 1, 2, 3, 4])
        shape = idx.shape
        assert shape == (5,)

    def test_size_property(self):
        """Test size property."""
        idx = Index([0, 1, 2, 3, 4])
        size = idx.size
        assert size == 5


class TestIndexRepresentation:
    """Test Index string representations."""

    def test_repr(self):
        """Test __repr__ returns a string."""
        idx = Index([0, 1, 2, 3, 4])
        repr_str = repr(idx)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0

    def test_str(self):
        """Test __str__ returns a string."""
        idx = Index([0, 1, 2, 3, 4])
        str_repr = str(idx)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0

    def test_tolist(self):
        """Test tolist() method."""
        # Line 77: tolist() method
        idx = Index([0, 1, 2, 3, 4])
        result = idx.tolist()
        assert isinstance(result, list)
        assert result == [0, 1, 2, 3, 4]

    def test_index_with_nulls(self):
        """Test Index with null values."""
        idx = Index([0, None, 2, None, 4])
        assert len(idx) == 5
        # Should handle nulls gracefully
        assert idx._series.null_count() == 2

    def test_index_comparison_operations(self):
        """Test Index comparison operations."""
        idx1 = Index([1, 2, 3])
        idx2 = Index([1, 2, 4])

        # Test that comparison operations work through delegation
        # These will be handled by the underlying Series
        result = idx1._series == idx2._series
        assert isinstance(result, pl.Series)
        assert result.to_list() == [True, True, False]

    def test_index_slice_edge_cases(self):
        """Test Index slicing edge cases."""
        idx = Index([0, 1, 2, 3, 4])

        # Test iteration (line 73 covered)
        values = list(idx)
        assert values == [0, 1, 2, 3, 4]

        # Test negative indices (if supported)
        # This is handled by underlying Series slicing


class TestDataFrameSeriesInterop:
    """Test interoperability between DataFrame and Series."""

    def test_dataframe_column_returns_series(self):
        """Test that accessing a DataFrame column returns a Series (will be implemented later)."""
        # This will be tested more thoroughly once we implement __getitem__ for DataFrame
        pass


# ============================================================================
# Boolean/Logical Operations
# ============================================================================


class TestIndexBooleanOperations:
    """Test boolean/logical Index operations."""

    def test_all(self):
        """Test all() method."""
        idx = Index([True, True, True])
        assert idx.all() is True

        idx2 = Index([True, False, True])
        assert idx2.all() is False

        idx3 = Index([1, 2, 3])
        assert idx3.all() is True  # Non-zero values are truthy

    def test_any(self):
        """Test any() method."""
        idx = Index([False, False, True])
        assert idx.any() is True

        idx2 = Index([False, False, False])
        assert idx2.any() is False

        idx3 = Index([0, 0, 1])
        assert idx3.any() is True

    def test_is_(self):
        """Test is_() method."""
        idx = Index([1, 2, 3])
        result = idx.is_(2)
        assert isinstance(result, Index)
        assert result.tolist() == [False, True, False]

    def test_isin(self):
        """Test isin() method."""
        idx = Index([1, 2, 3, 4, 5])
        result = idx.isin([2, 4])
        assert isinstance(result, Index)
        assert result.tolist() == [False, True, False, True, False]

    def test_isna(self):
        """Test isna() method."""
        idx = Index([1, None, 3, None, 5])
        result = idx.isna()
        assert isinstance(result, Index)
        assert result.tolist() == [False, True, False, True, False]

    def test_isnull(self):
        """Test isnull() method (alias for isna)."""
        idx = Index([1, None, 3])
        result = idx.isnull()
        assert isinstance(result, Index)
        assert result.tolist() == [False, True, False]

    def test_notna(self):
        """Test notna() method."""
        idx = Index([1, None, 3, None, 5])
        result = idx.notna()
        assert isinstance(result, Index)
        assert result.tolist() == [True, False, True, False, True]

    def test_notnull(self):
        """Test notnull() method (alias for notna)."""
        idx = Index([1, None, 3])
        result = idx.notnull()
        assert isinstance(result, Index)
        assert result.tolist() == [True, False, True]

    def test_where(self):
        """Test where() method."""
        idx = Index([1, 2, 3, 4, 5])
        cond = Index([True, False, True, False, True])
        result = idx.where(cond, 0)
        assert isinstance(result, Index)
        assert result.tolist() == [1, 0, 3, 0, 5]


# ============================================================================
# Statistical Operations
# ============================================================================


class TestIndexStatisticalOperations:
    """Test statistical Index operations."""

    def test_argmax(self):
        """Test argmax() method."""
        idx = Index([3, 1, 4, 1, 5])
        result = idx.argmax()
        assert result == 4  # Index of maximum value (5)

    def test_argmin(self):
        """Test argmin() method."""
        idx = Index([3, 1, 4, 1, 5])
        result = idx.argmin()
        assert result == 1  # Index of first minimum value (1)

    def test_argsort(self):
        """Test argsort() method."""
        idx = Index([3, 1, 4, 2, 5])
        result = idx.argsort()
        assert isinstance(result, Index)
        # Should return indices that would sort the Index
        sorted_indices = result.tolist()
        assert sorted_indices == [1, 3, 0, 2, 4]  # Indices for [1, 2, 3, 4, 5]

    def test_max(self):
        """Test max() method."""
        idx = Index([3, 1, 4, 2, 5])
        assert idx.max() == 5

    def test_min(self):
        """Test min() method."""
        idx = Index([3, 1, 4, 2, 5])
        assert idx.min() == 1

    def test_nunique(self):
        """Test nunique() method."""
        idx = Index([1, 2, 2, 3, 3, 3])
        assert idx.nunique() == 3

    def test_value_counts(self):
        """Test value_counts() method."""
        from polarpandas import Series

        idx = Index([1, 2, 2, 3, 3, 3])
        result = idx.value_counts()
        assert isinstance(result, Series)
        # Should return counts of each value


# ============================================================================
# Data Manipulation
# ============================================================================


class TestIndexDataManipulation:
    """Test data manipulation Index operations."""

    def test_append(self):
        """Test append() method."""
        idx1 = Index([1, 2, 3])
        idx2 = Index([4, 5, 6])
        result = idx1.append(idx2)
        assert isinstance(result, Index)
        assert result.tolist() == [1, 2, 3, 4, 5, 6]

    def test_append_list(self):
        """Test append() with list."""
        idx = Index([1, 2, 3])
        result = idx.append([4, 5, 6])
        assert isinstance(result, Index)
        assert result.tolist() == [1, 2, 3, 4, 5, 6]

    def test_astype(self):
        """Test astype() method."""
        idx = Index([1, 2, 3, 4, 5])
        result = idx.astype(float)
        assert isinstance(result, Index)
        assert result.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_copy(self):
        """Test copy() method."""
        idx = Index([1, 2, 3])
        result = idx.copy()
        assert isinstance(result, Index)
        assert result.tolist() == [1, 2, 3]
        # Should be a different object
        assert result is not idx

    def test_delete(self):
        """Test delete() method."""
        idx = Index([0, 1, 2, 3, 4])
        result = idx.delete(2)
        assert isinstance(result, Index)
        assert result.tolist() == [0, 1, 3, 4]

    def test_delete_multiple(self):
        """Test delete() with multiple indices."""
        idx = Index([0, 1, 2, 3, 4])
        result = idx.delete([1, 3])
        assert isinstance(result, Index)
        assert result.tolist() == [0, 2, 4]

    def test_diff(self):
        """Test diff() method."""
        idx = Index([1, 3, 6, 10, 15])
        result = idx.diff()
        assert isinstance(result, Index)
        # First element should be None/null, rest are differences
        values = result.tolist()
        assert values[0] is None
        assert values[1:] == [2, 3, 4, 5]

    def test_drop(self):
        """Test drop() method."""
        idx = Index([1, 2, 3, 4, 5])
        result = idx.drop([2, 4])
        assert isinstance(result, Index)
        assert result.tolist() == [1, 3, 5]

    def test_drop_duplicates(self):
        """Test drop_duplicates() method."""
        idx = Index([1, 2, 2, 3, 3, 3])
        result = idx.drop_duplicates()
        assert isinstance(result, Index)
        assert result.tolist() == [1, 2, 3]

    def test_drop_duplicates_keep_last(self):
        """Test drop_duplicates() with keep='last'."""
        idx = Index([1, 2, 2, 3, 3, 3])
        result = idx.drop_duplicates(keep="last")
        assert isinstance(result, Index)
        assert len(result) == 3

    def test_dropna(self):
        """Test dropna() method."""
        idx = Index([1, None, 3, None, 5])
        result = idx.dropna()
        assert isinstance(result, Index)
        assert result.tolist() == [1, 3, 5]

    def test_fillna(self):
        """Test fillna() method."""
        idx = Index([1, None, 3, None, 5])
        result = idx.fillna(0)
        assert isinstance(result, Index)
        assert result.tolist() == [1, 0, 3, 0, 5]

    def test_insert(self):
        """Test insert() method."""
        idx = Index([1, 2, 3, 4])
        result = idx.insert(2, 99)
        assert isinstance(result, Index)
        assert result.tolist() == [1, 2, 99, 3, 4]

    def test_map(self):
        """Test map() method with function."""
        idx = Index([1, 2, 3, 4, 5])
        result = idx.map(lambda x: x * 2)
        assert isinstance(result, Index)
        assert result.tolist() == [2, 4, 6, 8, 10]

    def test_map_dict(self):
        """Test map() method with dictionary."""
        idx = Index([1, 2, 3, 4, 5])
        mapping = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e"}
        result = idx.map(mapping)
        assert isinstance(result, Index)
        assert result.tolist() == ["a", "b", "c", "d", "e"]

    def test_putmask(self):
        """Test putmask() method."""
        idx = Index([1, 2, 3, 4, 5])
        mask = Index([True, False, True, False, True])
        result = idx.putmask(mask, 0)
        assert isinstance(result, Index)
        assert result.tolist() == [0, 2, 0, 4, 0]

    def test_ravel(self):
        """Test ravel() method."""
        idx = Index([1, 2, 3])
        result = idx.ravel()
        assert isinstance(result, Index)
        assert result.tolist() == [1, 2, 3]

    def test_repeat(self):
        """Test repeat() method."""
        idx = Index([1, 2, 3])
        result = idx.repeat(2)
        assert isinstance(result, Index)
        assert result.tolist() == [1, 1, 2, 2, 3, 3]

    def test_round(self):
        """Test round() method."""
        idx = Index([1.234, 2.567, 3.891])
        result = idx.round(1)
        assert isinstance(result, Index)
        assert result.tolist() == [1.2, 2.6, 3.9]

    def test_shift(self):
        """Test shift() method."""
        idx = Index([1, 2, 3, 4, 5])
        result = idx.shift(1)
        assert isinstance(result, Index)
        values = result.tolist()
        assert values[0] is None
        assert values[1:] == [1, 2, 3, 4]

    def test_shift_with_fill_value(self):
        """Test shift() with fill_value."""
        idx = Index([1, 2, 3, 4, 5])
        result = idx.shift(1, fill_value=0)
        assert isinstance(result, Index)
        assert result.tolist() == [0, 1, 2, 3, 4]

    def test_take(self):
        """Test take() method."""
        idx = Index([10, 20, 30, 40, 50])
        result = idx.take([0, 2, 4])
        assert isinstance(result, Index)
        assert result.tolist() == [10, 30, 50]

    def test_transpose(self):
        """Test transpose() method."""
        idx = Index([1, 2, 3])
        result = idx.transpose()
        assert isinstance(result, Index)
        assert result.tolist() == [1, 2, 3]  # No-op for 1D


# ============================================================================
# Set Operations
# ============================================================================


class TestIndexSetOperations:
    """Test set operation Index methods."""

    def test_difference(self):
        """Test difference() method."""
        idx1 = Index([1, 2, 3, 4, 5])
        idx2 = Index([2, 4, 6])
        result = idx1.difference(idx2)
        assert isinstance(result, Index)
        assert set(result.tolist()) == {1, 3, 5}

    def test_intersection(self):
        """Test intersection() method."""
        idx1 = Index([1, 2, 3, 4, 5])
        idx2 = Index([2, 4, 6, 8])
        result = idx1.intersection(idx2)
        assert isinstance(result, Index)
        assert set(result.tolist()) == {2, 4}

    def test_symmetric_difference(self):
        """Test symmetric_difference() method."""
        idx1 = Index([1, 2, 3, 4])
        idx2 = Index([3, 4, 5, 6])
        result = idx1.symmetric_difference(idx2)
        assert isinstance(result, Index)
        assert set(result.tolist()) == {1, 2, 5, 6}

    def test_union(self):
        """Test union() method."""
        idx1 = Index([1, 2, 3])
        idx2 = Index([3, 4, 5])
        result = idx1.union(idx2)
        assert isinstance(result, Index)
        assert set(result.tolist()) == {1, 2, 3, 4, 5}

    def test_unique(self):
        """Test unique() method."""
        idx = Index([1, 2, 2, 3, 3, 3])
        result = idx.unique()
        assert isinstance(result, Index)
        assert set(result.tolist()) == {1, 2, 3}


# ============================================================================
# Indexing/Location Operations
# ============================================================================


class TestIndexLocationOperations:
    """Test indexing/location Index operations."""

    def test_asof(self):
        """Test asof() method."""
        idx = Index([1, 3, 5, 7, 9])
        result = idx.asof(6)
        assert result == 5  # Last value <= 6

    def test_get_indexer(self):
        """Test get_indexer() method."""
        idx = Index([10, 20, 30, 40, 50])
        target = [20, 40, 60]
        result = idx.get_indexer(target)
        assert result == [1, 3, -1]  # 20 at index 1, 40 at index 3, 60 not found

    def test_get_indexer_for(self):
        """Test get_indexer_for() method."""
        idx = Index([10, 20, 30])
        target = [20, 30]
        result = idx.get_indexer_for(target)
        assert result == [1, 2]

    def test_get_indexer_non_unique(self):
        """Test get_indexer_non_unique() method."""
        idx = Index([10, 20, 20, 30])
        target = [20, 30, 40]
        indexer, missing = idx.get_indexer_non_unique(target)
        assert indexer == [1, 3, -1]  # First occurrence of 20, 30 found, 40 missing
        assert missing == [2]  # Index 2 in target is missing

    def test_get_level_values(self):
        """Test get_level_values() method."""
        idx = Index([1, 2, 3])
        result = idx.get_level_values(0)
        assert isinstance(result, Index)
        assert result.tolist() == [1, 2, 3]

    def test_get_loc(self):
        """Test get_loc() method."""
        idx = Index([10, 20, 30, 40, 50])
        result = idx.get_loc(30)
        assert result == 2

    def test_get_loc_not_found(self):
        """Test get_loc() with missing key."""
        import pytest

        idx = Index([10, 20, 30])
        with pytest.raises(KeyError):
            idx.get_loc(40)

    def test_get_slice_bound(self):
        """Test get_slice_bound() method."""
        idx = Index([10, 20, 30, 40, 50])
        result = idx.get_slice_bound(25, "left")
        assert result == 2  # First index >= 25

    def test_searchsorted(self):
        """Test searchsorted() method."""
        idx = Index([10, 20, 30, 40, 50])
        result = idx.searchsorted(25)
        assert result == 2  # Insert position for 25

    def test_slice_indexer(self):
        """Test slice_indexer() method."""
        idx = Index([10, 20, 30, 40, 50])
        result = idx.slice_indexer(20, 40)
        assert isinstance(result, slice)
        assert result.start == 1
        # end should be the first index > 40, which is 4 (index of 50)
        assert result.stop == 4

    def test_slice_locs(self):
        """Test slice_locs() method."""
        idx = Index([10, 20, 30, 40, 50])
        start, end = idx.slice_locs(20, 40)
        assert start == 1
        # end should be the first index > 40, which is 4 (index of 50)
        assert end == 4


# ============================================================================
# MultiIndex Operations
# ============================================================================


class TestIndexMultiIndexOperations:
    """Test MultiIndex operation Index methods."""

    def test_droplevel(self):
        """Test droplevel() method."""
        idx = Index([1, 2, 3])
        result = idx.droplevel(0)
        assert isinstance(result, Index)
        # For regular Index, should return self
        assert result.tolist() == [1, 2, 3]

    def test_set_names(self):
        """Test set_names() method."""
        idx = Index([1, 2, 3])
        result = idx.set_names("new_name")
        assert isinstance(result, Index)
        assert result._series.name == "new_name"

    def test_sortlevel(self):
        """Test sortlevel() method."""
        idx = Index([3, 1, 4, 2, 5])
        sorted_idx, indices = idx.sortlevel()
        assert isinstance(sorted_idx, Index)
        assert sorted_idx.tolist() == [1, 2, 3, 4, 5]
        assert isinstance(indices, list)

    def test_to_flat_index(self):
        """Test to_flat_index() method."""
        idx = Index([1, 2, 3])
        result = idx.to_flat_index()
        assert isinstance(result, Index)
        assert result.tolist() == [1, 2, 3]


# ============================================================================
# Type Conversion/Export
# ============================================================================


class TestIndexTypeConversion:
    """Test type conversion Index methods."""

    def test_to_frame(self):
        """Test to_frame() method."""
        from polarpandas import DataFrame

        idx = Index([1, 2, 3, 4, 5])
        result = idx.to_frame()
        assert isinstance(result, DataFrame)
        assert len(result) == 5

    def test_to_list(self):
        """Test to_list() method."""
        idx = Index([1, 2, 3])
        result = idx.to_list()
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_to_numpy(self):
        """Test to_numpy() method."""
        import numpy as np

        idx = Index([1, 2, 3, 4, 5])
        result = idx.to_numpy()
        assert isinstance(result, np.ndarray)
        assert result.tolist() == [1, 2, 3, 4, 5]

    def test_to_series(self):
        """Test to_series() method."""
        from polarpandas import Series

        idx = Index([1, 2, 3, 4, 5])
        result = idx.to_series()
        assert isinstance(result, Series)
        assert result.tolist() == [1, 2, 3, 4, 5]

    def test_view(self):
        """Test view() method."""
        idx = Index([1, 2, 3])
        result = idx.view()
        assert isinstance(result, Index)
        assert result.tolist() == [1, 2, 3]


# ============================================================================
# Comparison/Equality
# ============================================================================


class TestIndexComparison:
    """Test comparison Index methods."""

    def test_duplicated(self):
        """Test duplicated() method."""
        idx = Index([1, 2, 2, 3, 3, 3])
        result = idx.duplicated()
        assert isinstance(result, Index)
        assert result.tolist() == [False, False, True, False, True, True]

    def test_duplicated_keep_last(self):
        """Test duplicated() with keep='last'."""
        idx = Index([1, 2, 2, 3, 3, 3])
        result = idx.duplicated(keep="last")
        assert isinstance(result, Index)
        # Last occurrence of each duplicate is False
        values = result.tolist()
        assert values[0] is False
        assert values[1] is True  # First 2
        assert values[2] is False  # Last 2

    def test_equals(self):
        """Test equals() method."""
        idx1 = Index([1, 2, 3])
        idx2 = Index([1, 2, 3])
        idx3 = Index([1, 2, 4])
        assert idx1.equals(idx2) is True
        assert idx1.equals(idx3) is False

    def test_identical(self):
        """Test identical() method."""
        idx1 = Index([1, 2, 3])
        idx2 = Index([1, 2, 3])
        idx3 = idx1
        assert idx1.identical(idx1) is True
        assert idx1.identical(idx3) is True
        assert idx1.identical(idx2) is False  # Different objects


# ============================================================================
# Other Operations
# ============================================================================


class TestIndexOtherOperations:
    """Test other Index operations."""

    def test_factorize(self):
        """Test factorize() method."""
        idx = Index(["a", "b", "a", "c", "b"])
        codes, uniques = idx.factorize()
        assert isinstance(codes, list)
        assert isinstance(uniques, Index)
        assert len(codes) == 5
        assert len(uniques) == 3

    def test_groupby(self):
        """Test groupby() method."""
        import pytest

        idx = Index([1, 2, 2, 3, 3, 3])
        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="groupby.*not yet implemented"):
            idx.groupby()

    def test_infer_objects(self):
        """Test infer_objects() method."""
        idx = Index([1, 2, 3])
        result = idx.infer_objects()
        assert isinstance(result, Index)
        assert result.tolist() == [1, 2, 3]

    def test_item(self):
        """Test item() method."""
        idx = Index([42])
        result = idx.item()
        assert result == 42

    def test_item_error(self):
        """Test item() raises error for non-single element."""
        import pytest

        idx = Index([1, 2, 3])
        with pytest.raises(ValueError, match="can only convert an array of size 1"):
            idx.item()

    def test_join(self):
        """Test join() method."""
        idx1 = Index([1, 2, 3])
        idx2 = Index([2, 3, 4])
        result = idx1.join(idx2, how="inner")
        assert isinstance(result, Index)
        assert set(result.tolist()) == {2, 3}

    def test_join_outer(self):
        """Test join() with how='outer'."""
        idx1 = Index([1, 2, 3])
        idx2 = Index([3, 4, 5])
        result = idx1.join(idx2, how="outer")
        assert isinstance(result, Index)
        assert set(result.tolist()) == {1, 2, 3, 4, 5}

    def test_memory_usage(self):
        """Test memory_usage() method."""
        idx = Index([1, 2, 3, 4, 5])
        result = idx.memory_usage()
        assert isinstance(result, int)
        assert result > 0

    def test_reindex(self):
        """Test reindex() method."""
        idx = Index([10, 20, 30])
        target = [20, 30, 40]
        result, indexer = idx.reindex(target, fill_value=0)
        assert isinstance(result, Index)
        assert result.tolist() == [20, 30, 0]
        assert indexer == [1, 2, -1]

    def test_rename(self):
        """Test rename() method."""
        idx = Index([1, 2, 3])
        result = idx.rename("new_name")
        assert isinstance(result, Index)
        assert result._series.name == "new_name"

    def test_sort_values(self):
        """Test sort_values() method."""
        idx = Index([3, 1, 4, 2, 5])
        result = idx.sort_values()
        assert isinstance(result, Index)
        assert result.tolist() == [1, 2, 3, 4, 5]

    def test_sort_values_descending(self):
        """Test sort_values() with ascending=False."""
        idx = Index([3, 1, 4, 2, 5])
        result = idx.sort_values(ascending=False)
        assert isinstance(result, Index)
        assert result.tolist() == [5, 4, 3, 2, 1]

    def test_sort_values_with_indexer(self):
        """Test sort_values() with return_indexer=True."""
        idx = Index([3, 1, 4, 2, 5])
        sorted_idx, indexer = idx.sort_values(return_indexer=True)
        assert isinstance(sorted_idx, Index)
        assert isinstance(indexer, Index)
        assert sorted_idx.tolist() == [1, 2, 3, 4, 5]


# ============================================================================
# String Accessor
# ============================================================================


class TestIndexStringAccessor:
    """Test string accessor property."""

    def test_str_property(self):
        """Test str property."""
        idx = Index(["hello", "world", "test"])
        str_accessor = idx.str
        assert str_accessor is not None
        # Should be able to use string methods
        result = str_accessor.upper()
        assert result.tolist() == ["HELLO", "WORLD", "TEST"]
