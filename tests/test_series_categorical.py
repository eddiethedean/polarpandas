"""Tests for categorical accessor functionality."""

import pytest

import polarpandas as ppd
from tests.test_helpers import assert_index_equal, assert_series_equal


class TestSeriesCategoricalAccessor:
    def test_categories_and_codes(self) -> None:
        s = ppd.Series(["b", "a", "b", None]).astype("category")

        assert_index_equal(s.cat.categories, ["b", "a"])

        codes = s.cat.codes
        assert_series_equal(codes, [0, 1, 0, -1])

    def test_rename_categories_list(self) -> None:
        s = ppd.Series(["a", "b", "a"]).astype("category")

        renamed = s.cat.rename_categories(["first", "second"])

        assert_index_equal(renamed.cat.categories, ["first", "second"])
        assert_series_equal(renamed, ["first", "second", "first"])
        # Original remains unchanged
        assert_index_equal(s.cat.categories, ["a", "b"])

    def test_rename_categories_dict_inplace(self) -> None:
        s = ppd.Series(["a", "b", "c"]).astype("category")
        s.cat.rename_categories({"b": "beta"}, inplace=True)

        assert_index_equal(s.cat.categories, ["a", "beta", "c"])
        assert_series_equal(s, ["a", "beta", "c"])

    def test_rename_categories_validation(self) -> None:
        s = ppd.Series(["a", "b"]).astype("category")

        with pytest.raises(KeyError):
            s.cat.rename_categories({"missing": "value"})

        with pytest.raises(ValueError):
            s.cat.rename_categories(["dup", "dup"])

    def test_reorder_categories(self) -> None:
        s = ppd.Series(["b", "a", "c", "b"]).astype("category")

        reordered = s.cat.reorder_categories(["c", "b", "a"])

        assert_index_equal(reordered.cat.categories, ["c", "b", "a"])
        assert_series_equal(reordered.cat.codes, [1, 2, 0, 1])

        s.cat.reorder_categories(["b", "c", "a"], inplace=True)
        assert_index_equal(s.cat.categories, ["b", "c", "a"])

    def test_reorder_categories_validation(self) -> None:
        s = ppd.Series(["a", "b"]).astype("category")

        with pytest.raises(ValueError):
            s.cat.reorder_categories(["a", "a"])

        with pytest.raises(ValueError):
            s.cat.reorder_categories(["a", "b", "c"])

    def test_cat_requires_categorical_dtype(self) -> None:
        s = ppd.Series(["a", "b"])

        with pytest.raises(TypeError):
            _ = s.cat
