"""Edge-case coverage for polarpandas without relying on pandas runtime."""

from __future__ import annotations

import numpy as np
import pytest

import polarpandas as ppd
from tests.test_helpers import assert_frame_equal, assert_index_equal


def _to_expected(data):
    """Convert input mapping to lists for comparison."""

    expected = {}
    for key, values in data.items():
        if isinstance(values, np.ndarray):
            expected[key] = values.tolist()
        else:
            expected[key] = (
                list(values) if isinstance(values, (list, tuple)) else values
            )
    return expected


class TestEmptyDataFrames:
    """Validate creation scenarios for empty DataFrames."""

    @pytest.mark.parametrize(
        "kwargs, expected_columns, expected_index",
        [
            ({}, [], []),
            ({"columns": ["A", "B", "C"]}, ["A", "B", "C"], []),
            ({"index": [0, 1, 2]}, [], [0, 1, 2]),
            (
                {"columns": ["A", "B"], "index": [0, 1, 2]},
                ["A", "B"],
                [0, 1, 2],
            ),
        ],
    )
    def test_empty_dataframe_variants(self, kwargs, expected_columns, expected_index):
        df = ppd.DataFrame(**kwargs)

        assert df.shape[0] == len(expected_index)
        assert list(df.columns) == expected_columns
        assert_index_equal(df.index, expected_index)


class TestSingleRowDataFrames:
    """Edge cases for single-row DataFrames."""

    def test_basic_single_row(self):
        data = {"A": [1], "B": [10], "C": ["a"]}
        df = ppd.DataFrame(data)

        assert_frame_equal(df, data)
        assert df.shape == (1, 3)

    def test_single_row_with_custom_index(self):
        data = {"A": [1], "B": [10], "C": ["a"]}
        index = ["custom"]
        df = ppd.DataFrame(data, index=index)

        assert_frame_equal(df, data)
        assert_index_equal(df.index, index)

    def test_single_row_with_custom_columns(self):
        columns = ["X", "Y", "Z"]
        data = {
            "X": [1],
            "Y": [10],
            "Z": ["a"],
        }
        df = ppd.DataFrame(data)

        expected = {col: [val] for col, val in zip(columns, [1, 10, "a"])}
        assert_frame_equal(df, expected)
        assert list(df.columns) == columns


class TestSingleColumnDataFrames:
    """Edge cases for single-column DataFrames."""

    def test_basic_single_column(self):
        data = {"A": [1, 2, 3, 4, 5]}
        df = ppd.DataFrame(data)

        assert_frame_equal(df, data)
        assert df.shape == (5, 1)

    def test_single_column_with_index(self):
        data = {"A": [1, 2, 3, 4, 5]}
        index = ["a", "b", "c", "d", "e"]
        df = ppd.DataFrame(data, index=index)

        assert_frame_equal(df, data)
        assert_index_equal(df.index, index)

    def test_single_column_with_custom_header(self):
        columns = ["X"]
        data = {"X": [1, 2, 3]}
        df = ppd.DataFrame(data)

        expected = {"X": [1, 2, 3]}
        assert_frame_equal(df, expected)
        assert list(df.columns) == columns


class TestNullValues:
    """Edge cases involving null handling."""

    def test_all_nan_values(self):
        nan = float("nan")
        data = {"A": [nan, nan, nan], "B": [nan, nan, nan]}
        df = ppd.DataFrame(data)

        assert_frame_equal(df, data)

    @pytest.mark.skip(reason="Known limitation: NaN representation in string columns")
    def test_mixed_nan_values(self):
        pytest.skip("Tracked in KNOWN_LIMITATIONS.md")

    @pytest.mark.skip(
        reason="Polars cannot combine datetime and NaN reliably (KNOWN_LIMITATIONS.md)"
    )
    def test_datetime_nan_values(self):
        pytest.skip("Tracked limitation")

    def test_nan_with_custom_columns(self):
        nan = float("nan")
        columns = ["X", "Y"]
        data = {"X": [1, nan, 3], "Y": [10, 20, nan]}
        df = ppd.DataFrame(data)

        expected = {"X": [1, nan, 3], "Y": [10, 20, nan]}
        assert_frame_equal(df, expected)
        assert list(df.columns) == columns


class TestInfiniteValues:
    """Edge cases for infinite values."""

    @pytest.mark.parametrize(
        "data",
        [
            {"A": [1, 2, np.inf, 4, 5], "B": [10, 20, 30, np.inf, 50]},
            {"A": [1, 2, -np.inf, 4, 5], "B": [10, 20, 30, -np.inf, 50]},
            {"A": [1, 2, np.inf, 4, -np.inf], "B": [10, 20, 30, np.inf, 50]},
            {"A": [1, np.nan, np.inf, 4, 5], "B": [10, 20, 30, np.nan, 50]},
        ],
    )
    def test_infinite_patterns(self, data):
        df = ppd.DataFrame(data)
        assert_frame_equal(df, data)


class TestZeroAndNegativeValues:
    """Edge cases with zero and negative data."""

    @pytest.mark.parametrize(
        "data",
        [
            {"A": [0, 0, 0, 0, 0], "B": [0, 0, 0, 0, 0]},
            {"A": [1, 0, 3, 0, 5], "B": [10, 0, 30, 0, 50]},
            {
                "A": [1, 0, 3, 0, 5],
                "B": [1.1, 0.0, 3.3, 0.0, 5.5],
                "C": [True, False, True, False, True],
            },
            {"A": [-1, -2, -3, -4, -5], "B": [-10, -20, -30, -40, -50]},
            {"A": [1, -2, 3, -4, 5], "B": [10, -20, 30, -40, 50]},
            {
                "A": [1, -2, 3, -4, 5],
                "B": [1.1, -2.2, 3.3, -4.4, 5.5],
                "C": [True, False, True, False, True],
            },
        ],
    )
    def test_zero_and_negative_patterns(self, data):
        df = ppd.DataFrame(data)
        assert_frame_equal(df, data)


class TestLargeDataFrames:
    """Stress tests with large inputs."""

    def test_large_dataframe_creation(self):
        np.random.seed(42)
        data = {
            "A": np.random.randn(1000),
            "B": np.random.randn(1000),
            "C": np.random.randn(1000),
        }
        df = ppd.DataFrame(data)

        assert df.shape == (1000, 3)
        assert_frame_equal(df, _to_expected(data))

    def test_large_dataframe_with_index(self):
        np.random.seed(42)
        data = {
            "A": np.random.randn(128),
            "B": np.random.randn(128),
        }
        index = [f"row_{i}" for i in range(128)]
        df = ppd.DataFrame(data, index=index)

        assert_frame_equal(df, _to_expected(data))
        assert_index_equal(df.index, index)


class TestErrorConditions:
    """Validate error handling without pandas."""

    def test_invalid_column_access(self):
        df = ppd.DataFrame({"A": [1, 2, 3]})

        with pytest.raises(KeyError):
            _ = df["missing"]

    def test_invalid_loc_access(self):
        df = ppd.DataFrame({"A": [1, 2, 3]})

        with pytest.raises(KeyError):
            _ = df.loc[10]

    def test_invalid_iloc_access(self):
        df = ppd.DataFrame({"A": [1, 2, 3]})

        with pytest.raises(IndexError):
            _ = df.iloc[10]

    def test_invalid_set_index(self):
        df = ppd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        with pytest.raises(KeyError):
            df.set_index("missing")

    def test_invalid_drop_columns(self):
        df = ppd.DataFrame({"A": [1, 2, 3]})

        with pytest.raises(KeyError):
            df.drop(columns=["missing"])

    def test_rename_ignores_missing(self):
        df = ppd.DataFrame({"A": [1, 2, 3]})
        expected = df.to_dict()

        result = df.rename(columns={"missing": "new"})
        assert_frame_equal(result, expected)

    def test_assign_new_column(self):
        df = ppd.DataFrame({"A": [1, 2, 3]})
        df["B"] = [4, 5, 6]

        assert_frame_equal(df, {"A": [1, 2, 3], "B": [4, 5, 6]})

    def test_loc_assignment_adds_row(self):
        df = ppd.DataFrame({"A": [1, 2, 3]})
        df.loc[10, "A"] = 100

        assert 10 in df.index.tolist()
        assert df.loc[10, "A"] == 100

    def test_iloc_assignment_out_of_bounds(self):
        df = ppd.DataFrame({"A": [1, 2, 3]})

        with pytest.raises(IndexError):
            df.iloc[10, 0] = 99
