"""Comprehensive GroupBy tests without pandas dependency."""

import polars as pl

import polarpandas as ppd
from tests.test_helpers import assert_frame_equal


class TestGroupByAggregations:
    """Validate core aggregation behaviour on grouped data."""

    def setup_method(self):
        self.data = {
            "A": ["foo", "foo", "bar", "bar", "foo"],
            "B": ["one", "two", "one", "two", "one"],
            "C": [1, 2, 3, 4, 5],
            "D": [10, 20, 30, 40, 50],
        }
        self.df = ppd.DataFrame(self.data)

    def test_groupby_mean(self):
        result = self.df.groupby("A").agg(pl.col("C").mean()).sort_values("A")
        expected = {"A": ["bar", "foo"], "C": [3.5, 8 / 3]}
        assert_frame_equal(result, expected, rtol=1e-6)

    def test_groupby_sum(self):
        result = self.df.groupby("A").agg(pl.col("C").sum()).sort_values("A")
        expected = {"A": ["bar", "foo"], "C": [7, 8]}
        assert_frame_equal(result, expected)

    def test_groupby_min(self):
        result = self.df.groupby("A").agg(pl.col("C").min()).sort_values("A")
        expected = {"A": ["bar", "foo"], "C": [3, 1]}
        assert_frame_equal(result, expected)

    def test_groupby_max(self):
        result = self.df.groupby("A").agg(pl.col("C").max()).sort_values("A")
        expected = {"A": ["bar", "foo"], "C": [4, 5]}
        assert_frame_equal(result, expected)

    def test_groupby_std(self):
        result = self.df.groupby("A").agg(pl.col("C").std()).sort_values("A")
        expected = {
            "A": ["bar", "foo"],
            "C": [0.7071067811865476, 2.081665999466133],
        }
        assert_frame_equal(result, expected, rtol=1e-9)

    def test_groupby_var(self):
        result = self.df.groupby("A").agg(pl.col("C").var()).sort_values("A")
        expected = {"A": ["bar", "foo"], "C": [0.5, 4.333333333333334]}
        assert_frame_equal(result, expected, rtol=1e-9)

    def test_groupby_median(self):
        result = self.df.groupby("A").agg(pl.col("C").median()).sort_values("A")
        expected = {"A": ["bar", "foo"], "C": [3.5, 2.0]}
        assert_frame_equal(result, expected)

    def test_groupby_multiple_columns(self):
        result = (
            self.df.groupby(["A", "B"]).agg(pl.col("C").sum()).sort_values(["A", "B"])
        )
        expected = {
            "A": ["bar", "bar", "foo", "foo"],
            "B": ["one", "two", "one", "two"],
            "C": [3, 4, 6, 2],
        }
        assert_frame_equal(result, expected)

    def test_groupby_multiple_aggregations(self):
        result = (
            self.df.groupby("A")
            .agg(
                [
                    pl.col("C").mean().alias("C_mean"),
                    pl.col("C").sum().alias("C_sum"),
                ]
            )
            .sort_values("A")
        )
        expected = {
            "A": ["bar", "foo"],
            "C_mean": [3.5, 8 / 3],
            "C_sum": [7, 8],
        }
        assert_frame_equal(result, expected, rtol=1e-6)

    def test_groupby_empty_groups(self):
        result = (
            ppd.DataFrame({"A": ["foo"], "C": [10]}).groupby("A").agg(pl.col("C").sum())
        )
        expected = {"A": ["foo"], "C": [10]}
        assert_frame_equal(result, expected)

    def test_groupby_with_nulls(self):
        data = {
            "A": ["foo", "bar", None, "foo"],
            "B": [1, 2, 3, 4],
            "C": [10, 20, 30, 40],
        }
        df = ppd.DataFrame(data)
        result = df.groupby("A").agg(pl.col("C").sum()).sort_values("A")
        expected = {"A": [None, "bar", "foo"], "C": [30, 20, 50]}
        assert_frame_equal(result, expected)
