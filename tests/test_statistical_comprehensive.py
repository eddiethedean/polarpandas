"""Comprehensive statistical tests without relying on pandas."""

import math

import numpy as np
import pytest

import polarpandas as ppd
from tests.test_helpers import assert_frame_equal


class TestStatisticalMethodsComprehensive:
    """Focused checks for DataFrame statistical helpers."""

    def setup_method(self) -> None:
        self.data = {
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": [1.1, 2.2, 3.3, 4.4, 5.5],
        }

    def test_corr_linear_columns(self) -> None:
        df = ppd.DataFrame(self.data)
        result = df.corr()
        expected = {
            "A": [1.0, 1.0, 1.0],
            "B": [1.0, 1.0, 1.0],
            "C": [1.0, 1.0, 1.0],
        }
        assert_frame_equal(result, expected, rtol=1e-9)

    def test_covariance_values(self) -> None:
        df = ppd.DataFrame(self.data)
        result = df.cov()
        expected = {
            "A": [2.5, 25.0, 2.75],
            "B": [25.0, 250.0, 27.5],
            "C": [2.75, 27.5, 3.025],
        }
        assert_frame_equal(result, expected, rtol=1e-9)

    def test_covariance_with_nulls(self) -> None:
        df = ppd.DataFrame({"A": [1, None, 3, 4, 5], "B": [10, 20, None, 40, 50]})
        cov = df.cov().to_dict()
        assert math.isclose(cov["A"][0], np.nanvar([1, 3, 4, 5], ddof=1))
        assert math.isclose(cov["B"][1], np.nanvar([10, 20, 40, 50], ddof=1))

    def test_rank_variants(self) -> None:
        df = ppd.DataFrame(self.data)
        expected_standard = {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [1.0, 2.0, 3.0, 4.0, 5.0],
            "C": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
        assert_frame_equal(df.rank(), expected_standard)
        assert_frame_equal(df.rank(method="min"), expected_standard)
        assert_frame_equal(
            df.rank(ascending=False),
            {
                "A": [5.0, 4.0, 3.0, 2.0, 1.0],
                "B": [5.0, 4.0, 3.0, 2.0, 1.0],
                "C": [5.0, 4.0, 3.0, 2.0, 1.0],
            },
        )

    def test_rank_numeric_only(self) -> None:
        df = ppd.DataFrame({**self.data, "D": ["a", "b", "c", "d", "e"]})
        numeric_ranks = df.rank(numeric_only=True)
        assert list(numeric_ranks.columns) == ["A", "B", "C"]
        assert_frame_equal(
            numeric_ranks,
            {
                "A": [1.0, 2.0, 3.0, 4.0, 5.0],
                "B": [1.0, 2.0, 3.0, 4.0, 5.0],
                "C": [1.0, 2.0, 3.0, 4.0, 5.0],
            },
        )

    def test_diff_multiple_periods(self) -> None:
        df = ppd.DataFrame(self.data)
        assert_frame_equal(
            df.diff(),
            {
                "A": [None, 1, 1, 1, 1],
                "B": [None, 10, 10, 10, 10],
                "C": [None, 1.1, 1.1, 1.1, 1.1],
            },
            rtol=1e-9,
        )
        assert_frame_equal(
            df.diff(periods=2),
            {
                "A": [None, None, 2, 2, 2],
                "B": [None, None, 20, 20, 20],
                "C": [None, None, 2.2, 2.2, 2.2],
            },
            rtol=1e-9,
        )

    def test_pct_change(self) -> None:
        df = ppd.DataFrame(self.data)
        assert_frame_equal(
            df.pct_change(),
            {
                "A": [None, 1.0, 0.5, 1 / 3, 0.25],
                "B": [None, 1.0, 0.5, 1 / 3, 0.25],
                "C": [None, 1.0, 0.5, 1 / 3, 0.25],
            },
            rtol=1e-9,
        )

    def test_cumulative_operations_with_nulls(self) -> None:
        df = ppd.DataFrame({"A": [1, None, 3, 4, 5], "B": [10, 20, None, 40, 50]})
        assert_frame_equal(
            df.cumsum(), {"A": [1, None, 4, 8, 13], "B": [10, 30, None, 70, 120]}
        )
        assert_frame_equal(
            df.cumprod(),
            {"A": [1, None, 3, 12, 60], "B": [10, 200, None, 8000, 400000]},
        )
        assert_frame_equal(
            df.cummax(), {"A": [1, None, 3, 4, 5], "B": [10, 20, None, 40, 50]}
        )
        assert_frame_equal(
            df.cummin(), {"A": [1, None, 1, 1, 1], "B": [10, 10, None, 10, 10]}
        )

    def test_statistical_methods_empty_dataframe(self) -> None:
        df = ppd.DataFrame()
        corr = df.corr()
        assert corr.shape == (0, 0)
        assert len(corr.columns) == 0

    def test_statistical_methods_single_row(self) -> None:
        df = ppd.DataFrame({"A": [1], "B": [10], "C": [1.1]})
        cov = df.cov()
        nan_row = [float("nan"), float("nan"), float("nan")]
        assert_frame_equal(
            cov,
            {"A": nan_row, "B": nan_row, "C": nan_row},
            check_order=False,
        )

    def test_corr_method_validation(self) -> None:
        df = ppd.DataFrame(self.data)
        with pytest.raises(NotImplementedError):
            df.corr(method="kendall")


class TestStatisticalEdgeCases:
    """Additional edge-focused exercises."""

    def test_pct_change_with_nulls(self) -> None:
        df = ppd.DataFrame({"A": [1, None, 3, 6]})
        result = df.pct_change().to_dict()["A"]
        assert result[0] is None
        assert result[1] is None
        assert result[2] is None
        assert math.isclose(result[3], 1.0)

    def test_mode_skew_kurt_handles_nulls(self) -> None:
        series = ppd.Series([1, 2, 2, None, 3])
        assert series.mode().to_list() == [2]
        assert math.isfinite(series.skew())
        assert math.isfinite(series.kurt())
