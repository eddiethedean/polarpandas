"""Advanced `loc`/`iloc` tests that run without pandas."""

import pytest

import polarpandas as ppd
from tests.test_helpers import assert_frame_equal


class TestLocAdvanced:
    """Behavioural checks for `.loc` indexing."""

    def setup_method(self) -> None:
        self.df = ppd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [10, 20, 30, 40, 50],
                "C": ["a", "b", "c", "d", "e"],
            }
        )

    def test_single_cell_access(self) -> None:
        assert self.df.loc[0, "A"] == 1
        indexed = ppd.DataFrame(self.df.to_dict(), index=["x", "y", "z", "w", "v"])
        assert indexed.loc["x", "B"] == 10

    def test_slice_access(self) -> None:
        result = self.df.loc[1:3, "A":"B"]
        assert_frame_equal(result, {"A": [2, 3], "B": [20, 30]})

    def test_list_indexing(self) -> None:
        result = self.df.loc[[0, 2], ["A", "B"]]
        assert_frame_equal(result, {"A": [1, 3], "B": [10, 30]})

    def test_column_selection_returns_dataframe(self) -> None:
        column_df = self.df.loc[:, "A"]
        assert isinstance(column_df, ppd.DataFrame)
        assert column_df.to_dict() == {"A": [1, 2, 3, 4, 5]}

    def test_loc_errors(self) -> None:
        with pytest.raises(TypeError):
            _ = self.df.loc["missing", "A"]
        with pytest.raises(KeyError):
            _ = self.df.loc[0, "missing"]


class TestILocAdvanced:
    """Behavioural checks for `.iloc` indexing."""

    def setup_method(self) -> None:
        self.df = ppd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [10, 20, 30, 40, 50],
                "C": ["a", "b", "c", "d", "e"],
            }
        )

    def test_single_cell_access(self) -> None:
        assert self.df.iloc[0, 0] == 1
        assert self.df.iloc[2, 1] == 30

    def test_slice_access(self) -> None:
        result = self.df.iloc[1:3, 0:2]
        assert_frame_equal(result, {"A": [2, 3], "B": [20, 30]})

    def test_list_indexing(self) -> None:
        result = self.df.iloc[[0, 2, 4], [0, 2]]
        assert_frame_equal(result, {"A": [1, 3, 5], "C": ["a", "c", "e"]})

    def test_iloc_errors(self) -> None:
        with pytest.raises(IndexError):
            _ = self.df.iloc[10]
        with pytest.raises(IndexError):
            _ = self.df.iloc[:, 10]


class TestAtIat:
    """Scalar accessor coverage."""

    def setup_method(self) -> None:
        self.df = ppd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [10, 20, 30],
                "C": ["x", "y", "z"],
            },
            index=["p", "q", "r"],
        )

    def test_at_access(self) -> None:
        assert self.df.at["p", "A"] == 1
        self.df.at["q", "B"] = 999
        assert self.df.at["q", "B"] == 999

    def test_iat_access(self) -> None:
        assert self.df.iat[0, 0] == 1
        self.df.iat[1, 1] = 111
        assert self.df.iat[1, 1] == 111


class TestEdgeCases:
    """Edge-focused loc/iloc scenarios."""

    def test_empty_dataframe(self) -> None:
        df = ppd.DataFrame(columns=["A", "B"])
        with pytest.raises(KeyError):
            _ = df.loc[0]
        with pytest.raises(IndexError):
            _ = df.iloc[0]

    def test_single_column_dataframe(self) -> None:
        df = ppd.DataFrame({"A": [1, 2, 3]})
        assert df.loc[:, "A"].to_dict() == {"A": [1, 2, 3]}
        assert df.iloc[:, 0].to_dict() == {"A": [1, 2, 3]}

    def test_with_nulls(self) -> None:
        df = ppd.DataFrame({"A": [1, None, 3], "B": [10, 20, None]})
        assert df.loc[:, "A"].to_dict() == {"A": [1, None, 3]}
        assert df.iloc[:, 1].to_dict() == {"B": [10, 20, None]}
