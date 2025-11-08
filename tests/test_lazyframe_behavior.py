import polars as pl
import pytest

import polarpandas as ppd


def test_lazyframe_repr_does_not_materialize(monkeypatch):
    df = ppd.DataFrame({"a": [1, 2], "b": [3, 4]})
    lf = df.lazy()

    def boom():
        raise AssertionError("lazy frame was materialized")

    monkeypatch.setattr(lf, "_materialize", boom)

    # repr should not trigger materialization
    output = repr(lf)
    assert "LazyFrame" in output


def test_lazyframe_len_and_height_require_materialization():
    lf = ppd.DataFrame({"a": [1, 2]}).lazy()
    with pytest.raises(TypeError):
        len(lf)
    with pytest.raises(TypeError):
        _ = lf.height


def test_lazyframe_shape_reports_unknown_rows():
    lf = ppd.DataFrame({"a": [1, 2], "b": [3, 4]}).lazy()
    assert lf.shape == (None, 2)


def test_lazyframe_info_uses_schema(capsys):
    lf = ppd.DataFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int64)}).lazy()
    lf.info()
    captured = capsys.readouterr()
    assert "row count unknown" in captured.out.lower()
    assert "a" in captured.out


def test_lazyframe_join_accepts_polars_inputs():
    left = ppd.DataFrame({"key": [1, 2], "a": [3, 4]}).lazy()
    right_pl = pl.DataFrame({"key": [1, 2], "b": [5, 6]}).lazy()
    result = left.join(right_pl, on="key", how="inner").collect()
    assert result.columns == ["key", "a", "b"]


def test_lazyframe_join_invalid_type():
    lf = ppd.DataFrame({"key": [1, 2]}).lazy()
    with pytest.raises(TypeError):
        lf.join({"key": [1, 2]}, on="key")  # type: ignore[arg-type]


def test_lazyframe_join_invalid_how():
    lf = ppd.DataFrame({"key": [1, 2]}).lazy()
    other = ppd.DataFrame({"key": [1, 2]}).lazy()
    with pytest.raises(ValueError):
        lf.join(other, on="key", how="unsupported")

