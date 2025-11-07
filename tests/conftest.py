"""Shared pytest fixtures for PolarPandas tests."""

from __future__ import annotations

import pytest

import polarpandas as ppd


@pytest.fixture
def mixed_schema_frame() -> ppd.DataFrame:
    """DataFrame fixture with diverse dtypes and nulls for reuse in tests."""

    return ppd.DataFrame(
        {
            "int_col": [1, 2, None],
            "float_col": [1.5, None, 3.5],
            "str_col": ["alpha", "beta", None],
            "bool_col": [True, False, None],
            "category_col": ["x", "y", "x"],
        }
    )


@pytest.fixture
def timedelta_string_series() -> ppd.Series:
    """Series fixture with human-readable timedelta strings."""

    return ppd.Series(["1 day", "2 days", "3 days"], name="delta")


@pytest.fixture
def timezone_datetime_strings() -> list[str]:
    """List of ISO8601 datetime strings with timezone offsets."""

    return [
        "2023-01-01T00:00:00+00:00",
        "2023-01-02T05:30:00+05:30",
        "2023-01-03T12:00:00-04:00",
    ]
