"""
Helper utilities for testing polarpandas without pandas dependency.

This module provides custom assertion functions to verify polarpandas
outputs match expected values without requiring pandas for comparison.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Union


def load_expected(test_file: str, test_name: str) -> Dict[str, Any]:
    """
    Load expected values from JSON file.

    Parameters
    ----------
    test_file : str
        Name of test file (without .py extension)
    test_name : str
        Name of the test function

    Returns
    -------
    dict
        Expected values for the test
    """
    expected_dir = Path(__file__).parent / "expected_values"
    json_file = expected_dir / f"{test_file}.json"

    if not json_file.exists():
        raise FileNotFoundError(
            f"Expected values file not found: {json_file}\n"
            f"Run scripts/generate_test_expected_values.py to generate it."
        )

    with open(json_file) as f:
        data = json.load(f)

    if test_name not in data:
        raise KeyError(
            f"Test '{test_name}' not found in {json_file}.\n"
            f"Available tests: {list(data.keys())}"
        )

    return data[test_name]  # type: ignore[no-any-return]


def assert_frame_equal(
    ppd_df: Any,
    expected: Union[Dict[str, List[Any]], List[Dict[str, Any]]],
    check_order: bool = True,
    check_dtype: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Assert polarpandas DataFrame matches expected values.

    Parameters
    ----------
    ppd_df : DataFrame
        polarpandas DataFrame to check
    expected : dict or list
        Expected values as dict of lists (column-oriented) or list of dicts (row-oriented)
    check_order : bool, default True
        Whether to check row order
    check_dtype : bool, default False
        Whether to check data types (not implemented yet)
    rtol : float, default 1e-5
        Relative tolerance for floating point comparison
    atol : float, default 1e-8
        Absolute tolerance for floating point comparison
    """
    # Convert DataFrame to dict
    result_dict = ppd_df.to_dict()

    # Remove _index column if it exists (internal column)
    if "_index" in result_dict:
        del result_dict["_index"]

    # Handle list of dicts format (row-oriented)
    if isinstance(expected, list):
        # Convert to column-oriented dict
        expected_dict = {}
        if expected:
            for key in expected[0]:
                expected_dict[key] = [row[key] for row in expected]
        expected = expected_dict

    # Check columns match
    result_cols = set(result_dict.keys())
    expected_cols = set(expected.keys())

    if result_cols != expected_cols:
        raise AssertionError(
            f"Columns don't match.\n"
            f"Expected: {sorted(expected_cols)}\n"
            f"Got: {sorted(result_cols)}\n"
            f"Missing: {sorted(expected_cols - result_cols)}\n"
            f"Extra: {sorted(result_cols - expected_cols)}"
        )

    # Check values for each column
    for col in expected:
        result_values = result_dict[col]
        expected_values = expected[col]

        if len(result_values) != len(expected_values):
            raise AssertionError(
                f"Column '{col}' length mismatch.\n"
                f"Expected: {len(expected_values)}\n"
                f"Got: {len(result_values)}"
            )

        if check_order:
            _assert_values_equal(result_values, expected_values, col, rtol, atol)
        else:
            # Sort both lists for comparison
            _assert_values_equal(
                sorted(result_values, key=_sort_key),
                sorted(expected_values, key=_sort_key),
                col,
                rtol,
                atol,
            )


def assert_series_equal(
    ppd_series: Any,
    expected: List[Any],
    check_order: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Assert polarpandas Series matches expected list.

    Parameters
    ----------
    ppd_series : Series
        polarpandas Series to check
    expected : list
        Expected values
    check_order : bool, default True
        Whether to check element order
    rtol : float, default 1e-5
        Relative tolerance for floating point comparison
    atol : float, default 1e-8
        Absolute tolerance for floating point comparison
    """
    result = ppd_series.tolist()

    if len(result) != len(expected):
        raise AssertionError(
            f"Series length mismatch.\nExpected: {len(expected)}\nGot: {len(result)}"
        )

    if check_order:
        _assert_values_equal(result, expected, "Series", rtol, atol)
    else:
        _assert_values_equal(
            sorted(result, key=_sort_key),
            sorted(expected, key=_sort_key),
            "Series",
            rtol,
            atol,
        )


def assert_index_equal(
    ppd_index: Any,
    expected: List[Any],
    check_order: bool = True,
) -> None:
    """
    Assert polarpandas Index matches expected list.

    Parameters
    ----------
    ppd_index : Index
        polarpandas Index to check
    expected : list
        Expected values
    check_order : bool, default True
        Whether to check element order
    """
    result = list(ppd_index)

    if len(result) != len(expected):
        raise AssertionError(
            f"Index length mismatch.\nExpected: {len(expected)}\nGot: {len(result)}"
        )

    if check_order:
        assert result == expected, (
            f"Index values don't match.\nExpected: {expected}\nGot: {result}"
        )
    else:
        assert sorted(result, key=_sort_key) == sorted(expected, key=_sort_key), (
            f"Index values don't match (order ignored).\n"
            f"Expected: {sorted(expected, key=_sort_key)}\n"
            f"Got: {sorted(result, key=_sort_key)}"
        )


def _assert_values_equal(
    result: List[Any],
    expected: List[Any],
    label: str,
    rtol: float,
    atol: float,
) -> None:
    """
    Assert two lists of values are equal, handling floats with tolerance.

    Parameters
    ----------
    result : list
        Actual values
    expected : list
        Expected values
    label : str
        Label for error messages
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    """
    for i, (res_val, exp_val) in enumerate(zip(result, expected)):
        # Handle None values
        if res_val is None and exp_val is None:
            continue
        if res_val is None or exp_val is None:
            raise AssertionError(
                f"{label}[{i}] mismatch.\nExpected: {exp_val}\nGot: {res_val}"
            )

        # Handle floating point with tolerance
        if isinstance(exp_val, float) or isinstance(res_val, float):
            if not _isclose(res_val, exp_val, rtol, atol):
                raise AssertionError(
                    f"{label}[{i}] mismatch (float comparison with rtol={rtol}, atol={atol}).\n"
                    f"Expected: {exp_val}\n"
                    f"Got: {res_val}\n"
                    f"Diff: {abs(res_val - exp_val)}"
                )
        else:
            # Exact comparison for non-floats
            if res_val != exp_val:
                raise AssertionError(
                    f"{label}[{i}] mismatch.\nExpected: {exp_val}\nGot: {res_val}"
                )


def _isclose(a: float, b: float, rtol: float, atol: float) -> bool:
    """Check if two floats are close within tolerance."""
    # Handle NaN
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False

    # Handle infinity
    if math.isinf(a) and math.isinf(b):
        return a == b  # Both must be same infinity
    if math.isinf(a) or math.isinf(b):
        return False

    # Regular comparison
    return abs(a - b) <= atol + rtol * abs(b)


def _sort_key(x: Any) -> Any:
    """
    Generate a sort key for mixed type values.

    Handles None values and ensures stable sorting.
    """
    if x is None:
        return (0, 0)  # None sorts first
    elif isinstance(x, (int, float)):
        return (1, x)
    elif isinstance(x, str):
        return (2, x)
    else:
        return (3, str(x))


def assert_dict_equal(result: Dict[str, Any], expected: Dict[str, Any]) -> None:
    """
    Assert two dictionaries are equal.

    Parameters
    ----------
    result : dict
        Actual dictionary
    expected : dict
        Expected dictionary
    """
    if result.keys() != expected.keys():
        raise AssertionError(
            f"Dictionary keys don't match.\n"
            f"Expected: {sorted(expected.keys())}\n"
            f"Got: {sorted(result.keys())}"
        )

    for key in expected:
        if result[key] != expected[key]:
            raise AssertionError(
                f"Dictionary['{key}'] mismatch.\n"
                f"Expected: {expected[key]}\n"
                f"Got: {result[key]}"
            )
