"""
Generate expected values for tests using pandas.

This script runs once to generate expected output values from pandas operations,
which are then stored as JSON files. Tests can then use these expected values
without requiring pandas as a runtime dependency.

Usage:
    python scripts/generate_test_expected_values.py

Requirements:
    - pandas must be installed to run this script
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required to generate expected values")
    print("Install with: pip install pandas")
    sys.exit(1)


def generate_dataframe_statistical_expected() -> Dict[str, Any]:
    """Generate expected values for test_dataframe_statistical.py"""
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50],
        "C": [1.1, 2.2, 3.3, 4.4, 5.5],
    }

    expected = {}

    # test_nlargest_basic
    result = pd.DataFrame(data).nlargest(3, "A")
    expected["test_nlargest_basic"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    # test_nlargest_multiple_columns
    result = pd.DataFrame(data).nlargest(3, ["A", "B"])
    expected["test_nlargest_multiple_columns"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    # test_nsmallest_basic
    result = pd.DataFrame(data).nsmallest(3, "A")
    expected["test_nsmallest_basic"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    # test_nsmallest_multiple_columns
    result = pd.DataFrame(data).nsmallest(3, ["A", "B"])
    expected["test_nsmallest_multiple_columns"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    # test_corr_basic
    result = pd.DataFrame(data).corr()
    expected["test_corr_basic"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    # test_corr_method
    result = pd.DataFrame(data).corr(method="pearson")
    expected["test_corr_method"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    # test_cov_basic
    result = pd.DataFrame(data).cov()
    expected["test_cov_basic"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    # test_rank_basic
    result = pd.DataFrame(data).rank()
    expected["test_rank_basic"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    # test_rank_method
    result = pd.DataFrame(data).rank(method="min")
    expected["test_rank_method"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    # test_rank_numeric_only
    data_with_str = data.copy()
    data_with_str["D"] = ["a", "b", "c", "d", "e"]
    result = pd.DataFrame(data_with_str).rank(numeric_only=True)
    expected["test_rank_numeric_only"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    # test_diff_basic
    result = pd.DataFrame(data).diff()
    expected["test_diff_basic"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    # test_diff_periods
    result = pd.DataFrame(data).diff(periods=2)
    expected["test_diff_periods"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    # test_pct_change_basic
    result = pd.DataFrame(data).pct_change()
    expected["test_pct_change_basic"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    # test_pct_change_periods
    result = pd.DataFrame(data).pct_change(periods=2)
    expected["test_pct_change_periods"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    # test_cumsum_basic
    result = pd.DataFrame(data).cumsum()
    expected["test_cumsum_basic"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    # test_cumprod_basic
    result = pd.DataFrame(data).cumprod()
    expected["test_cumprod_basic"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    # test_cummax_basic
    result = pd.DataFrame(data).cummax()
    expected["test_cummax_basic"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    # test_cummin_basic
    result = pd.DataFrame(data).cummin()
    expected["test_cummin_basic"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    # test_statistical_with_nulls
    data_with_nulls = {
        "A": [1, None, 3, 4, 5],
        "B": [10, 20, None, 40, 50],
        "C": [1.1, 2.2, 3.3, None, 5.5],
    }
    result = pd.DataFrame(data_with_nulls).corr()
    expected["test_statistical_with_nulls"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    # test_diff_basic
    result = pd.DataFrame(data).diff()
    # Convert NaN to None for JSON serialization
    result_dict = result.to_dict(orient="list")  # type: ignore[attr-defined]
    for key in result_dict:
        result_dict[key] = [None if pd.isna(v) else v for v in result_dict[key]]
    expected["test_diff_basic"] = result_dict

    # test_diff_periods
    result = pd.DataFrame(data).diff(periods=2)
    result_dict = result.to_dict(orient="list")  # type: ignore[attr-defined]
    for key in result_dict:
        result_dict[key] = [None if pd.isna(v) else v for v in result_dict[key]]
    expected["test_diff_periods"] = result_dict

    # test_pct_change_basic
    result = pd.DataFrame(data).pct_change()
    result_dict = result.to_dict(orient="list")  # type: ignore[attr-defined]
    for key in result_dict:
        result_dict[key] = [None if pd.isna(v) else v for v in result_dict[key]]
    expected["test_pct_change_basic"] = result_dict

    # test_pct_change_periods
    result = pd.DataFrame(data).pct_change(periods=2)
    result_dict = result.to_dict(orient="list")  # type: ignore[attr-defined]
    for key in result_dict:
        result_dict[key] = [None if pd.isna(v) else v for v in result_dict[key]]
    expected["test_pct_change_periods"] = result_dict

    # test_statistical_single_row
    data_single = {"A": [1], "B": [10], "C": [1.1]}
    result = pd.DataFrame(data_single).nlargest(1, "A")
    expected["test_statistical_single_row"] = result.to_dict(orient="list")  # type: ignore[attr-defined]

    return expected


def main() -> None:
    """Generate all expected values and save to JSON files."""
    print("Generating expected values from pandas...")

    # Create expected_values directory
    expected_dir = Path(__file__).parent.parent / "tests" / "expected_values"
    expected_dir.mkdir(exist_ok=True)
    print(f"Output directory: {expected_dir}")

    # Generate for each test file
    generators = {
        "test_dataframe_statistical": generate_dataframe_statistical_expected,
    }

    for test_file, generator in generators.items():
        print(f"\nGenerating {test_file}.json...")
        try:
            expected_values = generator()
            output_file = expected_dir / f"{test_file}.json"

            with open(output_file, "w") as f:
                json.dump(expected_values, f, indent=2)

            print(f"  ✓ Generated {len(expected_values)} test cases")
            print(f"  ✓ Saved to {output_file}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 50)
    print("Expected values generation complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
