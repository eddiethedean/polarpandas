#!/usr/bin/env python3
"""
Discover pandas API availability across versions 1.5-2.3.

This script installs each pandas version sequentially and catalogs:
- All callables in pandas module (module-level functions)
- All public methods on pandas.DataFrame class
- All public methods on pandas.Series class
- All public methods on pandas.Index class
- All methods on .str and .dt accessors

Outputs:
- pandas_api_matrix.json - Machine-readable data
- PANDAS_FUNCTION_MATRIX.md - Human-readable markdown table
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

# Pandas versions to test
PANDAS_VERSIONS = ["1.5.0", "2.0.0", "2.1.0", "2.2.0", "2.3.0"]


def discover_api(pandas_version: str) -> Dict[str, List[str]]:
    """
    Install pandas version and catalog functions and methods.

    Args:
        pandas_version: Pandas version to install (e.g., "2.0.0")

    Returns:
        Dictionary with 'functions', 'dataframe_methods', 'series_methods',
        'index_methods', 'str_accessor_methods', and 'dt_accessor_methods' lists
    """
    print(f"\n{'=' * 80}")
    print(f"Discovering pandas {pandas_version} API...")
    print(f"{'=' * 80}")

    # Install specific pandas version
    print(f"Installing pandas=={pandas_version}...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", f"pandas=={pandas_version}"],
        check=True,
    )

    # Force reimport by removing from sys.modules
    modules_to_remove = [m for m in sys.modules if m.startswith("pandas")]
    for module in modules_to_remove:
        del sys.modules[module]

    # Import fresh
    import pandas as pd
    from pandas import DataFrame, Index, Series

    # Discover module-level functions
    functions = sorted(
        [
            name
            for name in dir(pd)
            if callable(getattr(pd, name, None))
            and not name.startswith("_")
            and not (name[0].isupper() if name else False)  # Exclude classes
        ]
    )

    # Discover DataFrame methods
    df_methods = sorted(
        [
            name
            for name in dir(DataFrame)
            if callable(getattr(DataFrame, name, None)) and not name.startswith("_")
        ]
    )

    # Discover Series methods
    series_methods = sorted(
        [
            name
            for name in dir(Series)
            if callable(getattr(Series, name, None)) and not name.startswith("_")
        ]
    )

    # Discover Index methods
    index_methods = sorted(
        [
            name
            for name in dir(Index)
            if callable(getattr(Index, name, None)) and not name.startswith("_")
        ]
    )

    # Discover .str accessor methods
    str_accessor_methods = []
    try:
        # Create a temporary Series with string data to access .str
        temp_series = Series(["test"])
        str_accessor = temp_series.str
        str_accessor_methods = sorted(
            [
                name
                for name in dir(str_accessor)
                if callable(getattr(str_accessor, name, None))
                and not name.startswith("_")
            ]
        )
    except Exception as e:
        print(f"Warning: Could not discover .str accessor methods: {e}")

    # Discover .dt accessor methods
    dt_accessor_methods = []
    try:
        # Create a temporary Series with datetime data to access .dt
        temp_series = Series(pd.date_range("2020-01-01", periods=1))
        dt_accessor = temp_series.dt
        dt_accessor_methods = sorted(
            [
                name
                for name in dir(dt_accessor)
                if callable(getattr(dt_accessor, name, None))
                and not name.startswith("_")
            ]
        )
    except Exception as e:
        print(f"Warning: Could not discover .dt accessor methods: {e}")

    print(f"Found {len(functions)} module-level functions")
    print(f"Found {len(df_methods)} DataFrame methods")
    print(f"Found {len(series_methods)} Series methods")
    print(f"Found {len(index_methods)} Index methods")
    print(f"Found {len(str_accessor_methods)} .str accessor methods")
    print(f"Found {len(dt_accessor_methods)} .dt accessor methods")

    return {
        "functions": functions,
        "dataframe_methods": df_methods,
        "series_methods": series_methods,
        "index_methods": index_methods,
        "str_accessor_methods": str_accessor_methods,
        "dt_accessor_methods": dt_accessor_methods,
    }


def build_matrix(
    versions_data: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Dict[str, Dict[str, bool]]]:
    """
    Build a matrix showing which items exist in which versions.

    Args:
        versions_data: Dict mapping version -> {functions: [...], ...}

    Returns:
        Dict with matrices for each category
    """
    # Collect all unique items across all versions for each category
    categories = [
        "functions",
        "dataframe_methods",
        "series_methods",
        "index_methods",
        "str_accessor_methods",
        "dt_accessor_methods",
    ]

    matrices = {}
    for category in categories:
        all_items: Set[str] = set()
        for data in versions_data.values():
            all_items.update(data.get(category, []))

        category_matrix = {}
        for item in sorted(all_items):
            category_matrix[item] = {
                version: item in data.get(category, [])
                for version, data in versions_data.items()
            }

        matrices[category] = category_matrix

    return matrices


def check_polarpandas_availability(item_name: str, item_type: str) -> bool:
    """
    Check if a function or method is available in polarpandas.

    Args:
        item_name: Name of function or method
        item_type: Type of item ('function', 'dataframe_method', 'series_method',
                  'index_method', 'str_accessor_method', 'dt_accessor_method')

    Returns:
        True if available in polarpandas
    """
    try:
        if item_type == "function":
            import polarpandas as ppd

            return hasattr(ppd, item_name)
        elif item_type == "dataframe_method":
            from polarpandas.frame import DataFrame

            return hasattr(DataFrame, item_name)
        elif item_type == "series_method":
            from polarpandas.series import Series

            return hasattr(Series, item_name)
        elif item_type == "index_method":
            from polarpandas.index import Index

            return hasattr(Index, item_name)
        elif item_type == "str_accessor_method":
            import polars as pl

            from polarpandas.series import Series

            # Create a temporary Series to access .str
            temp_series = Series(pl.Series(["test"]))
            if hasattr(temp_series, "str"):
                str_accessor = temp_series.str
                return hasattr(str_accessor, item_name)
            return False
        elif item_type == "dt_accessor_method":
            import polars as pl

            from polarpandas.series import Series

            # Create a temporary Series with datetime to access .dt
            temp_series = Series(pl.Series([pl.datetime(2020, 1, 1)]))
            if hasattr(temp_series, "dt"):
                dt_accessor = temp_series.dt
                return hasattr(dt_accessor, item_name)
            return False
    except Exception:
        return False

    return False


def save_json(matrix: Dict, output_path: Path) -> None:
    """Save matrix as JSON."""
    print(f"\nSaving JSON to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"✓ Saved {output_path}")


def save_markdown(matrix: Dict, output_path: Path, versions: List[str]) -> None:
    """Save matrix as markdown table."""
    print(f"\nGenerating markdown table at {output_path}...")

    lines = [
        "# Pandas Function & Method Availability Matrix",
        "",
        "**Generated by:** `scripts/discover_pandas_api.py`",
        f"**Pandas Versions Tested:** {', '.join(versions)}",
        "",
        "This matrix shows which functions and methods are available in each pandas version,",
        "and whether they are implemented in polarpandas.",
        "",
        "**Note:** All polarpandas features are available to everyone - there is no version gating.",
        "This matrix is provided for reference to understand pandas version compatibility.",
        "",
        "## Legend",
        "",
        "- ✅ = Available in pandas version (or implemented in polarpandas)",
        "- ⚠️ = Deprecated in pandas (available but modern alternative recommended)",
        "- ❌ = Not available in pandas version (or not implemented in polarpandas)",
        "",
        "**PolarPandas Status:** All implemented features (marked with ✅) are available to all users regardless of pandas version.",
        "",
    ]

    # Category names and their display names
    categories = [
        ("functions", "Module-level Functions"),
        ("dataframe_methods", "DataFrame Methods"),
        ("series_methods", "Series Methods"),
        ("index_methods", "Index Methods"),
        ("str_accessor_methods", "String Accessor Methods (.str)"),
        ("dt_accessor_methods", "Datetime Accessor Methods (.dt)"),
    ]

    item_type_map = {
        "functions": "function",
        "dataframe_methods": "dataframe_method",
        "series_methods": "series_method",
        "index_methods": "index_method",
        "str_accessor_methods": "str_accessor_method",
        "dt_accessor_methods": "dt_accessor_method",
    }

    for category_key, category_name in categories:
        if category_key not in matrix:
            continue

        lines.extend(
            [
                f"## {category_name}",
                "",
                f"Total items cataloged: {len(matrix[category_key])}",
                "",
            ]
        )

        # Table header
        header = "| Item | " + " | ".join(versions) + " | PolarPandas |"
        separator = (
            "|"
            + "|".join(
                ["-" * (len(v) + 2) for v in ["Item"] + versions + ["PolarPandas"]]
            )
            + "|"
        )
        lines.extend([header, separator])

        # Table rows
        item_type = item_type_map[category_key]
        for item_name, availability in sorted(matrix[category_key].items()):
            polarpandas_available = check_polarpandas_availability(item_name, item_type)
            row = f"| `{item_name}` |"
            for version in versions:
                if availability.get(version, False):
                    row += " ✅ |"
                else:
                    row += " ❌ |"
            row += " ✅ |" if polarpandas_available else " ❌ |"
            lines.append(row)

        lines.append("")  # Empty line between sections

    # Summary statistics
    lines.extend(
        [
            "## Summary Statistics",
            "",
        ]
    )

    # Count items per version
    for version in versions:
        counts = {}
        for category_key in matrix:
            count = sum(
                1
                for avail in matrix[category_key].values()
                if avail.get(version, False)
            )
            counts[category_key] = count

        total = sum(counts.values())
        func_count = counts.get("functions", 0)
        df_count = counts.get("dataframe_methods", 0)
        series_count = counts.get("series_methods", 0)
        index_count = counts.get("index_methods", 0)
        str_count = counts.get("str_accessor_methods", 0)
        dt_count = counts.get("dt_accessor_methods", 0)

        lines.append(
            f"- **pandas {version}**: {func_count} functions, {df_count} DataFrame methods, "
            f"{series_count} Series methods, {index_count} Index methods, "
            f"{str_count} .str methods, {dt_count} .dt methods (total: {total} items)"
        )

    # PolarPandas coverage
    polarpandas_counts = {}
    for category_key, item_type in item_type_map.items():
        if category_key not in matrix:
            continue
        count = sum(
            1
            for item in matrix[category_key]
            if check_polarpandas_availability(item, item_type)
        )
        polarpandas_counts[category_key] = count

    total_ppd = sum(polarpandas_counts.values())
    func_count_ppd = polarpandas_counts.get("functions", 0)
    df_count_ppd = polarpandas_counts.get("dataframe_methods", 0)
    series_count_ppd = polarpandas_counts.get("series_methods", 0)
    index_count_ppd = polarpandas_counts.get("index_methods", 0)
    str_count_ppd = polarpandas_counts.get("str_accessor_methods", 0)
    dt_count_ppd = polarpandas_counts.get("dt_accessor_methods", 0)

    lines.append(
        f"- **PolarPandas**: {func_count_ppd} functions, {df_count_ppd} DataFrame methods, "
        f"{series_count_ppd} Series methods, {index_count_ppd} Index methods, "
        f"{str_count_ppd} .str methods, {dt_count_ppd} .dt methods (total: {total_ppd} items)"
    )

    lines.append("")
    lines.append(
        "**All polarpandas features are available to everyone** - there is no version gating or compatibility restrictions."
    )
    lines.append(
        "This provides comprehensive pandas compatibility across all supported versions!"
    )

    # Write file
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"✓ Saved {output_path}")


def main() -> None:
    """Main discovery process."""
    print("=" * 80)
    print("Pandas API Discovery Tool")
    print("=" * 80)

    # Discover API for each version
    versions_data = {}
    for version in PANDAS_VERSIONS:
        versions_data[version] = discover_api(version)

    # Build matrix
    print(f"\n{'=' * 80}")
    print("Building API matrix...")
    print(f"{'=' * 80}")
    matrix = build_matrix(versions_data)

    # Save outputs
    repo_root = Path(__file__).parent.parent
    json_path = repo_root / "pandas_api_matrix.json"
    md_path = repo_root / "PANDAS_FUNCTION_MATRIX.md"

    save_json(matrix, json_path)
    save_markdown(matrix, md_path, PANDAS_VERSIONS)

    print(f"\n{'=' * 80}")
    print("✓ Discovery complete!")
    print(f"{'=' * 80}")
    print("\nGenerated files:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")
    print("\nTotal items discovered:")
    for category, items in matrix.items():
        print(f"  - {len(items)} {category}")


if __name__ == "__main__":
    main()
