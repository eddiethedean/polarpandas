#!/usr/bin/env python3
"""
Update the PolarPandas column in PANDAS_FUNCTION_MATRIX.md and pandas_api_matrix.json
based on actual implementation.

This script scans the polarpandas codebase to identify which functions and methods
are actually implemented and updates the matrix to reflect the actual implementation status.
"""

import json
import re
import sys
from pathlib import Path

# Add parent directory to path to allow importing polarpandas
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


def get_implemented_functions():
    """Get all implemented module-level functions from polarpandas."""
    functions = set()

    try:
        import polarpandas as ppd

        # Get all public callables from the module
        for name in dir(ppd):
            if not name.startswith("_"):
                attr = getattr(ppd, name)
                if callable(attr):
                    functions.add(name)
    except Exception as e:
        print(f"Warning: Could not check module functions: {e}", file=sys.stderr)

    return functions


def get_implemented_dataframe_methods():
    """Get all implemented DataFrame methods."""
    methods = set()

    try:
        from polarpandas.frame import DataFrame

        # Get all public methods from DataFrame
        for attr_name in dir(DataFrame):
            if not attr_name.startswith("_"):
                attr = getattr(DataFrame, attr_name)
                if callable(attr):
                    methods.add(attr_name)
    except Exception as e:
        print(f"Warning: Could not check DataFrame methods: {e}", file=sys.stderr)

    return methods


def get_implemented_series_methods():
    """Get all implemented Series methods."""
    methods = set()

    try:
        from polarpandas.series import Series

        # Get all public methods and properties from Series
        for attr_name in dir(Series):
            if not attr_name.startswith("_"):
                attr = getattr(Series, attr_name)
                # Include callable methods
                if callable(attr) or isinstance(attr, property):
                    methods.add(attr_name)
    except Exception as e:
        print(f"Warning: Could not check Series methods: {e}", file=sys.stderr)

    return methods


def get_implemented_index_methods():
    """Get all implemented Index methods."""
    methods = set()

    try:
        from polarpandas.index import Index

        # Get all public methods and properties from Index
        for attr_name in dir(Index):
            if not attr_name.startswith("_"):
                attr = getattr(Index, attr_name)
                # Include callable methods
                if callable(attr) or isinstance(attr, property):
                    methods.add(attr_name)
    except Exception as e:
        print(f"Warning: Could not check Index methods: {e}", file=sys.stderr)

    return methods


def get_implemented_str_accessor_methods():
    """Get all implemented .str accessor methods."""
    methods = set()

    try:
        from polarpandas.series import _StringAccessor

        # Get methods from _StringAccessor class directly
        for attr_name in dir(_StringAccessor):
            if not attr_name.startswith("_"):
                attr = getattr(_StringAccessor, attr_name)
                if callable(attr):
                    methods.add(attr_name)
    except Exception as e:
        print(f"Warning: Could not check .str accessor methods: {e}", file=sys.stderr)

    return methods


def get_implemented_dt_accessor_methods():
    """Get all implemented .dt accessor methods."""
    methods = set()

    try:
        from polarpandas.series import _DatetimeAccessor

        # Get methods from _DatetimeAccessor class directly
        for attr_name in dir(_DatetimeAccessor):
            if not attr_name.startswith("_"):
                attr = getattr(_DatetimeAccessor, attr_name)
                # Include both callable methods and properties
                if callable(attr) or isinstance(attr, property):
                    methods.add(attr_name)
    except Exception as e:
        print(f"Warning: Could not check .dt accessor methods: {e}", file=sys.stderr)

    return methods


def get_implemented_lazyframe_methods():
    """Get all implemented LazyFrame methods."""
    methods = set()

    try:
        import polars as pl

        from polarpandas.lazyframe import LazyFrame

        # Create a sample LazyFrame to check methods
        sample_lf = LazyFrame(pl.LazyFrame({"a": [1, 2, 3]}))

        # Get all public methods from LazyFrame
        for attr_name in dir(LazyFrame):
            if not attr_name.startswith("_"):
                attr = getattr(LazyFrame, attr_name)
                # Include callable methods and properties
                if callable(attr) or isinstance(attr, property):
                    methods.add(attr_name)

        # Also check methods available via __getattr__ delegation
        for attr_name in dir(sample_lf._df):
            if not attr_name.startswith("_") and attr_name not in methods:
                try:
                    attr = getattr(sample_lf._df, attr_name)
                    if callable(attr):
                        methods.add(attr_name)
                except Exception:
                    pass
    except Exception as e:
        print(f"Warning: Could not check LazyFrame methods: {e}", file=sys.stderr)

    return methods


def get_lazyframe_async_methods():
    """Get mapping of LazyFrame methods to their async versions."""
    async_map = {}

    try:
        import polars as pl

        # Create a sample LazyFrame to check for async methods
        sample_lf = pl.LazyFrame({"a": [1, 2, 3]})

        # Known async methods in Polars
        # Check which async methods exist
        async_methods = [
            m for m in dir(sample_lf) if "async" in m.lower() and not m.startswith("_")
        ]

        # Map base methods to their async versions
        for async_method in async_methods:
            # Extract base method name (e.g., "collect_async" -> "collect")
            if async_method.endswith("_async"):
                base_method = async_method[:-6]  # Remove "_async" suffix
                if hasattr(sample_lf, base_method):
                    async_map[base_method] = async_method

    except Exception as e:
        print(f"Warning: Could not check LazyFrame async methods: {e}", file=sys.stderr)

    return async_map


def get_pandas_dataframe_methods(json_file: Path) -> set:
    """Get all pandas DataFrame methods from the JSON matrix."""
    try:
        with open(json_file) as f:
            matrix = json.load(f)

        if "dataframe_methods" in matrix:
            return set(matrix["dataframe_methods"].keys())
    except Exception as e:
        print(f"Warning: Could not load pandas DataFrame methods: {e}", file=sys.stderr)

    return set()


def update_matrix_markdown(
    functions,
    df_methods,
    series_methods,
    index_methods,
    str_methods,
    dt_methods,
    lazyframe_methods,
    async_map,
    pandas_df_methods,
    matrix_file,
    json_file,
):
    """Update the markdown matrix file with implementation status"""
    with open(matrix_file) as f:
        lines = f.readlines()

    # Track changes
    updated_count = 0
    current_section = None

    # Map section headers to item sets
    section_map = {
        "## Module-level Functions": functions,
        "## DataFrame Methods": df_methods,
        "## Series Methods": series_methods,
        "## Index Methods": index_methods,
        "## String Accessor Methods (.str)": str_methods,
        "## Datetime Accessor Methods (.dt)": dt_methods,
        "## LazyFrame Methods": lazyframe_methods,
    }

    # Check if LazyFrame section exists
    has_lazyframe_section = any("## LazyFrame Methods" in line for line in lines)
    lazyframe_section_start = None
    lazyframe_section_end = None

    # Find LazyFrame section boundaries
    if has_lazyframe_section:
        for i, line in enumerate(lines):
            if "## LazyFrame Methods" in line:
                lazyframe_section_start = i
            elif (
                lazyframe_section_start is not None
                and line.startswith("## ")
                and "LazyFrame" not in line
            ):
                lazyframe_section_end = i
                break
        if lazyframe_section_end is None:
            lazyframe_section_end = len(lines)

    # Iterate through each line (skip LazyFrame section for now)
    for i, line in enumerate(lines):
        # Skip LazyFrame section - we'll rebuild it
        if (
            lazyframe_section_start is not None
            and lazyframe_section_start <= i < lazyframe_section_end
        ):
            continue

        # Detect which section we're in
        for section_header, item_set in section_map.items():
            if section_header in line:
                current_section = item_set
                continue

        # Check if this line is a function/method row
        # Format: | `item_name` | 1.5.0 | 2.0.0 | ... | PolarPandas | [Async] |
        match = re.match(r"^\|\s+`([^`]+)`\s+\|.*$", line)
        if match and current_section is not None:
            item_name = match.group(1)
            implemented = item_name in current_section

            if implemented:
                # Parse the line to get the PolarPandas column
                # Format: | `item` | ✅ | ✅ | ✅ | ✅ | ✅ | status | [async_status] |
                # Split by |: ['', ' `item` ', ' ✅ ', ..., ' status ', ' async_status ', '']
                original_parts = line.rstrip().split("|")

                # For other sections, standard format
                if (
                    len(original_parts) >= 8
                ):  # At least: empty, item, 5 versions, PolarPandas, empty
                    # Get the PolarPandas status (second to last part, index -2)
                    current_status = original_parts[-2].strip()

                    # Update if it's not already ✅
                    if current_status != "✅":
                        # Update the PolarPandas column
                        original_parts[-2] = " ✅ "
                        lines[i] = "|".join(original_parts) + "\n"
                        updated_count += 1
                        section_name = [
                            k for k, v in section_map.items() if v is current_section
                        ][0]
                        print(
                            f"Updated {section_name}: {item_name} (was: '{current_status}')"
                        )

    # Rebuild LazyFrame section
    if lazyframe_methods or pandas_df_methods or df_methods:
        # Remove old LazyFrame section if it exists
        if has_lazyframe_section and lazyframe_section_start is not None:
            # Remove old section
            lines = lines[:lazyframe_section_start] + lines[lazyframe_section_end:]
            updated_count += 1

        # Create union of all methods (pandas DataFrame, polarpandas DataFrame, and LazyFrame)
        all_methods = sorted(pandas_df_methods | df_methods | lazyframe_methods)
        total_count = len(all_methods)

        # Add new LazyFrame section at the end
        lines.append("\n## LazyFrame Methods\n\n")
        lines.append(f"Total items cataloged: {total_count}\n\n")
        lines.append(
            "| Item | pandas.DataFrame | polarpandas.DataFrame | polarpandas.LazyFrame |\n"
        )
        lines.append(
            "|------|------------------|----------------------|----------------------|\n"
        )

        # For each method, show status in all three columns
        for method_name in all_methods:
            has_pandas_df = method_name in pandas_df_methods
            has_polarpandas_df = method_name in df_methods
            has_lazyframe = method_name in lazyframe_methods
            pandas_status = "✅" if has_pandas_df else "❌"
            polarpandas_df_status = "✅" if has_polarpandas_df else "❌"
            lazyframe_status = "✅" if has_lazyframe else "❌"
            lines.append(
                f"| `{method_name}` | {pandas_status} | {polarpandas_df_status} | {lazyframe_status} |\n"
            )
            updated_count += 1

        if has_lazyframe_section:
            print(
                f"Rebuilt LazyFrame Methods section with {total_count} methods ({len(pandas_df_methods)} pandas, {len(df_methods)} polarpandas.DataFrame, {len(lazyframe_methods)} LazyFrame)"
            )
        else:
            print(
                f"Added LazyFrame Methods section with {total_count} methods ({len(pandas_df_methods)} pandas, {len(df_methods)} polarpandas.DataFrame, {len(lazyframe_methods)} LazyFrame)"
            )

    # Write back
    with open(matrix_file, "w") as f:
        f.writelines(lines)

    return updated_count


def update_matrix_json(
    functions,
    df_methods,
    series_methods,
    index_methods,
    str_methods,
    dt_methods,
    lazyframe_methods,
    async_map,
    pandas_df_methods,
    json_file,
):
    """Update the JSON matrix file with implementation status"""
    with open(json_file) as f:
        matrix = json.load(f)

    updated_count = 0

    category_map = {
        "functions": functions,
        "dataframe_methods": df_methods,
        "series_methods": series_methods,
        "index_methods": index_methods,
        "str_accessor_methods": str_methods,
        "dt_accessor_methods": dt_methods,
        "lazyframe_methods": lazyframe_methods,
    }

    # Update each category
    for category_key, item_set in category_map.items():
        if category_key not in matrix:
            # Initialize LazyFrame methods category if it doesn't exist
            if category_key == "lazyframe_methods":
                matrix[category_key] = {}
            else:
                continue

        # For LazyFrame methods, add all methods to the matrix if they don't exist
        if category_key == "lazyframe_methods":
            for method_name in item_set:
                has_pandas_df = method_name in pandas_df_methods
                has_async = method_name in async_map

                if method_name not in matrix[category_key]:
                    matrix[category_key][method_name] = {
                        "pandas_dataframe": has_pandas_df,
                        "polarpandas": True,
                        "has_async": has_async,
                    }
                    updated_count += 1
                    print(
                        f"Added JSON {category_key}: {method_name} (pandas: {has_pandas_df}, async: {has_async})"
                    )
                else:
                    # Update existing entry
                    if "polarpandas" not in matrix[category_key][
                        method_name
                    ] or not matrix[category_key][method_name].get(
                        "polarpandas", False
                    ):
                        matrix[category_key][method_name]["polarpandas"] = True
                        updated_count += 1
                        print(f"Updated JSON {category_key}: {method_name}")

                    # Update pandas DataFrame status
                    if (
                        "pandas_dataframe" not in matrix[category_key][method_name]
                        or matrix[category_key][method_name].get("pandas_dataframe")
                        != has_pandas_df
                    ):
                        matrix[category_key][method_name]["pandas_dataframe"] = (
                            has_pandas_df
                        )
                        updated_count += 1
                        print(
                            f"Updated JSON {category_key} pandas DataFrame status: {method_name} -> {has_pandas_df}"
                        )

                    # Update async status
                    if (
                        "has_async" not in matrix[category_key][method_name]
                        or matrix[category_key][method_name].get("has_async")
                        != has_async
                    ):
                        matrix[category_key][method_name]["has_async"] = has_async
                        updated_count += 1
                        print(
                            f"Updated JSON {category_key} async status: {method_name} -> {has_async}"
                        )
        else:
            # For other categories, standard update
            for item_name in matrix[category_key]:
                if item_name in item_set and (
                    "polarpandas" not in matrix[category_key][item_name]
                    or not matrix[category_key][item_name].get("polarpandas", False)
                ):
                    # Add polarpandas field if not present
                    matrix[category_key][item_name]["polarpandas"] = True
                    updated_count += 1
                    print(f"Updated JSON {category_key}: {item_name}")

    # Write back
    with open(json_file, "w") as f:
        json.dump(matrix, f, indent=2)

    return updated_count


def main():
    """Main entry point"""
    matrix_md = repo_root / "PANDAS_FUNCTION_MATRIX.md"
    matrix_json = repo_root / "pandas_api_matrix.json"

    print("Scanning polarpandas module-level functions...")
    functions = get_implemented_functions()
    print(f"Found {len(functions)} implemented functions")

    print("Scanning polarpandas DataFrame methods...")
    df_methods = get_implemented_dataframe_methods()
    print(f"Found {len(df_methods)} implemented DataFrame methods")

    print("Scanning polarpandas Series methods...")
    series_methods = get_implemented_series_methods()
    print(f"Found {len(series_methods)} implemented Series methods")

    print("Scanning polarpandas Index methods...")
    index_methods = get_implemented_index_methods()
    print(f"Found {len(index_methods)} implemented Index methods")

    print("Scanning polarpandas .str accessor methods...")
    str_methods = get_implemented_str_accessor_methods()
    print(f"Found {len(str_methods)} implemented .str accessor methods")

    print("Scanning polarpandas .dt accessor methods...")
    dt_methods = get_implemented_dt_accessor_methods()
    print(f"Found {len(dt_methods)} implemented .dt accessor methods")

    print("Scanning polarpandas LazyFrame methods...")
    lazyframe_methods = get_implemented_lazyframe_methods()
    print(f"Found {len(lazyframe_methods)} implemented LazyFrame methods")

    print("Scanning LazyFrame async methods...")
    async_map = get_lazyframe_async_methods()
    print(
        f"Found {len(async_map)} methods with async versions: {list(async_map.keys())}"
    )

    print("Loading pandas DataFrame methods from matrix...")
    pandas_df_methods = get_pandas_dataframe_methods(matrix_json)
    print(f"Found {len(pandas_df_methods)} pandas DataFrame methods in matrix")

    if matrix_md.exists():
        print(f"\nUpdating {matrix_md}...")
        updated_md = update_matrix_markdown(
            functions,
            df_methods,
            series_methods,
            index_methods,
            str_methods,
            dt_methods,
            lazyframe_methods,
            async_map,
            pandas_df_methods,
            matrix_md,
            matrix_json,
        )
        print(f"Updated {updated_md} entries in markdown")
    else:
        print(
            f"\nWarning: {matrix_md} does not exist. Run discover_pandas_api.py first."
        )

    if matrix_json.exists():
        print(f"\nUpdating {matrix_json}...")
        updated_json = update_matrix_json(
            functions,
            df_methods,
            series_methods,
            index_methods,
            str_methods,
            dt_methods,
            lazyframe_methods,
            async_map,
            pandas_df_methods,
            matrix_json,
        )
        print(f"Updated {updated_json} entries in JSON")
    else:
        print(
            f"\nWarning: {matrix_json} does not exist. Run discover_pandas_api.py first."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
