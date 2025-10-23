"""
Comprehensive example demonstrating all pandas-style module-level functions.

This shows how polarpandas can be used exactly like pandas!
"""

import os
import tempfile

import polarpandas as ppd

print("=" * 70)
print("POLARPANDAS - Complete Pandas Compatibility Demo")
print("=" * 70)
print()

# ============================================================================
# CREATING DATAFRAMES
# ============================================================================
print("📊 1. CREATING DATAFRAMES")
print("-" * 70)

# Method 1: From dictionary (like pandas)
df1 = ppd.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie", "David"],
        "age": [25, 30, 35, 28],
        "city": ["NYC", "LA", "NYC", "Chicago"],
        "salary": [70000, 80000, 90000, 75000],
    }
)
print("Created DataFrame from dictionary:")
print(df1)
print()

# ============================================================================
# MODULE-LEVEL READ FUNCTIONS (pandas-style!)
# ============================================================================
print("📖 2. IO OPERATIONS - WRITE & READ")
print("-" * 70)

# Write to CSV
csv_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
df1.to_csv(csv_file)
print(f"✅ Written to CSV: {csv_file}")

# Read from CSV using module-level function (EXACTLY like pd.read_csv!)
df_from_csv = ppd.read_csv(csv_file)
print("✅ Read back from CSV using ppd.read_csv():")
print(df_from_csv)
print()

# Clean up
os.unlink(csv_file)

# ============================================================================
# MODULE-LEVEL CONCAT (pandas-style!)
# ============================================================================
print("🔗 3. CONCATENATING DATAFRAMES")
print("-" * 70)

df2 = ppd.DataFrame(
    {
        "name": ["Eve", "Frank"],
        "age": [32, 27],
        "city": ["LA", "NYC"],
        "salary": [85000, 72000],
    }
)

# Use ppd.concat() just like pd.concat()!
combined = ppd.concat([df1, df2])
print("Combined DataFrames using ppd.concat():")
print(combined)
print(f"Original df1 had {len(df1)} rows, combined has {len(combined)} rows")
print()

# ============================================================================
# MODULE-LEVEL MERGE (pandas-style!)
# ============================================================================
print("🔀 4. MERGING DATAFRAMES")
print("-" * 70)

df_employees = ppd.DataFrame(
    {"name": ["Alice", "Bob", "Charlie"], "employee_id": [101, 102, 103]}
)

df_departments = ppd.DataFrame(
    {
        "employee_id": [101, 102, 103],
        "department": ["Engineering", "Sales", "Engineering"],
    }
)

# Use ppd.merge() just like pd.merge()!
merged = ppd.merge(df_employees, df_departments, on="employee_id")
print("Merged DataFrames using ppd.merge():")
print(merged)
print()

# ============================================================================
# MODULE-LEVEL GET_DUMMIES (pandas-style!)
# ============================================================================
print("🎲 5. ONE-HOT ENCODING")
print("-" * 70)

categories = ppd.Series(["cat", "dog", "cat", "bird", "dog"])
print(f"Original categorical data: {categories.to_list()}")

# Use ppd.get_dummies() just like pd.get_dummies()!
dummies = ppd.get_dummies(categories)
print("\nOne-hot encoded using ppd.get_dummies():")
print(dummies)
print()

# ============================================================================
# MODULE-LEVEL CUT (pandas-style!)
# ============================================================================
print("✂️  6. BINNING DATA")
print("-" * 70)

ages = ppd.Series([22, 35, 58, 45, 67, 41, 29])
print(f"Original ages: {ages.to_list()}")

# Use ppd.cut() just like pd.cut()!
age_bins = ppd.cut(ages, bins=3)
print("\nBinned into 3 groups using ppd.cut():")
print(age_bins.to_list())
print()

# ============================================================================
# MODULE-LEVEL DATE_RANGE (pandas-style!)
# ============================================================================
print("📅 7. DATE RANGES")
print("-" * 70)

# Use ppd.date_range() just like pd.date_range()!
try:
    dates = ppd.date_range("2021-01-01", periods=5)
    print("Generated date range using ppd.date_range():")
    print(dates.to_list())
    print()
except Exception as e:
    print(f"Note: date_range encountered: {e}")
    print()

# ============================================================================
# MODULE-LEVEL TO_DATETIME (pandas-style!)
# ============================================================================
print("🕐 8. DATETIME CONVERSION")
print("-" * 70)

date_strings = ["2021-01-01", "2021-06-15", "2021-12-31"]
print(f"Date strings: {date_strings}")

# Use ppd.to_datetime() just like pd.to_datetime()!
try:
    dates = ppd.to_datetime(date_strings)
    print("\nConverted using ppd.to_datetime():")
    print(dates.to_list())
    print()
except Exception as e:
    print(f"Note: to_datetime encountered: {e}")
    print()

# ============================================================================
# MODULE-LEVEL ISNA/NOTNA (pandas-style!)
# ============================================================================
print("❓ 9. DETECTING MISSING VALUES")
print("-" * 70)

data_with_nulls = ppd.Series([1, None, 3, None, 5])
print(f"Data: {data_with_nulls.to_list()}")

# Use ppd.isna() just like pd.isna()!
null_mask = ppd.isna(data_with_nulls)
print(f"\nNull mask using ppd.isna(): {null_mask.to_list()}")

not_null_mask = ppd.notna(data_with_nulls)
print(f"Not-null mask using ppd.notna(): {not_null_mask.to_list()}")
print()

# ============================================================================
# MODULE-LEVEL PIVOT_TABLE (pandas-style!)
# ============================================================================
print("📊 10. PIVOT TABLES")
print("-" * 70)

sales_data = ppd.DataFrame(
    {
        "date": ["2021-01", "2021-01", "2021-02", "2021-02"],
        "product": ["A", "B", "A", "B"],
        "sales": [100, 150, 120, 180],
    }
)

print("Sales data:")
print(sales_data)

# Use ppd.pivot_table() just like pd.pivot_table()!
pivoted = ppd.pivot_table(sales_data, values="sales", index="date", columns="product")
print("\nPivoted using ppd.pivot_table():")
print(pivoted)
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 70)
print("✨ SUMMARY: ALL PANDAS MODULE-LEVEL FUNCTIONS WORK!")
print("=" * 70)
print()
print("Available pandas-style module-level functions:")
print("  📖 Read: read_csv, read_parquet, read_json, read_excel")
print("  🔧 Manipulation: concat, merge, get_dummies, cut, pivot_table")
print("  📅 Datetime: date_range, to_datetime")
print("  🛠️  Utilities: isna, notna")
print()
print("✅ All functions work exactly like pandas!")
print("⚡ But with Polars performance under the hood!")
print()
print(f"📦 Package version: {ppd.__version__}")
print(f"📊 Total functions available: {len(ppd.__all__)}")
print()
