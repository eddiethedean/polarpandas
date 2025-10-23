"""
Example usage of PolarPandas - a pandas-like API on top of Polars.
"""
from polarpandas import DataFrame, Series

# Create a DataFrame
print("Creating a DataFrame:")
df = DataFrame({
    "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "age": [25, 30, 35, 28, 32],
    "city": ["NYC", "LA", "Chicago", "NYC", "LA"],
    "salary": [70000, 80000, 90000, 75000, 85000]
})
print(df)
print()

# Basic properties
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns}")
print()

# Add a new column (in-place mutation!)
print("Adding a bonus column:")
df["bonus"] = df["salary"] * 0.1
print(df)
print()

# Filter rows
print("Filtering ages > 28:")
filtered = df.filter(df["age"] > 28)
print(filtered)
print()

# Sort (in-place)
print("Sorting by salary:")
df.sort_values("salary", inplace=True)
print(df)
print()

# GroupBy operations
print("Average salary by city:")
grouped = df.groupby("city").agg(df["salary"].mean())
print(grouped)
print()

# Head and tail
print("First 3 rows:")
print(df.head(3))
print()

# Missing data handling
df2 = DataFrame({
    "a": [1, None, 3, 4],
    "b": [5, 6, None, 8]
})
print("DataFrame with missing values:")
print(df2)
print("\nFilling missing values with 0:")
df2.fillna(0, inplace=True)
print(df2)
print()

# Series operations
print("Series arithmetic:")
s1 = Series([1, 2, 3, 4, 5], name="numbers")
s2 = s1 + 10
print(f"Original: {s1.to_list()}")
print(f"Plus 10: {s2.to_list()}")
print()

# Copy (creates independent DataFrame)
print("Copy operation:")
df_copy = df.copy()
df_copy["test_column"] = 999
print(f"Original columns: {df.columns}")
print(f"Copy columns: {df_copy.columns}")
print()

print("✅ All operations completed successfully!")
print(f"Final DataFrame has {len(df)} rows and {len(df.columns)} columns")

