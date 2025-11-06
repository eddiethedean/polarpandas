# Migration Guide: pandas to PolarPandas

This guide helps you migrate your pandas code to PolarPandas, a high-performance pandas-compatible API built on Polars.

## Quick Migration

Most pandas code works with minimal changes:

```python
# Before (pandas)
import pandas as pd
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
result = df.groupby("A").sum()

# After (polarpandas)
import polarpandas as ppd
df = ppd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
result = df.groupby("A").sum()
```

## Key Differences

### 1. Import Statement

```python
# pandas
import pandas as pd

# polarpandas
import polarpandas as ppd
```

### 2. DataFrame Creation

PolarPandas supports the same DataFrame creation methods as pandas:

```python
# From dictionary
df = ppd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

# From list of dicts
df = ppd.DataFrame([{"A": 1, "B": 4}, {"A": 2, "B": 5}])

# From CSV
df = ppd.read_csv("data.csv")

# With index
df = ppd.DataFrame({"A": [1, 2]}, index=["x", "y"])
```

### 3. Column Operations

Column operations work the same way:

```python
# Access columns
df["A"]
df[["A", "B"]]

# Add new columns
df["C"] = df["A"] + df["B"]

# Delete columns
del df["C"]
df.drop("C", inplace=True)
```

### 4. Index Operations

Index is supported but stored separately (Polars is columnar):

```python
# Set index
df.set_index("A", inplace=True)

# Reset index
df.reset_index(inplace=True)

# Access index
df.index
```

### 5. GroupBy Operations

GroupBy works similarly, but uses Polars expressions:

```python
# pandas
df.groupby("A").agg({"B": "sum"})

# polarpandas (same syntax)
df.groupby("A").agg({"B": "sum"})

# Or use Polars expressions for more control
import polars as pl
df.groupby("A").agg(pl.col("B").sum())
```

### 6. Merging DataFrames

Merge operations work the same:

```python
# Inner join
result = df1.merge(df2, on="key")

# Left join
result = df1.merge(df2, on="key", how="left")

# Multiple keys
result = df1.merge(df2, on=["key1", "key2"])
```

### 7. String Operations

String accessor works the same:

```python
# All pandas string methods work
df["name"].str.upper()
df["name"].str.lower()
df["name"].str.contains("pattern")
```

### 8. Datetime Operations

Datetime accessor works the same:

```python
# All pandas datetime properties work
df["date"].dt.year
df["date"].dt.month
df["date"].dt.day
```

## Performance Tips

### 1. Use LazyFrame for Large Files

For large datasets, use lazy evaluation:

```python
# Eager (loads everything into memory)
df = ppd.read_csv("large_file.csv")

# Lazy (loads on demand)
lf = ppd.scan_csv("large_file.csv")
result = lf.filter(pl.col("A") > 100).collect()
```

### 2. Minimize Conversions

Avoid unnecessary pandas conversions:

```python
# Good - stays in Polars
df = ppd.DataFrame({"A": [1, 2, 3]})
result = df.groupby("A").sum()

# Avoid - unnecessary conversion
df_pandas = df.to_pandas()  # Only if you need pandas-specific features
```

### 3. Use Polars Expressions

For complex operations, use Polars expressions directly:

```python
import polars as pl

# More efficient than multiple operations
df.select([
    pl.col("A").sum().alias("A_sum"),
    pl.col("B").mean().alias("B_mean"),
])
```

## Common Gotchas

### 1. In-Place Operations

In-place operations work but return `None`:

```python
# This modifies df and returns None
df.sort_values("A", inplace=True)

# This returns a new DataFrame
df_sorted = df.sort_values("A")
```

### 2. Index Behavior

Index is stored separately and may behave slightly differently:

```python
# Index operations may be slower due to Polars' columnar architecture
df.index = ["x", "y", "z"]  # Works but not as fast as column operations
```

### 3. Missing Features

Some pandas features are not yet implemented:

```python
# These raise NotImplementedError with helpful messages
df.apply(axis=1)  # Row-wise apply not yet implemented
df.corr()  # Correlation not yet implemented
df.cov()  # Covariance not yet implemented
```

### 4. Error Messages

Error messages are enhanced with suggestions:

```python
# If you get an error, it will include:
# - Context about what went wrong
# - Suggestions for fixes
# - Workarounds if available
```

## Migration Checklist

- [ ] Replace `import pandas as pd` with `import polarpandas as ppd`
- [ ] Update DataFrame creation calls
- [ ] Test all groupby operations
- [ ] Verify merge/join operations
- [ ] Check string and datetime accessors
- [ ] Test I/O operations (read_csv, read_parquet, etc.)
- [ ] Verify index operations if used
- [ ] Test any custom functions that use pandas internals
- [ ] Run your test suite
- [ ] Benchmark performance improvements

## Performance Comparison

After migration, you should see significant performance improvements:

- **DataFrame Creation**: 14.1x faster
- **Read CSV**: 9.1x faster
- **Sorting**: 7.1x faster
- **GroupBy**: 3.3x faster
- **Filtering**: 3.0x faster

## Getting Help

If you encounter issues during migration:

1. Check the error message - it often includes helpful suggestions
2. Review the [Known Limitations](KNOWN_LIMITATIONS.md) document
3. Check if the feature is in the [TODO](TODO.md) list
4. Use `df.to_pandas()` as a temporary workaround if needed

## Example: Complete Migration

```python
# Before (pandas)
import pandas as pd

df = pd.read_csv("data.csv")
df["new_col"] = df["A"] * 2
df = df[df["B"] > 10]
result = df.groupby("C").agg({"A": "sum", "B": "mean"})
result.to_csv("output.csv")

# After (polarpandas)
import polarpandas as ppd

df = ppd.read_csv("data.csv")
df["new_col"] = df["A"] * 2
df = df[df["B"] > 10]
result = df.groupby("C").agg({"A": "sum", "B": "mean"})
result.to_csv("output.csv")

# Same code, 5-14x faster!
```

## Next Steps

1. Start with a small script or module
2. Run tests to verify correctness
3. Benchmark performance improvements
4. Gradually migrate larger codebases
5. Report any issues or missing features

Happy migrating! 🚀

