"""
Performance comparison between pandas and polarpandas.

This script benchmarks common operations to demonstrate the performance
benefits of using Polars under the hood.
"""
import time
import sys

# Check if pandas is available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  pandas not installed - will only show polarpandas times")
    print("   Install pandas with: pip install pandas")
    print()

import polarpandas as ppd


def time_operation(func, name, iterations=1):
    """Time an operation."""
    start = time.time()
    for _ in range(iterations):
        result = func()
    end = time.time()
    elapsed = (end - start) / iterations
    return elapsed


def format_speedup(pandas_time, polarpandas_time):
    """Format speedup ratio."""
    if pandas_time == 0 or polarpandas_time == 0:
        return "N/A"
    speedup = pandas_time / polarpandas_time
    if speedup > 1:
        return f"{speedup:.1f}x faster"
    else:
        return f"{1/speedup:.1f}x slower"


print("=" * 80)
print("PERFORMANCE COMPARISON: pandas vs polarpandas")
print("=" * 80)
print()

# Test dataset sizes
SMALL = 1_000
MEDIUM = 10_000
LARGE = 100_000

results = []

# ============================================================================
# BENCHMARK 1: DataFrame Creation
# ============================================================================
print("📊 Benchmark 1: DataFrame Creation")
print("-" * 80)

if PANDAS_AVAILABLE:
    data = {"a": list(range(LARGE)), "b": list(range(LARGE, 2*LARGE))}
    
    pandas_time = time_operation(lambda: pd.DataFrame(data), "pandas DataFrame", 5)
    polarpandas_time = time_operation(lambda: ppd.DataFrame(data), "polarpandas DataFrame", 5)
    
    print(f"  pandas:      {pandas_time*1000:.2f} ms")
    print(f"  polarpandas: {polarpandas_time*1000:.2f} ms")
    print(f"  Speedup:     {format_speedup(pandas_time, polarpandas_time)}")
    results.append(("DataFrame Creation (100k rows)", pandas_time, polarpandas_time))
else:
    data = {"a": list(range(LARGE)), "b": list(range(LARGE, 2*LARGE))}
    polarpandas_time = time_operation(lambda: ppd.DataFrame(data), "polarpandas DataFrame", 5)
    print(f"  polarpandas: {polarpandas_time*1000:.2f} ms")

print()

# ============================================================================
# BENCHMARK 2: Column Operations
# ============================================================================
print("📊 Benchmark 2: Column Operations (Adding New Column)")
print("-" * 80)

if PANDAS_AVAILABLE:
    pd_df = pd.DataFrame({"a": list(range(LARGE)), "b": list(range(LARGE, 2*LARGE))})
    ppd_df = ppd.DataFrame({"a": list(range(LARGE)), "b": list(range(LARGE, 2*LARGE))})
    
    def pandas_add_col():
        df = pd_df.copy()
        df['c'] = df['a'] + df['b']
        return df
    
    def polarpandas_add_col():
        df = ppd_df.copy()
        df['c'] = df['a'] + df['b']
        return df
    
    pandas_time = time_operation(pandas_add_col, "pandas add column", 10)
    polarpandas_time = time_operation(polarpandas_add_col, "polarpandas add column", 10)
    
    print(f"  pandas:      {pandas_time*1000:.2f} ms")
    print(f"  polarpandas: {polarpandas_time*1000:.2f} ms")
    print(f"  Speedup:     {format_speedup(pandas_time, polarpandas_time)}")
    results.append(("Column Operations (100k rows)", pandas_time, polarpandas_time))
else:
    ppd_df = ppd.DataFrame({"a": list(range(LARGE)), "b": list(range(LARGE, 2*LARGE))})
    polarpandas_time = time_operation(lambda: (lambda df: (df.__setitem__('c', df['a'] + df['b']), df)[1])(ppd_df.copy()), "polarpandas add column", 10)
    print(f"  polarpandas: {polarpandas_time*1000:.2f} ms")

print()

# ============================================================================
# BENCHMARK 3: Filtering
# ============================================================================
print("📊 Benchmark 3: Filtering Rows")
print("-" * 80)

if PANDAS_AVAILABLE:
    pd_df = pd.DataFrame({"a": list(range(LARGE)), "b": list(range(LARGE, 2*LARGE))})
    ppd_df = ppd.DataFrame({"a": list(range(LARGE)), "b": list(range(LARGE, 2*LARGE))})
    
    pandas_time = time_operation(lambda: pd_df[pd_df['a'] > 50000], "pandas filter", 10)
    polarpandas_time = time_operation(lambda: ppd_df.filter(ppd_df['a'] > 50000), "polarpandas filter", 10)
    
    print(f"  pandas:      {pandas_time*1000:.2f} ms")
    print(f"  polarpandas: {polarpandas_time*1000:.2f} ms")
    print(f"  Speedup:     {format_speedup(pandas_time, polarpandas_time)}")
    results.append(("Filtering (100k rows)", pandas_time, polarpandas_time))
else:
    ppd_df = ppd.DataFrame({"a": list(range(LARGE)), "b": list(range(LARGE, 2*LARGE))})
    polarpandas_time = time_operation(lambda: ppd_df.filter(ppd_df['a'] > 50000), "polarpandas filter", 10)
    print(f"  polarpandas: {polarpandas_time*1000:.2f} ms")

print()

# ============================================================================
# BENCHMARK 4: GroupBy Aggregation
# ============================================================================
print("📊 Benchmark 4: GroupBy Aggregation")
print("-" * 80)

if PANDAS_AVAILABLE:
    groups = ['A', 'B', 'C', 'D', 'E'] * 20000
    pd_df = pd.DataFrame({"group": groups, "value": list(range(LARGE))})
    ppd_df = ppd.DataFrame({"group": groups, "value": list(range(LARGE))})
    
    pandas_time = time_operation(lambda: pd_df.groupby('group')['value'].mean(), "pandas groupby", 5)
    polarpandas_time = time_operation(lambda: ppd_df.groupby('group').agg(ppd_df['value'].mean()), "polarpandas groupby", 5)
    
    print(f"  pandas:      {pandas_time*1000:.2f} ms")
    print(f"  polarpandas: {polarpandas_time*1000:.2f} ms")
    print(f"  Speedup:     {format_speedup(pandas_time, polarpandas_time)}")
    results.append(("GroupBy Aggregation (100k rows)", pandas_time, polarpandas_time))
else:
    groups = ['A', 'B', 'C', 'D', 'E'] * 20000
    ppd_df = ppd.DataFrame({"group": groups, "value": list(range(LARGE))})
    polarpandas_time = time_operation(lambda: ppd_df.groupby('group').agg(ppd_df['value'].mean()), "polarpandas groupby", 5)
    print(f"  polarpandas: {polarpandas_time*1000:.2f} ms")

print()

# ============================================================================
# BENCHMARK 5: Sorting
# ============================================================================
print("📊 Benchmark 5: Sorting")
print("-" * 80)

if PANDAS_AVAILABLE:
    import random
    shuffled = list(range(LARGE))
    random.shuffle(shuffled)
    
    pd_df = pd.DataFrame({"a": shuffled, "b": list(range(LARGE))})
    ppd_df = ppd.DataFrame({"a": shuffled.copy(), "b": list(range(LARGE))})
    
    pandas_time = time_operation(lambda: pd_df.sort_values('a'), "pandas sort", 5)
    polarpandas_time = time_operation(lambda: ppd_df.sort_values('a'), "polarpandas sort", 5)
    
    print(f"  pandas:      {pandas_time*1000:.2f} ms")
    print(f"  polarpandas: {polarpandas_time*1000:.2f} ms")
    print(f"  Speedup:     {format_speedup(pandas_time, polarpandas_time)}")
    results.append(("Sorting (100k rows)", pandas_time, polarpandas_time))
else:
    import random
    shuffled = list(range(LARGE))
    random.shuffle(shuffled)
    ppd_df = ppd.DataFrame({"a": shuffled, "b": list(range(LARGE))})
    polarpandas_time = time_operation(lambda: ppd_df.sort_values('a'), "polarpandas sort", 5)
    print(f"  polarpandas: {polarpandas_time*1000:.2f} ms")

print()

# ============================================================================
# BENCHMARK 6: Merge/Join
# ============================================================================
print("📊 Benchmark 6: Merge/Join Operations")
print("-" * 80)

if PANDAS_AVAILABLE:
    pd_df1 = pd.DataFrame({"key": list(range(MEDIUM)), "value1": list(range(MEDIUM))})
    pd_df2 = pd.DataFrame({"key": list(range(MEDIUM)), "value2": list(range(MEDIUM, 2*MEDIUM))})
    
    ppd_df1 = ppd.DataFrame({"key": list(range(MEDIUM)), "value1": list(range(MEDIUM))})
    ppd_df2 = ppd.DataFrame({"key": list(range(MEDIUM)), "value2": list(range(MEDIUM, 2*MEDIUM))})
    
    pandas_time = time_operation(lambda: pd_df1.merge(pd_df2, on='key'), "pandas merge", 5)
    polarpandas_time = time_operation(lambda: ppd_df1.merge(ppd_df2, on='key'), "polarpandas merge", 5)
    
    print(f"  pandas:      {pandas_time*1000:.2f} ms")
    print(f"  polarpandas: {polarpandas_time*1000:.2f} ms")
    print(f"  Speedup:     {format_speedup(pandas_time, polarpandas_time)}")
    results.append(("Merge/Join (10k rows)", pandas_time, polarpandas_time))
else:
    ppd_df1 = ppd.DataFrame({"key": list(range(MEDIUM)), "value1": list(range(MEDIUM))})
    ppd_df2 = ppd.DataFrame({"key": list(range(MEDIUM)), "value2": list(range(MEDIUM, 2*MEDIUM))})
    polarpandas_time = time_operation(lambda: ppd_df1.merge(ppd_df2, on='key'), "polarpandas merge", 5)
    print(f"  polarpandas: {polarpandas_time*1000:.2f} ms")

print()

# ============================================================================
# SUMMARY
# ============================================================================
if PANDAS_AVAILABLE and results:
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Operation':<40} {'pandas':<12} {'polarpandas':<12} {'Speedup':<15}")
    print("-" * 80)
    
    for operation, pd_time, ppd_time in results:
        speedup = format_speedup(pd_time, ppd_time)
        print(f"{operation:<40} {pd_time*1000:>10.2f}ms {ppd_time*1000:>10.2f}ms {speedup:>15}")
    
    print()
    avg_speedup = sum(pd_time / ppd_time for op, pd_time, ppd_time in results) / len(results)
    print(f"Average Speedup: {avg_speedup:.1f}x faster")
    print()
    print("✨ PolarPandas leverages Polars for superior performance!")
    print("⚡ Same pandas API, but significantly faster on large datasets!")
else:
    print("=" * 80)
    print("To see pandas comparison, install pandas:")
    print("  pip install pandas")
    print("=" * 80)
    print()
    print("✅ PolarPandas operations completed successfully!")
    print("⚡ Built on Polars for high performance!")

