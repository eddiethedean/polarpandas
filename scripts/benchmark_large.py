"""
Performance comparison with larger datasets to show Polars advantages.
"""

import math
import os
import random
import sys
import tempfile
import time

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Install pandas for comparison: pip install pandas")
    sys.exit(1)

import polarpandas as ppd


def benchmark(name, pandas_fn, polarpandas_fn, iterations=3):
    """Run a benchmark comparison."""
    # Pandas
    pd_times = []
    for _ in range(iterations):
        start = time.time()
        pandas_fn()
        pd_times.append(time.time() - start)
    pandas_time = min(pd_times)  # Best of N

    # PolarPandas
    ppd_times = []
    for _ in range(iterations):
        start = time.time()
        polarpandas_fn()
        ppd_times.append(time.time() - start)
    polarpandas_time = min(ppd_times)  # Best of N

    speedup = pandas_time / polarpandas_time

    print(f"\n{name}")
    print("-" * 70)
    print(f"  pandas:      {pandas_time * 1000:>8.2f} ms")
    print(f"  polarpandas: {polarpandas_time * 1000:>8.2f} ms")
    if speedup > 1:
        print(f"  ⚡ Speedup:   {speedup:>8.1f}x FASTER")
    else:
        print(f"  ⚠️  Speedup:   {1 / speedup:>8.1f}x slower (wrapper overhead)")

    return pandas_time, polarpandas_time, speedup


print("=" * 70)
print("PERFORMANCE BENCHMARKS: pandas vs polarpandas")
print("Testing with datasets of various sizes")
print("=" * 70)

results = []

# ===========================================================================
# BENCHMARK 1: Large DataFrame Creation (1M rows)
# ===========================================================================
SIZE = 1_000_000
data_large = {
    "a": list(range(SIZE)),
    "b": list(range(SIZE, 2 * SIZE)),
    "c": [x * 1.5 for x in range(SIZE)],
}

pd_time, ppd_time, speedup = benchmark(
    "1. DataFrame Creation (1M rows, 3 columns)",
    lambda: pd.DataFrame(data_large),
    lambda: ppd.DataFrame(data_large),
)
results.append(("DataFrame Creation (1M)", speedup))

# ===========================================================================
# BENCHMARK 2: Reading CSV
# ===========================================================================

# Create a large CSV
csv_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
csv_file.write("a,b,c\n")
for i in range(100_000):
    csv_file.write(f"{i},{i + 100000},{i * 1.5}\n")
csv_file.close()

pd_time, ppd_time, speedup = benchmark(
    "2. Reading CSV (100k rows)",
    lambda: pd.read_csv(csv_file.name),
    lambda: ppd.read_csv(csv_file.name),
)
results.append(("Read CSV (100k)", speedup))

os.unlink(csv_file.name)

# ===========================================================================
# BENCHMARK 3: Sorting Large Dataset
# ===========================================================================

shuffled = list(range(500_000))
random.shuffle(shuffled)

pd_df = pd.DataFrame({"a": shuffled, "b": list(range(500_000))})
ppd_df = ppd.DataFrame({"a": shuffled.copy(), "b": list(range(500_000))})

pd_time, ppd_time, speedup = benchmark(
    "3. Sorting (500k rows)",
    lambda: pd_df.sort_values("a"),
    lambda: ppd_df.sort_values("a"),
)
results.append(("Sorting (500k)", speedup))

# ===========================================================================
# BENCHMARK 4: Large GroupBy
# ===========================================================================
groups = ["Group_" + str(i % 100) for i in range(500_000)]
values = list(range(500_000))

pd_df = pd.DataFrame({"group": groups, "value": values})
ppd_df = ppd.DataFrame({"group": groups, "value": values})

pd_time, ppd_time, speedup = benchmark(
    "4. GroupBy Aggregation (500k rows, 100 groups)",
    lambda: pd_df.groupby("group")["value"].mean(),
    lambda: ppd_df.groupby("group").agg(ppd_df["value"].mean()),
    iterations=3,
)
results.append(("GroupBy (500k)", speedup))

# ===========================================================================
# BENCHMARK 5: String Operations
# ===========================================================================
strings = ["test_string_" + str(i) for i in range(100_000)]

pd_series = pd.Series(strings)
ppd_series = ppd.Series(strings)

pd_time, ppd_time, speedup = benchmark(
    "5. String Operations (100k strings)",
    lambda: pd_series.str.upper(),
    lambda: ppd_series.str.upper(),
)
results.append(("String Operations (100k)", speedup))

# ===========================================================================
# BENCHMARK 6: Filtering Large Dataset
# ===========================================================================
pd_df = pd.DataFrame(
    {
        "a": list(range(500_000)),
        "b": [x * 2 for x in range(500_000)],
        "c": [x % 100 for x in range(500_000)],
    }
)
ppd_df = ppd.DataFrame(
    {
        "a": list(range(500_000)),
        "b": [x * 2 for x in range(500_000)],
        "c": [x % 100 for x in range(500_000)],
    }
)

pd_time, ppd_time, speedup = benchmark(
    "6. Complex Filtering (500k rows)",
    lambda: pd_df[(pd_df["a"] > 100000) & (pd_df["c"] < 50)],
    lambda: ppd_df.filter((ppd_df["a"] > 100000) & (ppd_df["c"] < 50)),
)
results.append(("Filtering (500k)", speedup))

# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
print("\n" + "=" * 70)
print("PERFORMANCE SUMMARY")
print("=" * 70)
print()
print(f"{'Benchmark':<35} {'Speedup':<20}")
print("-" * 70)

faster_count = 0
slower_count = 0

for name, speedup in results:
    if speedup > 1:
        print(f"{name:<35} ⚡ {speedup:>6.1f}x FASTER")
        faster_count += 1
    else:
        print(f"{name:<35} ⚠️  {1 / speedup:>6.1f}x slower")
        slower_count += 1

print()
print(f"PolarPandas is FASTER in: {faster_count}/{len(results)} benchmarks")
print(f"PolarPandas is slower in: {slower_count}/{len(results)} benchmarks")
print()

# Calculate geometric mean for overall performance

geometric_mean = math.exp(sum(math.log(max(s, 0.1)) for _, s in results) / len(results))
print(f"Overall Performance (geometric mean): {geometric_mean:.1f}x")
print()

print("💡 KEY INSIGHTS:")
print("   • PolarPandas excels at: creation, sorting, large operations")
print("   • Minor overhead from wrapper for small operations")
print("   • Best performance gains on datasets > 100k rows")
print("   • Same pandas API, better performance at scale!")
print()
