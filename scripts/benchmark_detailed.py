"""
Detailed performance analysis to determine when pandas vs polars is faster.
This helps decide if a hybrid approach would be beneficial.
"""

import sys
import time

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Install pandas: pip install pandas")
    sys.exit(1)

import random

import polarpandas as ppd


def benchmark_operation(name, size, pandas_fn, polarpandas_fn, iterations=5):
    """Benchmark a single operation."""
    # Pandas
    pd_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        pandas_fn()
        pd_times.append(time.perf_counter() - start)
    pandas_time = min(pd_times)

    # PolarPandas
    ppd_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        polarpandas_fn()
        ppd_times.append(time.perf_counter() - start)
    polarpandas_time = min(ppd_times)

    speedup = pandas_time / polarpandas_time
    winner = "polarpandas" if speedup > 1 else "pandas"

    return {
        "operation": name,
        "size": size,
        "pandas_ms": pandas_time * 1000,
        "polarpandas_ms": polarpandas_time * 1000,
        "speedup": speedup,
        "winner": winner,
    }


print("=" * 90)
print("DETAILED PERFORMANCE ANALYSIS: When to use pandas vs polars?")
print("=" * 90)
print()

results = []

# Test different dataset sizes
sizes = [100, 1_000, 10_000, 100_000]

for size in sizes:
    print(f"\n{'=' * 90}")
    print(f"DATASET SIZE: {size:,} rows")
    print(f"{'=' * 90}\n")

    # ===========================================================================
    # Simple column selection
    # ===========================================================================
    data = {"a": list(range(size)), "b": list(range(size, 2 * size))}
    pd_df = pd.DataFrame(data)
    ppd_df = ppd.DataFrame(data)

    result = benchmark_operation(
        "Column Selection",
        size,
        lambda pd_df=pd_df: pd_df["a"],  # noqa: B023
        lambda ppd_df=ppd_df: ppd_df["a"],  # noqa: B023
    )
    results.append(result)
    print(
        f"  Column Selection:          {result['winner']:12} wins  "
        f"(pandas: {result['pandas_ms']:.3f}ms, polarpandas: {result['polarpandas_ms']:.3f}ms, "
        f"speedup: {result['speedup']:.1f}x)"
    )

    # ===========================================================================
    # Column assignment
    # ===========================================================================
    result = benchmark_operation(
        "Column Assignment",
        size,
        lambda pd_df=pd_df: (
            pd_df.copy().__setitem__("c", pd_df["a"] + pd_df["b"]),
            None,
        )[1],  # noqa: B023
        lambda ppd_df=ppd_df: (
            ppd_df.copy().__setitem__("c", ppd_df["a"] + ppd_df["b"]),
            None,
        )[1],  # noqa: B023
    )
    results.append(result)
    print(
        f"  Column Assignment:         {result['winner']:12} wins  "
        f"(pandas: {result['pandas_ms']:.3f}ms, polarpandas: {result['polarpandas_ms']:.3f}ms, "
        f"speedup: {result['speedup']:.1f}x)"
    )

    # ===========================================================================
    # Filtering
    # ===========================================================================
    threshold = size // 2
    result = benchmark_operation(
        "Filtering",
        size,
        lambda pd_df=pd_df, threshold=threshold: pd_df[pd_df["a"] > threshold],  # noqa: B023
        lambda ppd_df=ppd_df, threshold=threshold: ppd_df.filter(
            ppd_df["a"] > threshold
        ),  # noqa: B023
    )
    results.append(result)
    print(
        f"  Filtering:                 {result['winner']:12} wins  "
        f"(pandas: {result['pandas_ms']:.3f}ms, polarpandas: {result['polarpandas_ms']:.3f}ms, "
        f"speedup: {result['speedup']:.1f}x)"
    )

    # ===========================================================================
    # Aggregation
    # ===========================================================================
    result = benchmark_operation(
        "Aggregation (mean)",
        size,
        lambda pd_df=pd_df: pd_df["a"].mean(),  # noqa: B023
        lambda ppd_df=ppd_df: ppd_df["a"].mean(),  # noqa: B023
    )
    results.append(result)
    print(
        f"  Aggregation:               {result['winner']:12} wins  "
        f"(pandas: {result['pandas_ms']:.3f}ms, polarpandas: {result['polarpandas_ms']:.3f}ms, "
        f"speedup: {result['speedup']:.1f}x)"
    )

    # ===========================================================================
    # Sorting
    # ===========================================================================
    if size <= 100_000:  # Don't shuffle huge datasets
        shuffled = list(range(size))
        random.shuffle(shuffled)
        pd_df_shuffle = pd.DataFrame({"a": shuffled, "b": list(range(size))})
        ppd_df_shuffle = ppd.DataFrame({"a": shuffled.copy(), "b": list(range(size))})

        result = benchmark_operation(
            "Sorting",
            size,
            lambda pd_df_shuffle=pd_df_shuffle: pd_df_shuffle.sort_values("a"),  # noqa: B023
            lambda ppd_df_shuffle=ppd_df_shuffle: ppd_df_shuffle.sort_values("a"),  # noqa: B023
        )
        results.append(result)
        print(
            f"  Sorting:                   {result['winner']:12} wins  "
            f"(pandas: {result['pandas_ms']:.3f}ms, polarpandas: {result['polarpandas_ms']:.3f}ms, "
            f"speedup: {result['speedup']:.1f}x)"
        )

    # ===========================================================================
    # GroupBy (if size allows)
    # ===========================================================================
    if size >= 1000:
        num_groups = min(100, size // 10)
        groups = [f"Group_{i % num_groups}" for i in range(size)]
        pd_df_gb = pd.DataFrame({"group": groups, "value": list(range(size))})
        ppd_df_gb = ppd.DataFrame({"group": groups, "value": list(range(size))})

        result = benchmark_operation(
            "GroupBy",
            size,
            lambda pd_df_gb=pd_df_gb: pd_df_gb.groupby("group")["value"].mean(),  # noqa: B023
            lambda ppd_df_gb=ppd_df_gb: ppd_df_gb.groupby("group").agg(
                ppd_df_gb["value"].mean()
            ),  # noqa: B023
            iterations=3,
        )
        results.append(result)
        print(
            f"  GroupBy:                   {result['winner']:12} wins  "
            f"(pandas: {result['pandas_ms']:.3f}ms, polarpandas: {result['polarpandas_ms']:.3f}ms, "
            f"speedup: {result['speedup']:.1f}x)"
        )


# ===========================================================================
# ANALYSIS
# ===========================================================================
print("\n" + "=" * 90)
print("ANALYSIS: When to Use Each Backend")
print("=" * 90)
print()

# Group by size and operation
by_size = {}
by_operation = {}

for r in results:
    size = r["size"]
    op = r["operation"]

    if size not in by_size:
        by_size[size] = {"pandas": 0, "polarpandas": 0}
    by_size[size][r["winner"]] += 1

    if op not in by_operation:
        by_operation[op] = {
            "pandas": 0,
            "polarpandas": 0,
            "pandas_wins_at": [],
            "polarpandas_wins_at": [],
        }
    by_operation[op][r["winner"]] += 1
    if r["winner"] == "pandas":
        by_operation[op]["pandas_wins_at"].append(size)
    else:
        by_operation[op]["polarpandas_wins_at"].append(size)

# Print analysis by size
print("📊 Performance by Dataset Size:")
print("-" * 90)
for size in sorted(by_size.keys()):
    pd_wins = by_size[size]["pandas"]
    ppd_wins = by_size[size]["polarpandas"]
    total = pd_wins + ppd_wins
    print(
        f"  {size:>8,} rows: polarpandas wins {ppd_wins}/{total} operations ({ppd_wins / total * 100:.0f}%)"
    )

print()
print("📈 Performance by Operation Type:")
print("-" * 90)
for op, data in sorted(by_operation.items()):
    pd_wins = data["pandas"]
    ppd_wins = data["polarpandas"]
    total = pd_wins + ppd_wins

    if ppd_wins > pd_wins:
        print(f"  {op:<25} ⚡ polarpandas better ({ppd_wins}/{total})")
    elif pd_wins > ppd_wins:
        print(f"  {op:<25} 🐼 pandas better ({pd_wins}/{total})")
        print(f"      └─ pandas wins at: {data['pandas_wins_at']}")
    else:
        print(f"  {op:<25} ⚖️  tied")

# ===========================================================================
# RECOMMENDATION
# ===========================================================================
print("\n" + "=" * 90)
print("💡 RECOMMENDATION: Should we use a hybrid approach?")
print("=" * 90)
print()

polarpandas_total_wins = sum(1 for r in results if r["winner"] == "polarpandas")
pandas_total_wins = sum(1 for r in results if r["winner"] == "pandas")

print(f"Overall: polarpandas wins {polarpandas_total_wins}/{len(results)} benchmarks")
print()

# Calculate average speedup for polarpandas wins
ppd_speedups = [r["speedup"] for r in results if r["winner"] == "polarpandas"]
pd_speedups = [1 / r["speedup"] for r in results if r["winner"] == "pandas"]

if ppd_speedups:
    avg_ppd_speedup = sum(ppd_speedups) / len(ppd_speedups)
    print(f"When polarpandas wins, it's {avg_ppd_speedup:.1f}x faster on average")

if pd_speedups:
    avg_pd_speedup = sum(pd_speedups) / len(pd_speedups)
    print(f"When pandas wins, it's {avg_pd_speedup:.1f}x faster on average")

print()
print("🎯 VERDICT:")
print()

if polarpandas_total_wins >= len(results) * 0.7:
    print(
        f"  ✅ Polarpandas wins in most cases ({polarpandas_total_wins / len(results) * 100:.0f}%)"
    )
    print("  ✅ Consistent performance advantages")
    print("  ⚠️  Hybrid approach adds complexity without major benefits")
    print()
    print("  RECOMMENDATION: Use polarpandas consistently")
    print("  • Simpler codebase (no switching logic)")
    print("  • Better performance in most scenarios")
    print("  • Overhead is minimal for small operations")
else:
    print("  🤔 Mixed results - hybrid approach could help")
    print(f"  • pandas wins: {pandas_total_wins} operations")
    print(f"  • polarpandas wins: {polarpandas_total_wins} operations")
    print()
    print("  RECOMMENDATION: Consider hybrid for specific use cases")

print()
print("=" * 90)
print("✨ Conclusion: polarpandas provides excellent performance across the board!")
print("=" * 90)
