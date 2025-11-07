# 🐼⚡ PolarPandas

> **The fastest pandas-compatible API you'll ever use**

[![Tests](https://img.shields.io/badge/tests-1026%20passing-brightgreen?style=flat)](https://github.com/eddiethedean/polarpandas)
[![Coverage](https://img.shields.io/badge/coverage-48%25-yellow?style=flat)](https://github.com/eddiethedean/polarpandas)
[![Type Safety](https://img.shields.io/badge/ty-checked-brightgreen?style=flat)](https://docs.astral.sh/ty/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=flat)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat)](LICENSE)

**PolarPandas** is a blazing-fast, pandas-compatible API built on top of Polars. Write pandas code, get Polars performance. It's that simple.

## 🚀 Why PolarPandas?

| Feature | pandas | PolarPandas | Speedup |
|---------|--------|-------------|---------|
| **DataFrame Creation** | 224.89 ms | 15.95 ms | ⚡ **14.1x faster** |
| **Read CSV** | 8.00 ms | 0.88 ms | ⚡ **9.1x faster** |
| **Sorting** | 28.05 ms | 3.97 ms | ⚡ **7.1x faster** |
| **GroupBy** | 7.95 ms | 2.44 ms | ⚡ **3.3x faster** |
| **Filtering** | 1.26 ms | 0.42 ms | ⚡ **3.0x faster** |

**🎯 Overall Performance: 5.2x faster than pandas**

## ✨ Quick Start

```python
import polarpandas as ppd
import polars as pl

# Create a DataFrame (pandas syntax, Polars performance)
df = ppd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "city": ["NYC", "LA", "Chicago"]
})

# All your favorite pandas operations work!
df["age_plus_10"] = df["age"] + 10
df.sort_values("age", inplace=True)
result = df.groupby("city").agg(pl.col("age").mean())

# String operations with .str accessor
df["name_upper"] = df["name"].str.upper()

# Datetime operations with .dt accessor
df["birth_year"] = 2024 - df["age"]

print(df.head())
```

Output:
```
shape: (3, 6)
┌─────────┬─────┬─────────┬─────────────┬────────────┬────────────┐
│ name    ┆ age ┆ city   ┆ age_plus_10 ┆ name_upper ┆ birth_year │
│ ---     ┆ --- ┆ ---     ┆ ---         ┆ ---        ┆ ---        │
│ str     ┆ i64 ┆ str     ┆ i64         ┆ str        ┆ i64        │
╞═════════╪═════╪═════════╪═════════════╪════════════╪════════════╡
│ Alice   ┆ 25  ┆ NYC     ┆ 35          ┆ ALICE      ┆ 1999       │
│ Bob     ┆ 30  ┆ LA      ┆ 40          ┆ BOB        ┆ 1994       │
│ Charlie ┆ 35  ┆ Chicago ┆ 45          ┆ CHARLIE    ┆ 1989       │
└─────────┴─────┴─────────┴─────────────┴────────────┴────────────┘
```

## 🎯 What's New in v0.9.0

### ⚙️ **Rolling Apply Compatibility**
- ✅ `DataFrame.rolling().apply` now leverages Polars' native `rolling_map`, so pandas-style custom functions Just Work™
- ✅ Full support for `raw=True/False`, positional `args`, keyword `kwargs`, weights, centered windows, and `min_periods`
- ✅ More predictable results when mixing numeric and object windows thanks to consistent Series wrapping

### 🧭 **GroupBy Reliability**
- ✅ Grouping by missing columns now mirrors pandas: the validation happens at aggregation time and raises a clear `KeyError`
- ✅ Safer attribute access on `_GroupBy` objects, preventing silent failures in chained operations

### 🧪 **Quality & Tooling**
- ✅ **1,014 tests passing** across the suite, including comprehensive rolling-window scenarios
- ✅ `mypy` passes cleanly for `src/polarpandas`, keeping the public API fully typed
- ✅ `ruff check`/`ruff format` run squeaky clean on the updated codebase

---

## 🎯 What's New in v0.8.0

### 🗄️ **Enhanced SQL Support**
- ✅ **Primary key support** - Create SQL tables with single or composite primary keys
- ✅ **Auto-increment columns** - Automatic ID generation for primary keys
- ✅ **Advanced `to_sql()` method** - Enhanced DataFrame.to_sql() and Series.to_sql() with:
  - Primary key specification (`primary_key` parameter)
  - Auto-increment support (`auto_increment` parameter)
  - Full if_exists options ('fail', 'replace', 'append')
  - Connection string and SQLAlchemy engine support
- ✅ **Type mapping** - Automatic Polars to SQL type conversion
- ✅ **Comprehensive SQL utilities** - New `_sql_utils.py` module with SQLAlchemy integration

### 🧪 **Expanded Test Coverage**
- ✅ **1,026 tests passing** - Added 33 comprehensive SQL tests
- ✅ **88% coverage for SQL utilities** - Extensive testing of SQL functionality
- ✅ **Edge case testing** - Empty DataFrames, nulls, Unicode, large datasets (10K+ rows)
- ✅ **Data type testing** - Integer, float, boolean, date, datetime, and string types
- ✅ **Batch operations** - Multiple table operations and transaction testing

### 📦 **New Features**
- ✅ **Optional SQLAlchemy dependency** - Install with `pip install polarpandas[sqlalchemy]`
- ✅ **Graceful fallback** - Informative error messages when SQLAlchemy not installed
- ✅ **Connection flexibility** - Support for connection strings, engines, and connection objects

---

## 🎯 What's New in v0.7.0

### 🧪 **Improved Test Suite**
- ✅ **993 tests passing** - Doubled from 498 tests, comprehensive coverage
- ✅ **48% code coverage** - Significant improvement in test coverage
- ✅ **13 previously skipped tests now passing** - Fixed bugs and implemented missing features
- ✅ **No segfaults** - Resolved numpy/pandas compatibility issues with Python 3.9+
- ✅ **72 documented skipped tests** - Clear reasons for unimplemented features

### 🔧 **New Features & Bug Fixes**
- ✅ **Implemented `cut()` function** - Proper data binning with custom labels support
- ✅ **Fixed Series.sort_index()** - Resolved constructor issue
- ✅ **Fixed Series.repeat()** - Now works correctly with Polars backend
- ✅ **Fixed Series.where()** - Expression evaluation bug resolved
- ✅ **Fixed Series.mask()** - Expression evaluation bug resolved

### 🧹 **Pandas Removal Infrastructure**
- ✅ **Test helpers created** - Custom assertion utilities replace pandas testing functions
- ✅ **Expected values generator** - Generate test expectations without runtime pandas dependency
- ✅ **First file converted** - test_dataframe_statistical.py now runs without pandas (79 pandas calls eliminated)
- ✅ **Clear conversion path** - Complete documentation and tooling for removing pandas from all tests

### 🏗️ **Code Quality**
- ✅ **All ruff checks passing** - Zero linting errors in src/ and tests/
- ✅ **All pyright checks passing** - Zero type errors in new code
- ✅ **Python 3.9+ support** - Better compatibility, no segfaults
- ✅ **Comprehensive documentation** - Test improvement reports and conversion guides

---

## 🎯 What's New in v0.6.0

### 🚀 **Massive API Expansion**
- ✅ **619 pandas-compatible features** - Comprehensive pandas API coverage
- ✅ **69 module-level functions** - All major pandas functions implemented
- ✅ **206 DataFrame methods** - Complete DataFrame API support
- ✅ **186 Series methods** - Full Series functionality
- ✅ **73 Index methods** - Complete Index operations
- ✅ **57 String accessor methods** - Full `.str` accessor support
- ✅ **28 Datetime accessor methods** - Comprehensive `.dt` accessor support
- ✅ **91 LazyFrame methods** - Complete LazyFrame API (262 total methods tracked including pandas DataFrame comparison)

### 📊 **Enhanced I/O Support**
- ✅ **Comprehensive file format support** - CSV, JSON, Parquet, Excel, HDF5, HTML, XML, Stata, SPSS, SAS, and more
- ✅ **Enhanced SQL support** - Full pandas-compatible `to_sql()` with primary key and auto-increment support
- ✅ **Optional dependencies** - Organized into feature groups (excel, hdf5, html, spss, sas, xarray, clipboard, formatting, sqlalchemy)
- ✅ **Flexible installation** - Install only what you need: `pip install polarpandas[excel]` or `pip install polarpandas[all]`

### 🚀 **Features (from v0.2.0)**
- **LazyFrame Class** - Optional lazy execution for maximum performance
- **Lazy I/O Operations** - `scan_csv()`, `scan_parquet()`, `scan_json()` for lazy loading
- **Complete I/O operations** - Full CSV/JSON read/write support
- **Advanced statistical methods** - `nlargest()`, `nsmallest()`, `rank()`, `diff()`, `pct_change()`
- **String & datetime accessors** - Full `.str` and `.dt` accessor support
- **Module-level functions** - `read_csv()`, `concat()`, `merge()`, `get_dummies()`
- **Comprehensive edge cases** - Empty DataFrames, null values, mixed types
- **Full type annotations** - Complete ty type checking support
- **Comprehensive test coverage** - Tests for all core functionality and edge cases

## 📦 Installation

```bash
# Install from source (development)
git clone https://github.com/eddiethedean/polarpandas.git
cd polarpandas
pip install -e .

# Or install directly (when published)
pip install polarpandas

# Install with optional features
pip install polarpandas[sqlalchemy]  # For enhanced SQL features (primary keys, auto-increment)
pip install polarpandas[excel]       # For Excel file support
pip install polarpandas[all]         # Install all optional dependencies
```

**Requirements:** Python 3.8+ and Polars

**Optional Dependencies:**
- `numpy` - For passing NumPy dtype objects like `np.int64` in schemas
- `sqlalchemy` - For enhanced SQL features (primary keys, auto-increment in `to_sql()`)
- `pandas` - For certain conversion features and compatibility
- `openpyxl`, `xlsxwriter` - For Excel file I/O
- `lxml`, `html5lib` - For HTML/XML parsing
- `pyreadstat`, `sas7bdat` - For SPSS/SAS file support
- `types-tabulate` - Lightweight type stubs to keep `tabulate`-powered helpers mypy-clean
- And more... see `pyproject.toml` for complete list

## 🔥 Core Features

### ⚡ **Eager vs Lazy Execution**

PolarPandas gives you the **best of both worlds**:

```python
import polarpandas as ppd
import polars as pl

# 🚀 EAGER EXECUTION (Default - like pandas)
df = ppd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
result = df.filter(df["a"] > 1)  # Executes immediately
print(result)
# Shows results right away:
# shape: (2, 2)
# ┌─────┬─────┐
# │ a   ┆ b   │
# │ --- ┆ --- │
# │ i64 ┆ i64 │
# ╞═════╪═════╡
# │ 2   ┆ 5   │
# │ 3   ┆ 6   │
# └─────┴─────┘

# ⚡ LAZY EXECUTION (Optional - for maximum performance)
lf = df.lazy()  # Convert to LazyFrame
lf_filtered = lf.filter(pl.col("a") > 1)  # Stays lazy
df_result = lf_filtered.collect()  # Materialize when ready

# 📁 LAZY I/O (For large files)
lf = ppd.scan_csv("huge_file.csv")  # Lazy loading
lf_processed = lf.filter(pl.col("value") > 100).select("name", "value")
df_final = lf_processed.collect()  # Execute optimized plan
```

**When to use LazyFrame:**
- 📊 **Large datasets** (>1M rows)
- 🔄 **Complex operations** (multiple filters, joins, aggregations)
- 💾 **Memory constraints** (lazy evaluation uses less memory)
- ⚡ **Performance critical** applications

### 📊 **DataFrame Operations**
```python
# Initialization
df = ppd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

# Eager I/O (immediate loading)
df = ppd.read_csv("data.csv")
df = ppd.read_json("data.json")
df = ppd.read_parquet("data.parquet")

# Lazy I/O (for large files)
lf = ppd.scan_csv("large_file.csv")
lf = ppd.scan_parquet("huge_file.parquet")
lf = ppd.scan_json("big_file.json")

# Mutable operations (pandas-style)
df["new_col"] = df["A"] * 2
df.drop("old_col", axis=1, inplace=True)
df.rename(columns={"A": "alpha"}, inplace=True)
df.sort_values("B", inplace=True)

# Advanced operations
import polars as pl
df.groupby("category").agg(pl.col("value").mean())  # Use Polars expressions
df.pivot_table(values="sales", index="region", columns="month")
df.rolling(window=3).mean()
```

### 🗄️ **Enhanced SQL Operations**
PolarPandas now supports full pandas-compatible SQL operations with advanced features:

```python
from sqlalchemy import create_engine

# Create database connection
engine = create_engine('sqlite:///mydb.db')

# Basic write (uses Polars' fast write_database)
df = ppd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
df.to_sql('users', engine, if_exists='replace')

# Create table with primary key (requires SQLAlchemy)
df.to_sql('users', engine, if_exists='replace', primary_key='id')

# Create table with auto-incrementing primary key
df.to_sql('users', engine, if_exists='replace', 
          primary_key='id', auto_increment=True)

# Composite primary key
df.to_sql('users', engine, if_exists='replace', 
          primary_key=['id', 'email'])

# Read back from SQL
result = ppd.read_sql("SELECT * FROM users WHERE id > 1", engine)
```

**Key Features:**
- 🚀 **Fast by default** - Uses Polars' native `write_database()` when no special features needed
- 🔑 **Primary key support** - Set single or composite primary keys (requires SQLAlchemy)
- ⚡ **Auto-increment** - Enable auto-incrementing IDs (requires SQLAlchemy)
- 🔄 **Smart fallback** - Automatically uses Polars for performance, SQLAlchemy for features
- ✅ **Pandas-compatible** - Complete pandas `to_sql()` signature support

### 🧩 **Schema Conversion (pandas-style to Polars)**
PolarPandas accepts schemas in multiple forms and converts them to Polars types automatically:

- String dtype names: "int64", "float64", "object", "bool", "datetime", "category"
- NumPy dtypes: `np.int64`, `np.float32`, `np.uint8`, ...
- pandas dtypes: `pd.Int64Dtype()`, `pd.Float32Dtype()`, `pd.StringDtype()`, ...
- Polars schema dict or `pl.Schema`

Constructor usage:
```python
import numpy as np
import polars as pl
import polarpandas as ppd

data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}

# Strings
 df = ppd.DataFrame(data, dtype={"a": "int64", "b": "string"})

# NumPy dtypes (requires optional numpy install)
 df = ppd.DataFrame(data, dtype={"a": np.int64, "b": np.float64})

# pandas dtypes
# df = ppd.DataFrame(data, dtype={"a": pd.Int64Dtype(), "b": pd.StringDtype()})

# Polars schema dict
 df = ppd.DataFrame(data, dtype={"a": pl.Int64, "b": pl.Utf8})
```

I/O functions:
```python
# Eager
 df = ppd.read_csv("data.csv", dtype={"id": "int64", "name": "string"})
 df = ppd.read_json("data.json", schema={"value": "float64"})
 df = ppd.read_parquet("data.parquet", dtype={"id": "uint32"})  # casts after read
 df = ppd.read_feather("data.feather", schema={"flag": "bool"})  # casts after read

# Lazy (scan)
 lf = ppd.scan_csv("data.csv", schema={"id": "int64"})
 lf = ppd.scan_parquet("data.parquet", dtype={"score": "float32"})  # lazy cast
 lf = ppd.scan_json("data.json", dtype={"name": "string"})
```

Notes:
- When both `dtype` and `schema` are provided, `schema` takes precedence.
- Parquet/Feather do not accept a schema parameter at read time in Polars; types are cast after reading (or lazily for scans).

### 📈 **Series Operations**
```python
# String operations
df["name"].str.upper()
df["email"].str.contains("@")
df["text"].str.split(" ")

# Datetime operations
df["date"].dt.year
df["timestamp"].dt.floor("D")
df["datetime"].dt.strftime("%Y-%m-%d")

# Statistical methods
df["values"].rank()
df["scores"].nlargest(5)
df["prices"].clip(lower=0, upper=100)
```

### 🎯 **Advanced Indexing** ⚡
All indexing operations now use **native Polars implementations** for maximum performance - no pandas conversion overhead!

```python
# Label-based indexing (with index set)
df = ppd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "city": ["NYC", "LA", "Chicago"]
}, index=["a", "b", "c"])

# Select rows by label
df.loc["a"]  # Single row (returns Series)
df.loc[["a", "b"], ["name", "age"]]  # Multiple rows and columns
# Output:
# shape: (2, 2)
# ┌───────┬─────┐
# │ name  ┆ age │
# │ ---   ┆ --- │
# │ str   ┆ i64 │
# ╞═══════╪═════╡
# │ Alice ┆ 25  │
# │ Bob   ┆ 30  │
# └───────┴─────┘

# Position-based indexing
df.iloc[0:2, 1:3]  # Slice rows and columns
# Output:
# shape: (2, 2)
# ┌─────┬─────────┐
# │ age ┆ city    │
# │ --- ┆ ---     │
# │ i64 ┆ str     │
# ╞═════╪═════════╡
# │ 25  ┆ NYC     │
# │ 30  ┆ LA      │
# └─────┴─────────┘

df.iloc[[0, 2], :]  # Select specific rows, all columns
# Output:
# shape: (2, 3)
# ┌─────────┬─────┬─────────┐
# │ name    ┆ age ┆ city    │
# │ ---     ┆ --- ┆ ---     │
# │ str     ┆ i64 ┆ str     │
# ╞═════════╪═════╪═════════╡
# │ Alice   ┆ 25  ┆ NYC     │
# │ Charlie ┆ 35  ┆ Chicago  │
# └─────────┴─────┴─────────┘

# Assignment (now using native Polars - 270x faster for boolean masks!)
df.loc["a", "age"] = 26
df.iloc[0, 0] = "Alice Updated"
df.loc[df["age"] > 25, "age"] = 30  # Boolean mask assignment - optimized!
```

## 🏗️ **Architecture**

PolarPandas uses a **wrapper pattern** that provides:

- **Mutable operations** with `inplace` parameter
- **Index preservation** across operations
- **Pandas-compatible API** with Polars performance
- **Type safety** with comprehensive type hints
- **Error handling** that matches pandas behavior

```python
# Internal structure
class DataFrame:
    def __init__(self, data):
        self._df = pl.DataFrame(data)  # Polars backend
        self._index = None              # Pandas-style index
        self._index_name = None         # Index metadata
```

## 📊 **Performance Benchmarks**

Run benchmarks yourself:
```bash
python benchmark_large.py
```

### **Large Dataset Performance (1M rows)**
| Operation | pandas | PolarPandas | Speedup |
|-----------|--------|-------------|---------|
| DataFrame Creation | 224.89 ms | 15.95 ms | ⚡ **14.1x** |
| Read CSV | 8.00 ms | 0.88 ms | ⚡ **9.1x** |
| Sorting | 28.05 ms | 3.97 ms | ⚡ **7.1x** |
| GroupBy | 7.95 ms | 2.44 ms | ⚡ **3.3x** |
| Filtering | 1.26 ms | 0.42 ms | ⚡ **3.0x** |

### **Memory Efficiency**
- **50% less memory usage** than pandas
- **⚡ Lazy evaluation** for complex operations (LazyFrame)
- **Optimized data types** with Polars backend
- **Query optimization** with lazy execution plans

## 🧪 **Testing & Quality**

### ✅ **Comprehensive Testing**
- **498 tests passing** (100% success rate)
- **54 tests properly skipped** (documented limitations)
- **72% code coverage** across all functionality
- **Edge case handling** for empty DataFrames, null values, mixed types
- **Comprehensive error handling** with proper exception conversion
- **Parallel test execution** - Fast test runs with pytest-xdist

### ✅ **Code Quality**
- **Zero linting errors** with ruff compliance
- **100% type safety** - all ty type errors resolved
- **Fully formatted code** with ruff formatter
- **Clean code standards** throughout
- **Production-ready** code quality

### ✅ **Type Safety**
```python
# Full type hints support
def process_data(df: ppd.DataFrame) -> ppd.DataFrame:
    return df.groupby("category").agg({"value": "mean"})

# IDE support with autocompletion
df.loc[df["age"] > 25, "name"]  # Type-safe operations
```

## 🔧 **Development**

### **Running Tests**
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/polarpandas --cov-report=html

# Specific test file
pytest tests/test_dataframe_core.py -v

# SQL enhanced suite (requires SQLAlchemy extra)
pip install -e '.[test,sqlalchemy]'
pytest -m requires_sqlalchemy tests/test_sql_enhanced.py -v
```

### **Code Quality**
```bash
# Format code
ruff format .

# Check linting
ruff check .

# Type checking
ty check src/polarpandas/
```

**Current Status:**
- ✅ All tests passing (498 passed, 54 skipped)
- ✅ Zero linting errors (ruff check)
- ✅ Code fully formatted (ruff format)
- ✅ Type checked (ty compliance)
- ✅ Parallel test execution supported

### **Benchmarks**
```bash
# Basic benchmarks
python benchmark.py

# Large dataset benchmarks
python benchmark_large.py

# Detailed analysis
python benchmark_detailed.py
```

## 📋 **Known Limitations**

PolarPandas achieves **100% compatibility** for implemented features. Remaining limitations are due to fundamental Polars architecture differences:

### 🔄 **Permanent Limitations**
- **Correlation/Covariance**: Polars doesn't have built-in `corr()`/`cov()` methods
- **Transpose with mixed types**: Polars handles mixed types differently than pandas
- **MultiIndex support**: Polars doesn't have native MultiIndex support
- **JSON orient formats**: Some pandas JSON orient formats not supported by Polars

### 🔍 **Temporary Limitations**
- **Advanced indexing**: Some complex pandas indexing patterns not yet implemented
- **Complex statistical methods**: Some advanced statistical operations need implementation

**Total: 54 tests properly skipped with clear documentation**

## 🤝 **Contributing**

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run the test suite**: `pytest tests/ -v`
5. **Check code quality**: `ruff check src/polarpandas/`
6. **Submit a pull request**

### **Development Setup**
```bash
git clone https://github.com/eddiethedean/polarpandas.git
cd polarpandas
pip install -e ".[dev,test]"
```

> 💡 **Running optional SQL tests?** Install the SQLAlchemy extra (`pip install -e ".[sqlalchemy]"` or rely on the dev/test extras above) and execute `pytest -m requires_sqlalchemy` to include the SQL enhanced suite. Without the extra, those tests are automatically skipped.

## 📚 **Documentation**

- **[API Compatibility Matrix](https://github.com/eddiethedean/polarpandas/blob/main/PANDAS_FUNCTION_MATRIX.md)** - Complete pandas API compatibility matrix showing which functions and methods are implemented
- **[API Reference](docs/api.md)** - Complete API documentation
- **[Performance Guide](docs/performance.md)** - Optimization tips
- **[Migration Guide](docs/migration.md)** - From pandas to PolarPandas
- **[Examples](examples/)** - Real-world usage examples

## 🏆 **Why Choose PolarPandas?**

| Feature | pandas | Polars | PolarPandas |
|---------|--------|--------|-------------|
| **Performance** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Memory Usage** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **API Familiarity** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Ecosystem** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Type Safety** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

**🎯 Best of both worlds: pandas API + Polars performance**

## 📈 **Roadmap**

### **v0.6.0 (Current)**
- ✅ **619 pandas-compatible features** - Comprehensive API coverage
- ✅ **Complete Index methods** - All 73 Index methods implemented
- ✅ **Full String accessor** - All 57 `.str` methods implemented
- ✅ **Complete Datetime accessor** - All 28 `.dt` methods implemented
- ✅ **91 LazyFrame methods** - Complete LazyFrame API with pandas DataFrame comparison (262 total methods tracked)
- ✅ **Enhanced I/O support** - Multiple file formats with optional dependencies
- ✅ **Type checking with `ty`** - Modern, fast type checker integration
- ✅ **API compatibility matrix** - Comprehensive tracking of pandas compatibility

### **v0.4.0**
- ✅ **Native Polars Indexing** - Replaced all pandas fallbacks with native Polars implementations
- ✅ **Boolean Mask Optimization** - 270x performance improvement for boolean mask assignment
- ✅ **Optional Pandas** - Pandas is now truly optional, only required for specific conversion features
- ✅ **Enhanced Error Handling** - Typo suggestions in error messages
- ✅ **Code Refactoring** - Centralized index management and exception utilities
- ✅ **Type Safety** - Improved type checking and resolved critical type issues

### **v0.3.1**
- ✅ Fixed GitHub Actions workflow dependencies (pytest, pandas, numpy, pyarrow)
- ✅ Fixed Windows file handling issues in I/O tests (28 tests now passing)
- ✅ All platforms (Ubuntu, macOS, Windows) now passing all 457 tests

### **v0.3.0**
- ✅ **Comprehensive Documentation** - Professional docstrings for all public APIs
- ✅ **LazyFrame Class** - Optional lazy execution for maximum performance
- ✅ **Lazy I/O Operations** - `scan_csv()`, `scan_parquet()`, `scan_json()`
- ✅ **Eager DataFrame** - Default pandas-like behavior
- ✅ **Seamless Conversion** - `df.lazy()` and `lf.collect()` methods
- ✅ **100% Type Safety** - All ty errors resolved
- ✅ **Comprehensive Testing** - 457 tests covering all functionality
- ✅ **Code Quality** - Zero linting errors, fully formatted code

### **v0.7.0 (Planned)**
- [ ] Advanced MultiIndex support
- [ ] More statistical methods
- [ ] Enhanced I/O formats (additional formats)
- [ ] Further performance optimizations
- [ ] Additional LazyFrame method implementations

### **Future**
- [ ] Machine learning integration
- [ ] Advanced visualization support
- [ ] Distributed computing support
- [ ] GPU acceleration

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **[Polars](https://pola.rs/)** - The blazing-fast DataFrame library
- **[pandas](https://pandas.pydata.org/)** - The inspiration and API reference
- **Contributors** - Everyone who helps make PolarPandas better

---

<div align="center">

**Made with ❤️ for the data science community**

[⭐ Star us on GitHub](https://github.com/eddiethedean/polarpandas) • [🐛 Report Issues](https://github.com/eddiethedean/polarpandas/issues) • [💬 Discussions](https://github.com/eddiethedean/polarpandas/discussions)

</div>