# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2025-11-07

### Added
- **Rolling apply compatibility** — `DataFrame.rolling().apply` now wraps Polars' `rolling_map`, enabling pandas-style custom window functions
- **Documentation refresh** — README/TODO now highlight the 0.9.0 release and current roadmap

### Changed
- **Window semantics** — Respects `raw`, `args`, `kwargs`, `weights`, `center`, and `min_periods` for rolling operations
- **GroupBy validation** — Missing columns are reported at aggregation time with pandas-compatible `KeyError` messaging
- **Developer tooling** — Project version bumped to 0.9.0 with README and TODO updates covering the new behaviour

### Fixed
- **GroupBy attribute access** — Prevents accidental usage of incomplete `_GroupBy` objects when column validation fails
- **Rolling window edge cases** — Ensures mixed-type windows and Series-wrapped inputs behave consistently
- **CI hygiene** — `mypy` (package scope) and `ruff` remain green; full pytest suite (1,014 tests) passes

## [0.8.0] - 2025-11-07

### Added
- **Enhanced SQL support** - Advanced `to_sql()` method with primary keys and auto-increment
- **Primary key support** - Single and composite primary key specification in SQL table creation
- **Auto-increment columns** - Automatic ID generation for primary key columns
- **SQL utilities module** - New `_sql_utils.py` with SQLAlchemy integration for advanced features
- **Type mapping** - Automatic Polars to SQL type conversion (Int8→SmallInteger, Float64→Float, etc.)
- **Connection flexibility** - Support for connection strings, SQLAlchemy engines, and connection objects
- **Comprehensive SQL tests** - 33 new tests covering edge cases, data types, and batch operations

### Changed
- **Test suite expanded** - 1,026 tests passing (up from 993)
- **SQL coverage** - 88% code coverage for SQL utilities
- **Optional SQLAlchemy dependency** - Enhanced SQL features require `pip install polarpandas[sqlalchemy]`

### Fixed
- **SQLAlchemy integration** - Proper error handling when SQLAlchemy not installed
- **SQL data type handling** - Correct mapping for all numeric, string, date, and boolean types
- **Composite key validation** - Proper error messages for invalid composite key + auto-increment combinations

## [0.7.0] - 2025-11-07

### Added
- **`cut()` function** - Proper data binning with support for integer bins and explicit bin edges
- **Custom labels support** - Ability to specify custom labels for bins in `cut()` function
- **Test helpers module** - Custom assertion utilities (`assert_frame_equal`, `assert_series_equal`, etc.)
- **Expected values generator** - Script to generate test expectations using pandas (one-time generation)
- **Pandas removal infrastructure** - Complete tooling for eliminating pandas dependency from tests

### Fixed
- **Series.sort_index()** - Fixed Polars Series constructor issue causing TypeError
- **Series.repeat()** - Implemented proper element repetition (Polars API compatibility)
- **Series.where()** - Fixed Expr vs Series conversion bug
- **Series.mask()** - Fixed Expr vs Series conversion bug
- **13 skipped tests** - Now passing after implementing features and fixing bugs

### Changed
- **Test suite doubled** - 993 tests passing (up from 498)
- **Coverage improved** - 48% code coverage (up from 12%)
- **Skipped tests reduced** - From 114 to 72 with clear documentation
- **Python 3.9+ recommended** - Better compatibility, no numpy/pandas segfaults
- **test_dataframe_statistical.py** - Converted to use test helpers (eliminated 79 pandas calls)

### Removed
- **3 obsolete tests** - Removed tests for unimplemented rolling window operations
- **60 skipped tests** - Fixed or cleaned up previously skipped tests

## [0.2.0] - 2024-01-XX

### Added
- **Complete I/O operations**: Full CSV/JSON read/write support with pandas-compatible parameters
- **Advanced statistical methods**: `nlargest()`, `nsmallest()`, `rank()`, `diff()`, `pct_change()`
- **String and datetime accessors**: Full `.str` and `.dt` accessor support with comprehensive methods
- **Comprehensive edge case handling**: Empty DataFrames, null values, mixed types, infinite values
- **Module-level functions**: pandas-style functions like `read_csv()`, `concat()`, `merge()`, `get_dummies()`
- **Enhanced error handling**: Better exception handling and edge case coverage
- **Type annotations**: Complete type hints throughout the codebase

### Changed
- **Significantly improved type safety**: 30% reduction in mypy errors (101 → 71)
- **Enhanced code quality**: Full ruff compliance and clean code standards
- **Better test coverage**: 76% code coverage with comprehensive test scenarios
- **Proper limitation documentation**: Clear documentation of permanent Polars limitations
- **Performance optimizations**: Leveraging Polars' native performance for better speed

### Fixed
- **100% test success rate**: All implemented features work perfectly (324/324 tests passing)
- **Zero linting errors**: Clean, consistent code style throughout
- **Enhanced documentation**: Clear docstrings and type hints for all methods
- **Robust error handling**: Better exception handling for edge cases
- **Index preservation**: Proper index handling in various operations

### Removed
- **Unused code**: Cleaned up unused imports and dead code
- **Redundant implementations**: Streamlined code for better maintainability

### Security
- **No security issues**: Clean codebase with no known vulnerabilities
- **Dependency management**: Single dependency (Polars) for minimal attack surface

## [0.1.0] - 2024-01-XX

### Added
- Initial release of PolarPandas
- Basic DataFrame and Series implementations
- Core pandas-compatible operations
- Initial test suite
- Basic documentation

---

## Known Limitations

### Permanent Limitations (Polars Architecture)
- **Correlation/Covariance**: Polars doesn't have built-in `corr()`/`cov()` methods
- **Transpose with mixed types**: Polars handles mixed types differently than pandas
- **MultiIndex support**: Polars doesn't have native MultiIndex support
- **NaN with datetime**: Polars type system limitations with mixed datetime/NaN values
- **JSON orient formats**: Some pandas JSON orient formats not supported by Polars

### Temporary Limitations (Can be improved)
- **Advanced indexing**: Some complex pandas indexing patterns not yet implemented
- **Complex statistical methods**: Some advanced statistical operations need implementation

**Total: 54 tests properly skipped with clear documentation**
