# Polarpandas Known Limitations

This document outlines the known limitations and differences between polarpandas and pandas that are due to fundamental differences between Polars and pandas libraries.

## Current Test Status

- **1,014 passing tests**
- **30 skipped tests** (documented limitations)
- **48% overall code coverage** (key modules: lazyframe.py 97%, index.py 73%, frame.py 40%, series.py 46%)

## Permanent Limitations (Pure Polars Constraints)

Polarpandas is a pure Polars library and does not depend on pandas. Some pandas behaviors cannot be replicated exactly:

### 1. Correlation/Covariance
- **Status**: ✅ **IMPLEMENTED** — `corr()` and `cov()` methods now available
- **Implementation**: Uses pure Polars expressions (no numpy dependency)
- **Limitation**: Only Pearson correlation is supported (Kendall/Spearman raise NotImplementedError)
- **Note**: Matches pandas behavior for sample statistics (ddof=1)

### 2. Transpose with Mixed Dtypes
- **Issue**: Mixed dtypes after transpose behave differently
- **Reason**: Polars promotes to a common supertype; pandas preserves object-like behavior
- **Workaround**: Avoid mixed-type transpose or cast explicitly after
- **Status**: PERMANENT — behavior differs by design; some tests skipped
- **Tracked tests**: `tests/test_transpose.py` (multiple scenarios)

### 3. MultiIndex Support
- **Issue**: Limited/absent MultiIndex parity (only pragmatic helpers available)
- **Reason**: Polars has no native MultiIndex concept
- **Workaround**: Use concatenated columns or struct types as a surrogate
- **Status**: PERMANENT — architectural limitation
- **Note**: Basic MultiIndex operations are implemented, but full hierarchical index support is limited

### 4. JSON Orient Formats
- **Issue**: Some pandas JSON orient formats are unsupported
- **Reason**: Polars supports a subset of pandas JSON orients
- **Status**: PERMANENT — documented skips where applicable

### 5. Loc/Iloc with Mixed Types
- **Issue**: Mixed type handling in loc/iloc operations differs from pandas
- **Reason**: Polars converts mixed types to strings; pandas preserves as objects
- **Workaround**: Results are functionally equivalent when converted to strings
- **Status**: PERMANENT — behavior differs by design; some tests skipped
- **Note**: Full loc/iloc implementation is available with native Polars performance

### 6. Index with Null Values
- **Issue**: Limited support for null values in index
- **Reason**: Polars has restrictions on null values in index columns
- **Status**: PERMANENT — documented limitation (tracked in index test skips)

### 7. Datetime Accessor with Empty Series
- **Issue**: Datetime accessor behavior differs for empty series
- **Reason**: Polars and pandas handle empty datetime series differently
- **Status**: PERMANENT — documented limitation
- **Note**: All 28 datetime accessor methods are implemented and work correctly for non-empty series
- **Tracked tests**: `tests/test_string_datetime_enhanced.py::{test_date_property,test_datetime_methods_empty_series}`

### 8. Dynamic DataFrame Expansion
- **Issue**: Polars doesn't support dynamic DataFrame expansion (assigning to out-of-bounds indices)
- **Reason**: Polars DataFrames have fixed size
- **Status**: PERMANENT — documented limitation (tests: `TestErrorConditions.test_loc_assignment_adds_row`)

### 9. String Accessor Edge Cases
- **Issue**: Some string operations may behave slightly differently with null values
- **Reason**: Polars and pandas handle null values in string operations differently
- **Status**: PERMANENT — documented limitation
- **Note**: All 57 string accessor methods are implemented

### 10. Optional Dependencies
- **Issue**: Some I/O operations require optional dependencies
- **Reason**: To keep the core package lightweight, certain file format support requires additional packages
- **Status**: PERMANENT — by design
- **Workaround**: Install optional dependencies as needed:
  - `pip install polarpandas[excel]` for Excel support
  - `pip install polarpandas[hdf5]` for HDF5 support
  - `pip install polarpandas[html]` for HTML/XML support
  - `pip install polarpandas[all]` for all optional features

### 11. NaN Representation in String Columns
- **Issue**: Mixed string/numeric columns with `NaN` render differently from pandas
- **Reason**: Polars promotes string columns with `NaN` sentinels differently, causing equality checks to fail
- **Status**: PERMANENT — tests such as `TestNullValues.test_mixed_nan_values` remain skipped until Polars alignment
- **Workaround**: Cast affected columns to a consistent dtype (e.g., `Utf8`) before comparison

### 12. Higher-Order Statistical Moments
- **Issue**: Calculating `mode`, `skew`, or `kurt` via NumPy-backed fallbacks can segfault on certain platforms
- **Reason**: Upstream NumPy issues when invoked through Polars' FFI path (tracked in test skips)
- **Status**: PERMANENT — `tests/test_series_coverage.py::{test_mode,test_skew,test_kurt}` stay skipped until upstream fix
- **Workaround**: Use lazy queries or aggregate via Polars expressions where possible

## Recent Improvements (v0.9.0)

- **Rolling apply parity** — `DataFrame.rolling().apply` now calls `rolling_map`, honoring `raw`, `args`, `kwargs`, weights, centered windows, and `min_periods`.
- **GroupBy validation** — Missing columns are surfaced at aggregation time with pandas-style `KeyError` messages; helper attributes guard against invalid `_GroupBy` usage.
- **Tooling hygiene** — `mypy src/polarpandas`, `ruff check`, and the full pytest suite are part of release validation.

## Notes on Dtypes and Schema Conversion

- Polarpandas converts schemas provided as pandas-style strings, NumPy/pandas dtypes, or Polars schema to Polars dtypes.
- For Parquet/Feather, Polars does not accept a schema parameter at read time; types are cast after reading (or lazily for scans).
- When both `dtype` and `schema` are provided, `schema` takes precedence.
- `numpy` is an optional dependency, only needed if you pass NumPy dtype objects (e.g., `np.int64`) in schemas.

## API Coverage

### Current Implementation Status (v0.9.0)
- **619+ pandas-compatible features** implemented
- **69 module-level functions** — All major pandas functions
- **206 DataFrame methods** — Complete DataFrame API
- **186 Series methods** — Full Series functionality
- **73 Index methods** — Complete Index operations
- **57 String accessor methods** — Full `.str` accessor
- **28 Datetime accessor methods** — Comprehensive `.dt` accessor
- **91 LazyFrame methods** — Complete LazyFrame API

### Comprehensive I/O Support
- **Eager I/O**: CSV, JSON, Parquet, Excel, Feather, HDF5, HTML, XML, Stata, SPSS, SAS, Iceberg, ORC, Pickle, Clipboard
- **Lazy I/O**: CSV, Parquet, JSON (with schema/dtype support)
- **Export**: All major formats supported with optional dependencies

## Performance Considerations

### Polars Advantages
- **Speed**: Polars is significantly faster than pandas for most operations (5.2x faster on average)
- **Memory**: More efficient memory usage (50% less memory usage)
- **Parallelization**: Built-in parallel processing
- **Lazy Evaluation**: Optional lazy execution for maximum performance

### Pandas Fallbacks
- **Compatibility**: A small number of features may require pandas-like behavior; where used, it is documented
- **Performance Trade-off**: Any fallback paths may be slower than native Polars
- **Memory**: Fallback operations may use more memory due to conversion
- **Note**: Most operations now use native Polars implementations for maximum performance

## Recommendations

### For Users
1. Prefer native operations for performance-critical workloads
2. Use LazyFrame for large datasets (>1M rows) or complex operations
3. Be aware of dtype and JSON orient differences when comparing with pandas
4. Explicitly cast after operations that can produce mixed types (e.g., transpose)
5. Use schema conversion to standardize dtypes at read and construction time
6. Install optional dependencies only for the file formats you need

### For Developers
1. Prioritize native Polars implementations
2. Document any fallback behavior clearly
3. Consider performance implications when matching pandas semantics
4. Keep limitations synchronized with skipped tests and docs
5. Use the API compatibility matrix to track implementation status

## Future Improvements

### Short Term
1. Improve documentation/examples for schema conversion and casting after I/O
2. Expand JSON support within Polars constraints
3. Add guidance for MultiIndex workarounds
4. Increase test coverage toward >60%

### Long Term
1. Track upstream Polars enhancements for correlation/covariance and indexing
2. Explore ergonomic helpers for mixed-dtype workflows
3. Continue performance optimizations across common pandas APIs
4. Advanced MultiIndex support
5. Enhanced categorical operations

## Conclusion

Polarpandas delivers a pandas-compatible API with Polars performance for the vast majority of workflows. With **well over 619 pandas-compatible features** implemented, the library provides comprehensive coverage of the pandas API. Remaining limitations stem from fundamental differences in Polars' design and are documented via skipped tests and guidance above.

For the most up-to-date compatibility information, see the [API Compatibility Matrix](https://github.com/eddiethedean/polarpandas/blob/main/PANDAS_FUNCTION_MATRIX.md).
