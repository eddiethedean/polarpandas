# Polarpandas Known Limitations

This document outlines the known limitations and differences between polarpandas and pandas that are due to fundamental differences between Polars and pandas libraries.

## Current Test Status

- **498 passing tests**
- **54 skipped tests** (documented limitations)
- **72% code coverage**

## Permanent Limitations (Pure Polars Constraints)

Polarpandas is a pure Polars library and does not depend on pandas. Some pandas behaviors cannot be replicated exactly:

### 1. Correlation/Covariance
- **Issue**: No native `corr()`/`cov()` parity
- **Reason**: Polars lacks built-in `corr()`/`cov()` methods identical to pandas
- **Status**: PERMANENT — tracked via skipped tests

### 2. Transpose with Mixed Dtypes
- **Issue**: Mixed dtypes after transpose behave differently
- **Reason**: Polars promotes to a common supertype; pandas preserves object-like behavior
- **Workaround**: Avoid mixed-type transpose or cast explicitly after
- **Status**: PERMANENT — behavior differs by design; some tests skipped

### 3. MultiIndex Support
- **Issue**: Limited/absent MultiIndex parity
- **Reason**: Polars has no native MultiIndex concept
- **Workaround**: Use concatenated columns or struct types as a surrogate
- **Status**: PERMANENT — architectural limitation

### 4. JSON Orient Formats
- **Issue**: Some pandas JSON orient formats are unsupported
- **Reason**: Polars supports a subset of pandas JSON orients
- **Status**: PERMANENT — documented skips where applicable

## Notes on Dtypes and Schema Conversion

- Polarpandas converts schemas provided as pandas-style strings, NumPy/pandas dtypes, or Polars schema to Polars dtypes.
- For Parquet/Feather, Polars does not accept a schema parameter at read time; types are cast after reading (or lazily for scans).
- When both `dtype` and `schema` are provided, `schema` takes precedence.
- `numpy` is an optional dependency, only needed if you pass NumPy dtype objects (e.g., `np.int64`) in schemas.

## Previously Listed Limitations (Now Addressed)

- Outdated failing-test counts and pandas fallbacks for core features have been removed. The current suite reports 498 passing and 54 skipped, with no failing tests.

## Performance Considerations

### Polars Advantages
- **Speed**: Polars is significantly faster than pandas for most operations
- **Memory**: More efficient memory usage
- **Parallelization**: Built-in parallel processing

### Pandas Fallbacks
- **Compatibility**: A small number of features may require pandas-like behavior; where used, it is documented
- **Performance Trade-off**: Any fallback paths may be slower than native Polars
- **Memory**: Fallback operations may use more memory due to conversion

## Recommendations

### For Users
1. Prefer native operations for performance-critical workloads
2. Be aware of dtype and JSON orient differences when comparing with pandas
3. Explicitly cast after operations that can produce mixed types (e.g., transpose)
4. Use schema conversion to standardize dtypes at read and construction time

### For Developers
1. Prioritize native Polars implementations
2. Document any fallback behavior clearly
3. Consider performance implications when matching pandas semantics
4. Keep limitations synchronized with skipped tests and docs

## Future Improvements

### Short Term
1. Improve documentation/examples for schema conversion and casting after I/O
2. Expand JSON support within Polars constraints
3. Add guidance for MultiIndex workarounds

### Long Term
1. Track upstream Polars enhancements for correlation/covariance and indexing
2. Explore ergonomic helpers for mixed-dtype workflows
3. Continue performance optimizations across common pandas APIs

## Conclusion

Polarpandas delivers a pandas-compatible API with Polars performance for the vast majority of workflows. Remaining limitations stem from fundamental differences in Polars’ design and are documented via skipped tests and guidance above.
