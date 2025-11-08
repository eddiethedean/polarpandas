"""
I/O operations for PolarPandas.

This module provides pandas-compatible functions for reading and writing data
in various formats. It supports both eager loading (immediate) and lazy loading
(deferred) operations for different file sizes and use cases.

Functions
---------
Eager I/O (immediate loading):
    read_csv, read_parquet, read_json, read_excel, read_sql, read_feather
    Load data immediately into DataFrame

Lazy I/O (deferred loading):
    scan_csv, scan_parquet, scan_json
    Load data lazily into LazyFrame for optimization

Examples
--------
>>> import polarpandas as ppd
>>> # Eager loading
>>> df = ppd.read_csv("data.csv")
>>> # Lazy loading for large files
>>> lf = ppd.scan_csv("large_file.csv")
>>> df = lf.collect()  # Materialize when ready

Notes
-----
- Use eager I/O for small to medium files (< 1M rows)
- Use lazy I/O for large files or when building complex query chains
- Lazy I/O allows Polars to optimize the query plan before execution
"""

import io
import re
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urlparse
from urllib.request import urlopen

from .frame import DataFrame
from .lazyframe import LazyFrame
from .utils import convert_schema_to_polars


def _read_text_source(source: Any) -> str:
    """
    Load textual content from a filesystem path, HTTP(S) URL, file-like object, or literal string.
    """

    if hasattr(source, "read"):
        return source.read()

    if isinstance(source, bytes):
        return source.decode("utf-8")

    if isinstance(source, str):
        stripped = source.strip()
        if stripped.lower().startswith("<"):
            return source

        parsed = urlparse(source)
        if parsed.scheme in {"http", "https"}:
            with urlopen(source) as response:  # pragma: no cover - network disabled in tests
                return response.read().decode("utf-8")

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"HTML source '{source}' does not exist.")
        return path.read_text(encoding="utf-8")

    raise ValueError(f"Unsupported HTML source type: {type(source)!r}")


def _extract_table_rows(table: Any) -> List[List[str]]:
    """
    Convert a BeautifulSoup <table> element to a list of string rows.
    """

    rows: List[List[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        row = [cell.get_text(strip=True) for cell in cells]
        rows.append(row)
    return rows


def _rows_to_frame(
    rows: List[List[str]],
    *,
    header: Optional[int],
    index_col: Optional[int],
    skiprows: Optional[Any],
) -> DataFrame:
    if not rows:
        return DataFrame()

    data_rows = rows
    header_row: Optional[List[str]] = None

    if header is not None:
        if isinstance(header, int):
            if header < 0 or header >= len(rows):
                raise ValueError(f"header index {header} is out of range for HTML table.")
            header_row = rows[header]
            data_rows = rows[:header] + rows[header + 1 :]
        else:
            raise NotImplementedError("header must be an integer when using read_html().")
    else:
        first = rows[0]
        header_row = first
        data_rows = rows[1:]

    if header_row is None:
        header_row = [f"column_{i}" for i in range(len(rows[0]))]

    skip_indexes: List[int]
    if skiprows is None:
        skip_indexes = []
    elif isinstance(skiprows, int):
        skip_indexes = list(range(skiprows))
    elif isinstance(skiprows, (list, tuple, set)):
        skip_indexes = sorted(skiprows)
    else:
        raise NotImplementedError("skiprows must be an int or sequence when using read_html().")

    filtered_rows = [
        row
        for idx, row in enumerate(data_rows)
        if idx not in skip_indexes
    ]

    width = max(len(header_row), *(len(row) for row in filtered_rows) or [0])
    normalized_header = header_row + [f"column_{i}" for i in range(len(header_row), width)]

    normalized_rows = [
        row + [None] * (width - len(row))
        for row in filtered_rows
    ]

    data = {normalized_header[i]: [row[i] for row in normalized_rows] for i in range(width)}
    frame = DataFrame(data)

    if index_col is not None and normalized_rows:
        if isinstance(index_col, int):
            column_keys = list(frame._df.columns)  # type: ignore[attr-defined]
            if index_col < 0 or index_col >= len(column_keys):
                raise ValueError("index_col is out of bounds for HTML table.")
            index_name = column_keys[index_col]
            frame._index = frame._df[index_name].to_list()
            frame._index_name = index_name
            frame._df = frame._df.drop(index_name)
        else:
            raise NotImplementedError("index_col must be an integer when using read_html().")

    return frame


def read_csv(path: str, **kwargs: Any) -> DataFrame:
    """
    Read a CSV file into DataFrame.

    Parameters
    ----------
    path : str
        Path to CSV file
    **kwargs
        Additional arguments passed to Polars read_csv()

    Returns
    -------
    DataFrame
        DataFrame loaded from CSV

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_csv("data.csv")
    """
    return DataFrame.read_csv(path, **kwargs)


def read_table(path: str, sep: str = "\t", **kwargs: Any) -> DataFrame:
    """
    Read general delimited file into DataFrame.

    This is a wrapper around read_csv() with different default separator.
    read_table() defaults to tab-separated values (TSV), while read_csv() defaults to comma.

    Parameters
    ----------
    path : str
        Path to delimited file
    sep : str, default '\\t'
        Delimiter to use. Default is tab character.
    **kwargs
        Additional arguments passed to Polars read_csv()

    Returns
    -------
    DataFrame
        DataFrame loaded from delimited file

    Examples
    --------
    >>> import polarpandas as ppd
    >>> # Read tab-separated file
    >>> df = ppd.read_table("data.tsv")
    >>> # Read pipe-separated file
    >>> df = ppd.read_table("data.txt", sep="|")
    """
    # read_table is essentially read_csv with different default separator
    # Pass sep to read_csv (it will convert to separator for Polars)
    if "sep" not in kwargs:
        kwargs["sep"] = sep
    return DataFrame.read_csv(path, **kwargs)


def read_parquet(path: str, **kwargs: Any) -> DataFrame:
    """
    Read a Parquet file into DataFrame.

    Parameters
    ----------
    path : str
        Path to Parquet file
    **kwargs
        Additional arguments passed to Polars read_parquet()

    Returns
    -------
    DataFrame
        DataFrame loaded from Parquet

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_parquet("data.parquet")
    """
    return DataFrame.read_parquet(path, **kwargs)


def read_json(path: str, **kwargs: Any) -> DataFrame:
    """
    Read a JSON file into DataFrame.

    Parameters
    ----------
    path : str
        Path to JSON file
    **kwargs
        Additional arguments passed to Polars read_json()

    Returns
    -------
    DataFrame
        DataFrame loaded from JSON

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_json("data.json")
    """
    return DataFrame.read_json(path, **kwargs)


def read_excel(path: str, **kwargs: Any) -> DataFrame:
    """
    Read an Excel file into DataFrame.

    Parameters
    ----------
    path : str
        Path to Excel file
    **kwargs
        Additional arguments passed to Polars read_excel()

    Returns
    -------
    DataFrame
        DataFrame loaded from Excel

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_excel("data.xlsx")
    """
    # Excel reading not yet implemented in polarpandas
    raise NotImplementedError(
        "read_excel() is not yet implemented.\n"
        "Workarounds:\n"
        "  - Use pandas: pd.read_excel(path) then convert with polarpandas.DataFrame(df)\n"
        "  - Export Excel to CSV/Parquet first, then use read_csv() or read_parquet()\n"
        "  - Use openpyxl/xlrd to read Excel, then create DataFrame from dict"
    )


def read_sql(sql: str, con: Any, **kwargs: Any) -> DataFrame:
    """
    Read SQL query into DataFrame.

    Parameters
    ----------
    sql : str
        SQL query string
    con : Any
        Database connection
    **kwargs
        Additional arguments passed to Polars read_sql()

    Returns
    -------
    DataFrame
        DataFrame loaded from SQL query

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_sql("SELECT * FROM table", connection)
    """
    return DataFrame.read_sql(sql, con, **kwargs)


def read_feather(path: str, **kwargs: Any) -> DataFrame:
    """
    Read a Feather file into DataFrame.

    Parameters
    ----------
    path : str
        Path to Feather file
    **kwargs
        Additional arguments passed to Polars read_ipc()

    Returns
    -------
    DataFrame
        DataFrame loaded from Feather

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_feather("data.feather")
    """
    return DataFrame.read_feather(path, **kwargs)


def scan_csv(path: str, **kwargs: Any) -> LazyFrame:
    """
    Scan a CSV file into LazyFrame for lazy execution.

    Parameters
    ----------
    path : str
        Path to CSV file
    dtype : dict, pl.Schema, or None, optional
        Schema specification for columns. Can be pandas-style dict or Polars schema.
        See DataFrame constructor for details.
    schema : dict, pl.Schema, or None, optional
        Direct Polars schema specification (alternative to dtype).
    **kwargs
        Additional arguments passed to Polars scan_csv()

    Returns
    -------
    LazyFrame
        LazyFrame loaded from CSV

    Examples
    --------
    >>> import polarpandas as ppd
    >>> lf = ppd.scan_csv("data.csv")
    >>> df = lf.collect()  # Materialize when ready
    """
    import polars as pl

    # Handle dtype/schema parameters
    dtype = kwargs.pop("dtype", None)
    schema = kwargs.pop("schema", None)

    # If both are provided, schema takes precedence
    schema_to_use = schema if schema is not None else dtype

    if schema_to_use is not None:
        polars_schema = convert_schema_to_polars(schema_to_use)
        if polars_schema is not None:
            # scan_csv accepts partial dtype mappings via schema overrides
            kwargs["schema_overrides"] = polars_schema

    return LazyFrame(pl.scan_csv(path, **kwargs))


def scan_parquet(path: str, **kwargs: Any) -> LazyFrame:
    """
    Scan a Parquet file into LazyFrame for lazy execution.

    Parameters
    ----------
    path : str
        Path to Parquet file
    dtype : dict, pl.Schema, or None, optional
        Schema specification for columns. Can be pandas-style dict or Polars schema.
        See DataFrame constructor for details.
    schema : dict, pl.Schema, or None, optional
        Direct Polars schema specification (alternative to dtype).
    **kwargs
        Additional arguments passed to Polars scan_parquet()

    Returns
    -------
    LazyFrame
        LazyFrame loaded from Parquet

    Examples
    --------
    >>> import polarpandas as ppd
    >>> lf = ppd.scan_parquet("data.parquet")
    >>> df = lf.collect()  # Materialize when ready
    """
    import polars as pl

    # Handle dtype/schema parameters
    dtype = kwargs.pop("dtype", None)
    schema = kwargs.pop("schema", None)

    # If both are provided, schema takes precedence
    schema_to_use = schema if schema is not None else dtype

    # Note: Parquet files don't support schema parameter in scan_parquet,
    # so we need to cast after scanning using lazy expressions
    lf = pl.scan_parquet(path, **kwargs)

    # Apply schema conversion if provided (cast columns using lazy expressions)
    if schema_to_use is not None:
        polars_schema = convert_schema_to_polars(schema_to_use)
        if polars_schema:
            # Build with_columns expression for casting
            cast_expressions = [
                pl.col(col).cast(dtype_val) for col, dtype_val in polars_schema.items()
            ]
            if cast_expressions:
                # Apply casts using lazy expressions
                # This will be evaluated during collect
                lf = lf.with_columns(cast_expressions)

    return LazyFrame(lf)


def scan_json(path: str, **kwargs: Any) -> LazyFrame:
    """
    Scan a JSON file into LazyFrame for lazy execution.

    Parameters
    ----------
    path : str
        Path to JSON file
    dtype : dict, pl.Schema, or None, optional
        Schema specification for columns. Can be pandas-style dict or Polars schema.
        See DataFrame constructor for details.
    schema : dict, pl.Schema, or None, optional
        Direct Polars schema specification (alternative to dtype).
    **kwargs
        Additional arguments passed to Polars scan_ndjson()

    Returns
    -------
    LazyFrame
        LazyFrame loaded from JSON

    Examples
    --------
    >>> import polarpandas as ppd
    >>> lf = ppd.scan_json("data.json")
    >>> df = lf.collect()  # Materialize when ready
    """
    import polars as pl

    # Handle dtype/schema parameters
    dtype = kwargs.pop("dtype", None)
    schema = kwargs.pop("schema", None)

    # If both are provided, schema takes precedence
    schema_to_use = schema if schema is not None else dtype

    # Scan JSON first (will infer types as strings if values are strings)
    lf = pl.scan_ndjson(path, **kwargs)

    # If dtype/schema is provided, cast columns to desired types
    if schema_to_use is not None:
        polars_schema = convert_schema_to_polars(schema_to_use)
        if polars_schema is not None:
            # Cast columns to desired types (handles string-to-numeric conversion)
            cast_exprs = [
                pl.col(col).cast(dtype) for col, dtype in polars_schema.items()
            ]
            lf = lf.with_columns(cast_exprs)

    return LazyFrame(lf)


def read_clipboard(sep: str = r"\s+", **kwargs: Any) -> DataFrame:
    """
    Read text from clipboard and pass to read_csv.

    Parameters
    ----------
    sep : str, default r"\\s+"
        Delimiter to use
    **kwargs
        Additional arguments passed to read_csv()

    Returns
    -------
    DataFrame
        DataFrame loaded from clipboard

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_clipboard()
    """
    try:
        import pyperclip
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise NotImplementedError(
            "read_clipboard() requires the 'pyperclip' package.\n"
            "Install it with: pip install pyperclip"
        ) from exc

    text = pyperclip.paste()
    if not text:
        raise ValueError("Clipboard is empty.")

    buffer = io.StringIO(text)
    return DataFrame.read_csv(buffer, sep=sep, **kwargs)


def read_fwf(filepath_or_buffer: Any, **kwargs: Any) -> DataFrame:
    """
    Read a table of fixed-width formatted lines into DataFrame.

    Parameters
    ----------
    filepath_or_buffer : str or file-like
        Path to file or file-like object
    **kwargs
        Additional arguments passed to Polars read_csv()

    Returns
    -------
    DataFrame
        DataFrame loaded from fixed-width file

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_fwf("data.txt")
    """
    # Fixed-width files can be read with read_csv using appropriate separator
    # This is a simplified implementation
    return DataFrame.read_csv(filepath_or_buffer, **kwargs)


def read_hdf(path_or_buf: Any, key: Optional[str] = None, **kwargs: Any) -> DataFrame:
    """
    Read HDF5 file into DataFrame.

    Parameters
    ----------
    path_or_buf : str or file-like
        Path to HDF5 file
    key : str, optional
        Identifier for the group in the HDF5 file
    **kwargs
        Additional arguments

    Returns
    -------
    DataFrame
        DataFrame loaded from HDF5

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_hdf("data.h5", key="table")
    """
    raise NotImplementedError(
        "read_hdf() is not yet implemented.\n"
        "Workarounds:\n"
        "  - Use pandas: pd.read_hdf(path, key) then convert with polarpandas.DataFrame(df)\n"
        "  - Export HDF5 to Parquet/CSV first, then use read_parquet() or read_csv()"
    )


def read_html(
    io: Any,
    match: str = ".+",
    flavor: Optional[str] = None,
    header: Optional[int] = None,
    index_col: Optional[int] = None,
    skiprows: Optional[Any] = None,
    attrs: Optional[Any] = None,
    parse_dates: bool = False,
    thousands: Optional[str] = None,
    encoding: Optional[str] = None,
    decimal: str = ".",
    converters: Optional[Any] = None,
    na_values: Optional[Any] = None,
    keep_default_na: bool = True,
    displayed_only: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Read HTML tables into a list of DataFrame objects.

    Parameters
    ----------
    io : str, path object, or file-like
        HTML string, file path, or file-like object
    match : str, default ".+"
        Regex to match table IDs
    flavor : str, optional
        Parser engine to use
    header : int, optional
        Row to use as column names
    index_col : int, optional
        Column to use as index
    skiprows : int or list, optional
        Rows to skip
    attrs : dict, optional
        Attributes to identify table
    parse_dates : bool, default False
        Parse dates
    thousands : str, optional
        Thousands separator
    encoding : str, optional
        Encoding to use
    decimal : str, default "."
        Decimal separator
    converters : dict, optional
        Converters for columns
    na_values : scalar or list, optional
        Values to recognize as NA
    keep_default_na : bool, default True
        Keep default NA values
    displayed_only : bool, default True
        Only parse displayed tables
    **kwargs
        Additional arguments

    Returns
    -------
    list of DataFrame
        List of DataFrames from HTML tables

    Examples
    --------
    >>> import polarpandas as ppd
    >>> dfs = ppd.read_html("page.html")
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise NotImplementedError(
            "read_html() requires 'beautifulsoup4' and an HTML parser such as 'lxml' or 'html5lib'.\n"
            "Install with: pip install beautifulsoup4 lxml html5lib"
        ) from exc

    if flavor is not None and flavor not in {"lxml", "html5lib", "bs4"}:
        raise NotImplementedError(f"Unsupported flavor '{flavor}'. Supported: lxml, html5lib, bs4.")

    html_text = _read_text_source(io)
    parser = flavor or "lxml"
    soup = BeautifulSoup(html_text, parser)

    search_attrs = attrs or {}
    tables = soup.find_all("table", attrs=search_attrs)

    pattern = re.compile(match)
    matched_tables = [
        table
        for table in tables
        if pattern.search(table.get_text(" ", strip=True)) is not None
    ]

    if not matched_tables:
        return []

    results: List[DataFrame] = []
    for table in matched_tables:
        rows = _extract_table_rows(table)
        frame = _rows_to_frame(
            rows,
            header=header,
            index_col=index_col,
            skiprows=skiprows,
        )
        results.append(frame)

    return results


def read_iceberg(path: str, **kwargs: Any) -> DataFrame:
    """
    Read Iceberg table into DataFrame.

    Parameters
    ----------
    path : str
        Path to Iceberg table
    **kwargs
        Additional arguments

    Returns
    -------
    DataFrame
        DataFrame loaded from Iceberg table

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_iceberg("s3://bucket/table")
    """
    raise NotImplementedError(
        "read_iceberg() is not yet implemented.\n"
        "Workarounds:\n"
        "  - Use Polars directly: pl.scan_iceberg(path).collect()\n"
        "  - Export Iceberg to Parquet first, then use read_parquet()"
    )


def read_orc(path: str, **kwargs: Any) -> DataFrame:
    """
    Read ORC file into DataFrame.

    Parameters
    ----------
    path : str
        Path to ORC file
    **kwargs
        Additional arguments passed to Polars read_orc()

    Returns
    -------
    DataFrame
        DataFrame loaded from ORC

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_orc("data.orc")
    """
    import polars as pl

    return DataFrame(pl.read_orc(path, **kwargs))  # type: ignore[attr-defined]


def read_pickle(
    filepath_or_buffer: Any,
    compression: Optional[str] = None,
    storage_options: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """
    Load pickled pandas object (or any object) from file.

    Parameters
    ----------
    filepath_or_buffer : str or file-like
        Path to pickle file
    compression : str, optional
        Compression type
    storage_options : dict, optional
        Storage options
    **kwargs
        Additional arguments

    Returns
    -------
    Any
        Unpickled object (DataFrame if it was a DataFrame)

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_pickle("data.pkl")
    """
    import pickle

    # Read pickle file
    if hasattr(filepath_or_buffer, "read"):
        obj = pickle.load(filepath_or_buffer, **kwargs)
    else:
        with open(filepath_or_buffer, "rb") as f:
            obj = pickle.load(f, **kwargs)

    # Convert DataFrame if it looks like a pandas object
    module_name = getattr(obj, "__class__", type(obj)).__module__
    class_name = getattr(obj, "__class__", type(obj)).__name__

    if module_name.startswith("polarpandas"):
        return obj

    if module_name.startswith("pandas"):
        if class_name == "DataFrame":
            data = obj.to_dict(orient="list")  # type: ignore[attr-defined]
            return DataFrame(data)
        if class_name == "Series":
            from .series import Series

            return Series(obj.tolist(), name=getattr(obj, "name", None))  # type: ignore[attr-defined]

    return obj


def read_sas(
    filepath_or_buffer: Any,
    format: Optional[str] = None,
    index: Optional[Any] = None,
    encoding: Optional[str] = None,
    chunksize: Optional[int] = None,
    iterator: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Read SAS files (.sas7bdat, .xport) into DataFrame.

    Parameters
    ----------
    filepath_or_buffer : str or file-like
        Path to SAS file
    format : str, optional
        File format
    index : str, optional
        Column to use as index
    encoding : str, optional
        Encoding to use
    chunksize : int, optional
        Number of rows to read per chunk
    iterator : bool, default False
        Return iterator
    **kwargs
        Additional arguments

    Returns
    -------
    DataFrame or iterator
        DataFrame loaded from SAS file

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_sas("data.sas7bdat")
    """
    if iterator or chunksize is not None:
        raise NotImplementedError("Chunked SAS reading is not supported without pandas.")
    if kwargs:
        unsupported = ", ".join(sorted(kwargs.keys()))
        raise NotImplementedError(f"Unsupported keyword arguments for read_sas(): {unsupported}")

    try:
        import pyreadstat
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise NotImplementedError(
            "read_sas() requires the 'pyreadstat' package.\n"
            "Install with: pip install pyreadstat"
        ) from exc

    chosen_format = (format or "").lower()
    if not chosen_format:
        path_str = str(filepath_or_buffer)
        if path_str.lower().endswith(".sas7bdat"):
            chosen_format = "sas7bdat"
        elif path_str.lower().endswith((".xpt", ".xport")):
            chosen_format = "xport"
        else:
            chosen_format = "sas7bdat"

    if chosen_format not in {"sas7bdat", "xport", "xpt"}:
        raise ValueError(f"Unsupported SAS format '{chosen_format}'.")

    if chosen_format == "sas7bdat":
        df, _ = pyreadstat.read_sas7bdat(filepath_or_buffer, encoding=encoding)
    else:
        df, _ = pyreadstat.read_xport(filepath_or_buffer, encoding=encoding)

    index_values: Optional[List[Any]] = None
    if index is not None:
        if index not in df.columns:
            raise ValueError(f"Index column '{index}' not found in SAS dataset.")
        index_values = df[index].tolist()
        df = df.drop(columns=[index])

    data = df.to_dict(orient="list")
    result = DataFrame(data)
    if index_values is not None:
        result._index = index_values
        result._index_name = index
    return result


def read_spss(
    filepath_or_buffer: Any,
    usecols: Optional[Any] = None,
    convert_categoricals: bool = True,
    **kwargs: Any,
) -> DataFrame:
    """
    Read SPSS file into DataFrame.

    Parameters
    ----------
    filepath_or_buffer : str or file-like
        Path to SPSS file
    usecols : list, optional
        Columns to read
    convert_categoricals : bool, default True
        Convert categorical variables
    **kwargs
        Additional arguments

    Returns
    -------
    DataFrame
        DataFrame loaded from SPSS file

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_spss("data.sav")
    """
    if kwargs:
        unsupported = ", ".join(sorted(kwargs.keys()))
        raise NotImplementedError(f"Unsupported keyword arguments for read_spss(): {unsupported}")

    try:
        import pyreadstat
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise NotImplementedError(
            "read_spss() requires the 'pyreadstat' package.\n"
            "Install with: pip install pyreadstat"
        ) from exc

    df, _ = pyreadstat.read_sav(
        filepath_or_buffer,
        usecols=usecols,
        apply_value_formats=convert_categoricals,
    )
    data = df.to_dict(orient="list")
    return DataFrame(data)


def read_sql_query(
    sql: str,
    con: Any,
    index_col: Optional[Any] = None,
    coerce_float: bool = True,
    params: Optional[Any] = None,
    parse_dates: Optional[Any] = None,
    chunksize: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """
    Read SQL query into DataFrame.

    Parameters
    ----------
    sql : str
        SQL query string
    con : Any
        Database connection
    index_col : str or list, optional
        Column(s) to use as index
    coerce_float : bool, default True
        Attempt to convert values to non-string, non-numeric objects to numeric
    params : list or dict, optional
        Parameters for SQL query
    parse_dates : list or dict, optional
        Columns to parse as dates
    chunksize : int, optional
        Number of rows to read per chunk
    **kwargs
        Additional arguments

    Returns
    -------
    DataFrame or iterator
        DataFrame loaded from SQL query

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_sql_query("SELECT * FROM table", connection)
    """
    # Delegate to read_sql
    return read_sql(
        sql=sql,
        con=con,
        index_col=index_col,
        coerce_float=coerce_float,
        params=params,
        parse_dates=parse_dates,
        chunksize=chunksize,
        **kwargs,
    )


def read_sql_table(
    table_name: str,
    con: Any,
    schema: Optional[str] = None,
    index_col: Optional[Any] = None,
    coerce_float: bool = True,
    parse_dates: Optional[Any] = None,
    columns: Optional[Any] = None,
    chunksize: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """
    Read SQL database table into DataFrame.

    Parameters
    ----------
    table_name : str
        Name of SQL table
    con : Any
        Database connection
    schema : str, optional
        Schema name
    index_col : str or list, optional
        Column(s) to use as index
    coerce_float : bool, default True
        Attempt to convert values to non-string, non-numeric objects to numeric
    parse_dates : list or dict, optional
        Columns to parse as dates
    columns : list, optional
        Columns to read
    chunksize : int, optional
        Number of rows to read per chunk
    **kwargs
        Additional arguments

    Returns
    -------
    DataFrame or iterator
        DataFrame loaded from SQL table

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_sql_table("table_name", connection)
    """
    # Build SQL query
    if schema:
        sql = f'SELECT * FROM "{schema}"."{table_name}"'
    else:
        sql = f'SELECT * FROM "{table_name}"'

    if columns:
        cols = ", ".join(f'"{col}"' for col in columns)
        sql = sql.replace("*", cols)

    return read_sql(
        sql=sql,
        con=con,
        index_col=index_col,
        coerce_float=coerce_float,
        parse_dates=parse_dates,
        chunksize=chunksize,
        **kwargs,
    )


def read_stata(
    filepath_or_buffer: Any,
    convert_dates: bool = True,
    convert_categoricals: bool = True,
    index_col: Optional[Any] = None,
    convert_missing: bool = False,
    preserve_dtypes: bool = True,
    columns: Optional[Any] = None,
    order_categoricals: bool = True,
    chunksize: Optional[int] = None,
    iterator: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Read Stata file into DataFrame.

    Parameters
    ----------
    filepath_or_buffer : str or file-like
        Path to Stata file
    convert_dates : bool, default True
        Convert date columns
    convert_categoricals : bool, default True
        Convert categorical variables
    index_col : str, optional
        Column to use as index
    convert_missing : bool, default False
        Convert missing values
    preserve_dtypes : bool, default True
        Preserve data types
    columns : list, optional
        Columns to read
    order_categoricals : bool, default True
        Order categorical variables
    chunksize : int, optional
        Number of rows to read per chunk
    iterator : bool, default False
        Return iterator
    **kwargs
        Additional arguments

    Returns
    -------
    DataFrame or iterator
        DataFrame loaded from Stata file

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_stata("data.dta")
    """
    if iterator or chunksize is not None:
        raise NotImplementedError("Chunked Stata reading is not supported without pandas.")
    if kwargs:
        unsupported = ", ".join(sorted(kwargs.keys()))
        raise NotImplementedError(f"Unsupported keyword arguments for read_stata(): {unsupported}")

    try:
        import pyreadstat
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise NotImplementedError(
            "read_stata() requires the 'pyreadstat' package.\n"
            "Install with: pip install pyreadstat"
        ) from exc

    df, _ = pyreadstat.read_dta(
        filepath_or_buffer,
        convert_dates=convert_dates,
        apply_value_formats=convert_categoricals,
        columns=columns,
    )

    if index_col is not None:
        if index_col not in df.columns:
            raise ValueError(f"Index column '{index_col}' not found in Stata dataset.")
        index_values = df[index_col].tolist()
        df = df.drop(columns=[index_col])
    else:
        index_values = None

    data = df.to_dict(orient="list")
    result = DataFrame(data)
    if index_values is not None:
        result._index = index_values
        result._index_name = index_col
    return result


def read_xml(
    path_or_buffer: Any,
    xpath: str = ".//",
    namespaces: Optional[Any] = None,
    elems_only: bool = False,
    attrs_only: bool = False,
    names: Optional[Any] = None,
    dtype: Optional[Any] = None,
    converters: Optional[Any] = None,
    parse_dates: Optional[Any] = None,
    encoding: Optional[str] = None,
    parser: str = "lxml",
    stylesheet: Optional[Any] = None,
    iterparse: Optional[Any] = None,
    compression: Optional[str] = None,
    storage_options: Optional[Any] = None,
    **kwargs: Any,
) -> DataFrame:
    """
    Read XML file into DataFrame.

    Parameters
    ----------
    path_or_buffer : str or file-like
        Path to XML file
    xpath : str, default ".//"
        XPath expression to select nodes
    namespaces : dict, optional
        Namespaces for XPath
    elems_only : bool, default False
        Parse only element nodes
    attrs_only : bool, default False
        Parse only attribute nodes
    names : list, optional
        Column names
    dtype : dict, optional
        Data types for columns
    converters : dict, optional
        Converters for columns
    parse_dates : list or dict, optional
        Columns to parse as dates
    encoding : str, optional
        Encoding to use
    parser : str, default "lxml"
        Parser to use
    stylesheet : str, optional
        XSLT stylesheet
    iterparse : dict, optional
        Iterparse configuration
    compression : str, optional
        Compression type
    storage_options : dict, optional
        Storage options
    **kwargs
        Additional arguments

    Returns
    -------
    DataFrame
        DataFrame loaded from XML

    Examples
    --------
    >>> import polarpandas as ppd
    >>> df = ppd.read_xml("data.xml")
    """
    if kwargs:
        unsupported = ", ".join(sorted(kwargs.keys()))
        raise NotImplementedError(f"Unsupported keyword arguments for read_xml(): {unsupported}")
    if parser not in {None, "lxml"}:
        raise NotImplementedError("Only the 'lxml' parser is supported.")
    if stylesheet is not None or iterparse is not None or compression is not None or storage_options is not None:
        raise NotImplementedError("Stylesheets, iterparse, compression, and storage options are not supported.")

    try:
        from lxml import etree
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise NotImplementedError(
            "read_xml() requires the 'lxml' package.\n"
            "Install with: pip install lxml"
        ) from exc

    text = _read_text_source(path_or_buffer)
    root = etree.fromstring(text.encode(encoding or "utf-8"))
    nodes = root.xpath(xpath, namespaces=namespaces)

    rows: List[Dict[str, Any]] = []
    for node in nodes:
        row: Dict[str, Any] = {}
        if not attrs_only:
            for child in node.iterchildren():
                if len(child):
                    # Nested element; store as stringified XML
                    row[child.tag] = etree.tostring(child, encoding="unicode")
                else:
                    row[child.tag] = child.text
        if not elems_only:
            for attr_key, attr_val in node.attrib.items():
                row[f"@{attr_key}"] = attr_val
        if row:
            rows.append(row)

    if not rows:
        return DataFrame()

    frame = DataFrame(rows)
    if names:
        selected = [name for name in names if name in frame.columns]
        if selected:
            frame = DataFrame({col: frame._df[col] for col in selected})  # type: ignore[attr-defined]
    return frame


def to_pickle(
    obj: Any,
    filepath_or_buffer: Any,
    compression: Optional[str] = None,
    protocol: Optional[int] = None,
    storage_options: Optional[Any] = None,
    **kwargs: Any,
) -> None:
    """
    Pickle (serialize) object to file.

    Parameters
    ----------
    obj : Any
        Object to pickle
    filepath_or_buffer : str or file-like
        Path to pickle file
    compression : str, optional
        Compression type
    protocol : int, optional
        Pickle protocol version
    storage_options : dict, optional
        Storage options
    **kwargs
        Additional arguments

    Examples
    --------
    >>> import polarpandas as ppd
    >>> ppd.to_pickle(df, "data.pkl")
    """
    import pickle

    # Convert polarpandas objects to pandas for better compatibility
    if hasattr(obj, "to_pandas"):
        obj = obj.to_pandas()

    # Write pickle file
    if hasattr(filepath_or_buffer, "write"):
        pickle.dump(obj, filepath_or_buffer, protocol=protocol, **kwargs)
    else:
        with open(filepath_or_buffer, "wb") as f:
            pickle.dump(obj, f, protocol=protocol, **kwargs)
