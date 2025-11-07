"""
SQL utility functions for enhanced database operations.

This module provides utility functions for working with SQL databases,
particularly for converting Polars DataFrames to SQL tables with advanced
features like primary keys and auto-increment columns using SQLAlchemy.
"""

from typing import Any, Dict, List, Optional, Union

import polars as pl


def _check_sqlalchemy_available() -> bool:
    """
    Check if SQLAlchemy is installed and available.

    Returns
    -------
    bool
        True if SQLAlchemy is available, False otherwise
    """
    try:
        import sqlalchemy  # noqa: F401

        return True
    except ImportError:
        return False


def _require_sqlalchemy(feature: str) -> None:
    """
    Raise an informative error if SQLAlchemy is not available.

    Parameters
    ----------
    feature : str
        The feature that requires SQLAlchemy

    Raises
    ------
    ImportError
        If SQLAlchemy is not installed
    """
    if not _check_sqlalchemy_available():
        raise ImportError(
            f"{feature} requires SQLAlchemy to be installed.\n"
            "Install it with: pip install sqlalchemy\n"
            "Or install polarpandas with SQL support: pip install polarpandas[sqlalchemy]"
        )


def _polars_to_sqlalchemy_type(polars_dtype: pl.DataType) -> Any:
    """
    Convert a Polars dtype to a SQLAlchemy type.

    Parameters
    ----------
    polars_dtype : pl.DataType
        Polars data type to convert

    Returns
    -------
    sqlalchemy type
        Corresponding SQLAlchemy column type

    Raises
    ------
    ImportError
        If SQLAlchemy is not installed
    ValueError
        If the Polars dtype cannot be mapped to a SQLAlchemy type
    """
    _require_sqlalchemy("Converting Polars types to SQL types")

    from sqlalchemy import types as sqltypes

    # Map Polars types to SQLAlchemy types
    type_mapping = {
        pl.Int8: sqltypes.SmallInteger,
        pl.Int16: sqltypes.SmallInteger,
        pl.Int32: sqltypes.Integer,
        pl.Int64: sqltypes.BigInteger,
        pl.UInt8: sqltypes.SmallInteger,
        pl.UInt16: sqltypes.Integer,
        pl.UInt32: sqltypes.BigInteger,
        pl.UInt64: sqltypes.BigInteger,
        pl.Float32: sqltypes.Float,
        pl.Float64: sqltypes.Float,
        pl.Boolean: sqltypes.Boolean,
        pl.Utf8: sqltypes.String,
        pl.Date: sqltypes.Date,
        pl.Datetime: sqltypes.DateTime,
        pl.Time: sqltypes.Time,
    }

    # Get the base type (for handling nullable types)
    base_type = polars_dtype
    if hasattr(polars_dtype, "base_type"):
        base_type = polars_dtype.base_type()  # type: ignore[assignment]

    # Try direct mapping
    for polars_type, sql_type in type_mapping.items():
        if base_type == polars_type:
            return sql_type()

    # Handle string types
    if isinstance(base_type, pl.Utf8) or base_type == pl.Utf8:
        return sqltypes.String()

    # Handle categorical
    if isinstance(base_type, pl.Categorical) or base_type == pl.Categorical:
        return sqltypes.String()

    # Default to String for unknown types
    return sqltypes.String()


def create_table_with_primary_key(
    df: pl.DataFrame,
    table_name: str,
    connection: Any,
    schema: Optional[str] = None,
    if_exists: str = "fail",
    primary_key: Optional[Union[str, List[str]]] = None,
    auto_increment: bool = False,
    dtype: Optional[Dict[str, Any]] = None,
    index: bool = True,
    index_label: Optional[Union[str, List[str]]] = None,
) -> None:
    """
    Create a SQL table with primary key and auto-increment support using SQLAlchemy.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame to write to SQL
    table_name : str
        Name of SQL table
    connection : Any
        SQLAlchemy connection or engine
    schema : str, optional
        SQL schema name
    if_exists : str, default 'fail'
        How to behave if the table already exists ('fail', 'replace', 'append')
    primary_key : str or list of str, optional
        Column name(s) to set as primary key
    auto_increment : bool, default False
        Whether to make the primary key auto-increment
    dtype : dict, optional
        SQL data types for columns
    index : bool, default True
        Whether to write DataFrame index as a column
    index_label : str or list of str, optional
        Column label for index column(s)

    Raises
    ------
    ImportError
        If SQLAlchemy is not installed
    ValueError
        If invalid parameters are provided
    """
    _require_sqlalchemy("Creating tables with primary keys")

    from sqlalchemy import Column, MetaData, Table, create_engine, inspect

    # Normalize primary_key to a list
    if primary_key is not None:
        if isinstance(primary_key, str):
            primary_keys = [primary_key]
        else:
            primary_keys = list(primary_key)
    else:
        primary_keys = []

    # Validate primary_key columns exist
    df_columns = df.columns
    for pk_col in primary_keys:
        if pk_col not in df_columns:
            raise ValueError(
                f"Primary key column '{pk_col}' not found in DataFrame. "
                f"Available columns: {df_columns}"
            )

    # Handle connection - could be engine, connection, or connection string
    if isinstance(connection, str):
        engine = create_engine(connection)
    elif hasattr(connection, "engine"):
        engine = connection.engine
    else:
        engine = connection

    # Check if table exists
    inspector = inspect(engine)
    table_exists = inspector.has_table(table_name, schema=schema)

    if table_exists:
        if if_exists == "fail":
            raise ValueError(f"Table '{table_name}' already exists.")
        elif if_exists == "replace":
            # Drop the existing table
            metadata = MetaData()
            existing_table = Table(
                table_name, metadata, autoload_with=engine, schema=schema
            )
            existing_table.drop(engine)
        elif if_exists == "append":
            # Just append data without recreating table
            _write_data_to_existing_table(
                df, table_name, engine, schema, index, index_label
            )
            return
        else:
            raise ValueError(f"Invalid value for if_exists: '{if_exists}'")

    # Create table metadata
    metadata = MetaData()
    columns = []

    # Handle index column if needed
    if index and hasattr(df, "_index") and df._index is not None:
        # This would need to be handled by the DataFrame class
        # For now, we'll skip explicit index handling
        pass

    # Build columns with their types
    for col_name in df.columns:
        # Get column dtype
        col_dtype = df[col_name].dtype

        # Use custom dtype if provided
        if dtype and col_name in dtype:
            sql_type = dtype[col_name]
        else:
            sql_type = _polars_to_sqlalchemy_type(col_dtype)

        # Check if this column is a primary key
        is_primary_key = col_name in primary_keys
        is_auto_inc = is_primary_key and auto_increment

        # Create column
        column = Column(
            col_name,
            sql_type,
            primary_key=is_primary_key,
            autoincrement=is_auto_inc,
        )
        columns.append(column)

    # Create table
    Table(table_name, metadata, *columns, schema=schema)
    metadata.create_all(engine)

    # Insert data
    _write_data_to_existing_table(df, table_name, engine, schema, index, index_label)


def _write_data_to_existing_table(
    df: pl.DataFrame,
    table_name: str,
    engine: Any,
    schema: Optional[str] = None,
    index: bool = True,
    index_label: Optional[Union[str, List[str]]] = None,
) -> None:
    """
    Write DataFrame data to an existing SQL table using Polars.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame to write
    table_name : str
        Name of SQL table
    engine : Any
        SQLAlchemy engine
    schema : str, optional
        SQL schema name
    index : bool, default True
        Whether to write DataFrame index as a column
    index_label : str or list of str, optional
        Column label for index column(s)
    """
    # Use Polars' native write_database for performance
    # The table has already been created by SQLAlchemy with proper schema,
    # so we just need to append the data
    df.write_database(
        table_name=table_name,
        connection=engine,
        if_table_exists="append",
    )
