"""
Tests for enhanced SQL functionality including primary keys and auto-increment.

This module tests the new to_sql parameters (primary_key, auto_increment)
that require SQLAlchemy as an optional dependency.
"""

import os
import tempfile

import polars as pl
import pytest

import polarpandas as ppd

# Check if SQLAlchemy is available
try:
    from sqlalchemy import create_engine, inspect

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# Allow deselection via `-m "not requires_sqlalchemy"` when the dependency isn't present.
pytestmark = pytest.mark.requires_sqlalchemy


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return ppd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "email": [
                "alice@test.com",
                "bob@test.com",
                "charlie@test.com",
                "david@test.com",
                "eve@test.com",
            ],
        }
    )


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not installed")
class TestToSQLWithPrimaryKey:
    """Test to_sql with primary key support."""

    def test_simple_primary_key(self, sample_df, temp_db):
        """Test creating a table with a simple primary key."""
        engine = create_engine(f"sqlite:///{temp_db}")

        # Write DataFrame with primary key
        sample_df.to_sql(
            "users",
            engine,
            if_exists="replace",
            primary_key="id",
        )

        # Verify table was created
        inspector = inspect(engine)
        assert inspector.has_table("users")

        # Check primary key constraint
        pk = inspector.get_pk_constraint("users")
        assert "id" in pk["constrained_columns"]

        # Verify data was written
        result = ppd.read_sql("SELECT * FROM users ORDER BY id", engine)
        assert len(result) == 5
        assert list(result["id"]) == [1, 2, 3, 4, 5]

    def test_composite_primary_key(self, sample_df, temp_db):
        """Test creating a table with a composite primary key."""
        engine = create_engine(f"sqlite:///{temp_db}")

        # Write DataFrame with composite primary key
        sample_df.to_sql(
            "users",
            engine,
            if_exists="replace",
            primary_key=["id", "email"],
        )

        # Verify table was created
        inspector = inspect(engine)
        assert inspector.has_table("users")

        # Check primary key constraint
        pk = inspector.get_pk_constraint("users")
        assert "id" in pk["constrained_columns"]
        assert "email" in pk["constrained_columns"]

        # Verify data was written
        result = ppd.read_sql("SELECT * FROM users ORDER BY id", engine)
        assert len(result) == 5

    def test_primary_key_with_auto_increment(self, temp_db):
        """Test creating a table with auto-increment primary key."""
        engine = create_engine(f"sqlite:///{temp_db}")

        # Create DataFrame without id column (will be auto-generated)
        df = ppd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
            }
        )

        # Write DataFrame with auto-increment primary key
        df.to_sql(
            "users",
            engine,
            if_exists="replace",
            primary_key="id",
            auto_increment=True,
        )

        # Verify table was created
        inspector = inspect(engine)
        assert inspector.has_table("users")

        # Check primary key constraint
        pk = inspector.get_pk_constraint("users")
        assert "id" in pk["constrained_columns"]

        # Verify data was written
        result = ppd.read_sql("SELECT * FROM users ORDER BY id", engine)
        assert len(result) == 3

    def test_primary_key_if_exists_fail(self, sample_df, temp_db):
        """Test that if_exists='fail' raises error when table exists."""
        engine = create_engine(f"sqlite:///{temp_db}")

        # Write DataFrame first time
        sample_df.to_sql(
            "users",
            engine,
            if_exists="replace",
            primary_key="id",
        )

        # Try to write again with if_exists='fail'
        with pytest.raises(ValueError, match="already exists"):
            sample_df.to_sql(
                "users",
                engine,
                if_exists="fail",
                primary_key="id",
            )

    def test_primary_key_if_exists_replace(self, sample_df, temp_db):
        """Test that if_exists='replace' replaces the table."""
        engine = create_engine(f"sqlite:///{temp_db}")

        # Write DataFrame first time
        sample_df.to_sql(
            "users",
            engine,
            if_exists="replace",
            primary_key="id",
        )

        # Write again with different data and if_exists='replace'
        new_df = ppd.DataFrame(
            {
                "id": [10, 20],
                "name": ["New1", "New2"],
                "age": [50, 60],
                "email": ["new1@test.com", "new2@test.com"],
            }
        )

        new_df.to_sql(
            "users",
            engine,
            if_exists="replace",
            primary_key="id",
        )

        # Verify only new data exists
        result = ppd.read_sql("SELECT * FROM users ORDER BY id", engine)
        assert len(result) == 2
        assert list(result["id"]) == [10, 20]

    def test_primary_key_if_exists_append(self, sample_df, temp_db):
        """Test that if_exists='append' appends to existing table."""
        engine = create_engine(f"sqlite:///{temp_db}")

        # Write DataFrame first time
        sample_df.to_sql(
            "users",
            engine,
            if_exists="replace",
            primary_key="id",
        )

        # Write more data with if_exists='append'
        new_df = ppd.DataFrame(
            {
                "id": [6, 7],
                "name": ["Frank", "Grace"],
                "age": [50, 55],
                "email": ["frank@test.com", "grace@test.com"],
            }
        )

        new_df.to_sql(
            "users",
            engine,
            if_exists="append",
            primary_key="id",
        )

        # Verify both datasets exist
        result = ppd.read_sql("SELECT * FROM users ORDER BY id", engine)
        assert len(result) == 7
        assert list(result["id"]) == [1, 2, 3, 4, 5, 6, 7]

    def test_primary_key_invalid_column(self, sample_df, temp_db):
        """Test that invalid primary key column raises error."""
        engine = create_engine(f"sqlite:///{temp_db}")

        with pytest.raises(ValueError, match="not found in DataFrame"):
            sample_df.to_sql(
                "users",
                engine,
                if_exists="replace",
                primary_key="nonexistent_column",
            )

    def test_auto_increment_without_primary_key(self, sample_df, temp_db):
        """Test that auto_increment without primary_key raises error."""
        engine = create_engine(f"sqlite:///{temp_db}")

        with pytest.raises(ValueError, match="auto_increment requires primary_key"):
            sample_df.to_sql(
                "users",
                engine,
                if_exists="replace",
                auto_increment=True,
            )


class TestToSQLWithoutSQLAlchemy:
    """Test to_sql without SQLAlchemy features (basic Polars functionality)."""

    def test_basic_to_sql_without_primary_key(self, sample_df, temp_db):
        """Test basic to_sql without primary key works with Polars."""
        # This should work even if we're using basic Polars functionality
        # We'll use a connection string that Polars can handle
        try:
            import sqlalchemy

            engine = sqlalchemy.create_engine(f"sqlite:///{temp_db}")

            # Write without primary_key should use Polars' write_database
            sample_df.to_sql("users", engine, if_exists="replace")

            # Verify data was written
            result = ppd.read_sql("SELECT * FROM users", engine)
            assert len(result) == 5
        except ImportError:
            pytest.skip("SQLAlchemy not available for this test")

    def test_if_exists_validation(self, sample_df, temp_db):
        """Test that invalid if_exists value raises error."""
        try:
            import sqlalchemy

            engine = sqlalchemy.create_engine(f"sqlite:///{temp_db}")

            with pytest.raises(ValueError, match="not valid for if_exists"):
                sample_df.to_sql("users", engine, if_exists="invalid_option")
        except ImportError:
            pytest.skip("SQLAlchemy not available for this test")


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not installed")
class TestSeriesToSQL:
    """Test Series.to_sql with enhanced features."""

    def test_series_with_primary_key(self, temp_db):
        """Test writing Series to SQL with primary key."""
        engine = create_engine(f"sqlite:///{temp_db}")

        series = ppd.Series([10, 20, 30, 40], name="values")

        # Write Series (it will become a single-column DataFrame)
        series.to_sql(
            "measurements",
            engine,
            if_exists="replace",
            primary_key="values",
        )

        # Verify table was created
        inspector = inspect(engine)
        assert inspector.has_table("measurements")

        # Check primary key
        pk = inspector.get_pk_constraint("measurements")
        assert "values" in pk["constrained_columns"]


@pytest.mark.skipif(SQLALCHEMY_AVAILABLE, reason="Test SQLAlchemy not available")
class TestSQLAlchemyNotInstalled:
    """Test error messages when SQLAlchemy is required but not installed."""

    def test_primary_key_without_sqlalchemy(self, sample_df, temp_db):
        """Test that helpful error is raised when SQLAlchemy needed but not installed."""
        # This test only runs when SQLAlchemy is NOT installed
        with pytest.raises(ImportError, match="requires SQLAlchemy"):
            sample_df.to_sql(
                "users",
                f"sqlite:///{temp_db}",
                primary_key="id",
            )


class TestToSQLTypes:
    """Test to_sql with different data types."""

    @pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not installed")
    def test_mixed_types_with_primary_key(self, temp_db):
        """Test to_sql with mixed data types and primary key."""
        engine = create_engine(f"sqlite:///{temp_db}")

        df = ppd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "score": [85.5, 92.3, 78.9],
                "passed": [True, True, False],
            }
        )

        df.to_sql(
            "students",
            engine,
            if_exists="replace",
            primary_key="id",
        )

        # Verify data types were preserved
        result = ppd.read_sql("SELECT * FROM students ORDER BY id", engine)
        assert len(result) == 3
        assert result["name"][0] == "Alice"
        assert result["passed"][0] == 1  # SQLite stores booleans as integers


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not installed")
class TestToSQLEdgeCases:
    """Test to_sql with edge cases."""

    def test_empty_dataframe_with_primary_key(self, temp_db):
        """Test writing an empty DataFrame with primary key."""
        engine = create_engine(f"sqlite:///{temp_db}")

        df = ppd.DataFrame(
            {
                "id": [],
                "name": [],
                "age": [],
            }
        )

        df.to_sql(
            "empty_users",
            engine,
            if_exists="replace",
            primary_key="id",
        )

        # Verify table was created but is empty
        inspector = inspect(engine)
        assert inspector.has_table("empty_users")

        result = ppd.read_sql("SELECT * FROM empty_users", engine)
        assert len(result) == 0

    def test_single_row_dataframe(self, temp_db):
        """Test writing a single row DataFrame."""
        engine = create_engine(f"sqlite:///{temp_db}")

        df = ppd.DataFrame(
            {
                "id": [1],
                "name": ["Alice"],
                "age": [25],
            }
        )

        df.to_sql(
            "single_user",
            engine,
            if_exists="replace",
            primary_key="id",
        )

        result = ppd.read_sql("SELECT * FROM single_user", engine)
        assert len(result) == 1
        assert result["name"][0] == "Alice"

    def test_single_column_dataframe(self, temp_db):
        """Test writing a single column DataFrame."""
        engine = create_engine(f"sqlite:///{temp_db}")

        df = ppd.DataFrame({"values": [1, 2, 3, 4, 5]})

        df.to_sql(
            "single_column",
            engine,
            if_exists="replace",
            primary_key="values",
        )

        result = ppd.read_sql("SELECT * FROM single_column", engine)
        assert len(result) == 5
        assert list(result["values"]) == [1, 2, 3, 4, 5]

    def test_large_dataframe(self, temp_db):
        """Test writing a large DataFrame."""
        engine = create_engine(f"sqlite:///{temp_db}")

        # Create a DataFrame with 10,000 rows
        n = 10000
        df = ppd.DataFrame(
            {
                "id": list(range(n)),
                "value": [i * 2 for i in range(n)],
            }
        )

        df.to_sql(
            "large_table",
            engine,
            if_exists="replace",
            primary_key="id",
        )

        result = ppd.read_sql("SELECT COUNT(*) as count FROM large_table", engine)
        assert result["count"][0] == n

    def test_unicode_strings(self, temp_db):
        """Test writing Unicode strings."""
        engine = create_engine(f"sqlite:///{temp_db}")

        df = ppd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice 中文", "Bob русский", "Charlie 日本語"],
            }
        )

        df.to_sql(
            "unicode_test",
            engine,
            if_exists="replace",
            primary_key="id",
        )

        result = ppd.read_sql("SELECT * FROM unicode_test ORDER BY id", engine)
        assert result["name"][0] == "Alice 中文"
        assert result["name"][1] == "Bob русский"
        assert result["name"][2] == "Charlie 日本語"

    def test_special_characters_in_column_names(self, temp_db):
        """Test columns with spaces and special characters."""
        engine = create_engine(f"sqlite:///{temp_db}")

        df = ppd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "user_name": ["Alice", "Bob", "Charlie"],
                "email_address": ["a@test.com", "b@test.com", "c@test.com"],
            }
        )

        df.to_sql(
            "special_cols",
            engine,
            if_exists="replace",
            primary_key="user_id",
        )

        result = ppd.read_sql("SELECT * FROM special_cols", engine)
        assert len(result) == 3


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not installed")
class TestToSQLDatetimeTypes:
    """Test to_sql with date and time types."""

    def test_date_column(self, temp_db):
        """Test writing date columns."""
        engine = create_engine(f"sqlite:///{temp_db}")

        df = ppd.DataFrame(
            {
                "id": [1, 2, 3],
                "event_date": pl.Series(
                    ["2023-01-01", "2023-06-15", "2023-12-31"]
                ).str.strptime(pl.Date, "%Y-%m-%d"),
            }
        )

        df.to_sql(
            "events",
            engine,
            if_exists="replace",
            primary_key="id",
        )

        result = ppd.read_sql("SELECT * FROM events ORDER BY id", engine)
        assert len(result) == 3

    def test_datetime_column(self, temp_db):
        """Test writing datetime columns."""
        engine = create_engine(f"sqlite:///{temp_db}")

        df = ppd.DataFrame(
            {
                "id": [1, 2, 3],
                "timestamp": pl.Series(
                    [
                        "2023-01-01 10:00:00",
                        "2023-06-15 15:30:00",
                        "2023-12-31 23:59:59",
                    ]
                ).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
            }
        )

        df.to_sql(
            "logs",
            engine,
            if_exists="replace",
            primary_key="id",
        )

        result = ppd.read_sql("SELECT * FROM logs ORDER BY id", engine)
        assert len(result) == 3


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not installed")
class TestToSQLNullHandling:
    """Test to_sql with null values."""

    def test_null_values_in_non_key_columns(self, temp_db):
        """Test writing null values in non-primary key columns."""
        engine = create_engine(f"sqlite:///{temp_db}")

        df = ppd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "name": ["Alice", None, "Charlie", "David"],
                "age": [25, 30, None, 40],
            }
        )

        df.to_sql(
            "users_with_nulls",
            engine,
            if_exists="replace",
            primary_key="id",
        )

        result = ppd.read_sql("SELECT * FROM users_with_nulls ORDER BY id", engine)
        assert len(result) == 4


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not installed")
class TestToSQLMultipleTables:
    """Test to_sql with multiple tables."""

    def test_multiple_tables_same_db(self, temp_db):
        """Test writing multiple tables to the same database."""
        engine = create_engine(f"sqlite:///{temp_db}")

        users_df = ppd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
            }
        )

        orders_df = ppd.DataFrame(
            {
                "order_id": [101, 102, 103],
                "user_id": [1, 2, 1],
                "amount": [50.0, 75.5, 120.0],
            }
        )

        users_df.to_sql("users", engine, if_exists="replace", primary_key="user_id")
        orders_df.to_sql("orders", engine, if_exists="replace", primary_key="order_id")

        # Verify both tables exist
        inspector = inspect(engine)
        assert inspector.has_table("users")
        assert inspector.has_table("orders")

        # Verify we can query both
        users_result = ppd.read_sql("SELECT * FROM users", engine)
        orders_result = ppd.read_sql("SELECT * FROM orders", engine)

        assert len(users_result) == 3
        assert len(orders_result) == 3

    def test_table_operations_sequence(self, temp_db):
        """Test a sequence of table operations."""
        engine = create_engine(f"sqlite:///{temp_db}")

        df1 = ppd.DataFrame({"id": [1, 2], "value": [10, 20]})

        # Create table
        df1.to_sql("test_table", engine, if_exists="replace", primary_key="id")

        # Append data
        df2 = ppd.DataFrame({"id": [3, 4], "value": [30, 40]})
        df2.to_sql("test_table", engine, if_exists="append", primary_key="id")

        # Verify all data
        result = ppd.read_sql("SELECT * FROM test_table ORDER BY id", engine)
        assert len(result) == 4
        assert list(result["id"]) == [1, 2, 3, 4]

        # Replace table
        df3 = ppd.DataFrame({"id": [100], "value": [1000]})
        df3.to_sql("test_table", engine, if_exists="replace", primary_key="id")

        # Verify only new data
        result = ppd.read_sql("SELECT * FROM test_table", engine)
        assert len(result) == 1
        assert result["id"][0] == 100


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not installed")
class TestToSQLValidation:
    """Test to_sql validation and error handling."""

    def test_composite_primary_key_with_auto_increment(self, temp_db):
        """Test that composite primary key with auto_increment raises error."""
        engine = create_engine(f"sqlite:///{temp_db}")

        df = ppd.DataFrame(
            {
                "id1": [1, 2, 3],
                "id2": ["A", "B", "C"],
                "value": [10, 20, 30],
            }
        )

        # SQLite doesn't support auto_increment on composite keys
        # This should raise a CompileError from SQLAlchemy
        from sqlalchemy.exc import CompileError

        with pytest.raises(CompileError, match="autoincrement.*composite"):
            df.to_sql(
                "composite_test",
                engine,
                if_exists="replace",
                primary_key=["id1", "id2"],
                auto_increment=True,
            )

    def test_invalid_table_name_characters(self, temp_db):
        """Test table names with special characters."""
        engine = create_engine(f"sqlite:///{temp_db}")

        df = ppd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

        # SQLite allows most characters in table names when quoted
        df.to_sql(
            "test_table_123",
            engine,
            if_exists="replace",
            primary_key="id",
        )

        inspector = inspect(engine)
        assert inspector.has_table("test_table_123")

    def test_duplicate_table_creation(self, temp_db):
        """Test creating same table twice with different strategies."""
        engine = create_engine(f"sqlite:///{temp_db}")

        df = ppd.DataFrame({"id": [1, 2], "value": [10, 20]})

        # First creation
        df.to_sql("dup_table", engine, if_exists="replace", primary_key="id")

        # Try to create again with fail
        with pytest.raises(ValueError, match="already exists"):
            df.to_sql("dup_table", engine, if_exists="fail", primary_key="id")


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not installed")
class TestToSQLNumericTypes:
    """Test to_sql with various numeric types."""

    def test_integer_types(self, temp_db):
        """Test different integer types."""
        engine = create_engine(f"sqlite:///{temp_db}")

        df = ppd.DataFrame(
            {
                "id": [1, 2, 3],
                "small_int": pl.Series([1, 2, 3], dtype=pl.Int8),
                "medium_int": pl.Series([100, 200, 300], dtype=pl.Int16),
                "large_int": pl.Series([10000, 20000, 30000], dtype=pl.Int64),
            }
        )

        df.to_sql(
            "int_types",
            engine,
            if_exists="replace",
            primary_key="id",
        )

        result = ppd.read_sql("SELECT * FROM int_types", engine)
        assert len(result) == 3

    def test_float_types(self, temp_db):
        """Test different float types."""
        engine = create_engine(f"sqlite:///{temp_db}")

        df = ppd.DataFrame(
            {
                "id": [1, 2, 3],
                "float32": pl.Series([1.5, 2.5, 3.5], dtype=pl.Float32),
                "float64": pl.Series(
                    [10.123456, 20.789012, 30.345678], dtype=pl.Float64
                ),
            }
        )

        df.to_sql(
            "float_types",
            engine,
            if_exists="replace",
            primary_key="id",
        )

        result = ppd.read_sql("SELECT * FROM float_types", engine)
        assert len(result) == 3

    def test_boolean_type(self, temp_db):
        """Test boolean type."""
        engine = create_engine(f"sqlite:///{temp_db}")

        df = ppd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "active": [True, False, True, False],
                "verified": [False, False, True, True],
            }
        )

        df.to_sql(
            "bool_types",
            engine,
            if_exists="replace",
            primary_key="id",
        )

        result = ppd.read_sql("SELECT * FROM bool_types ORDER BY id", engine)
        assert len(result) == 4
        # SQLite stores booleans as integers
        assert result["active"][0] == 1
        assert result["active"][1] == 0


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not installed")
class TestToSQLConnectionTypes:
    """Test to_sql with different connection types."""

    def test_connection_string(self, temp_db):
        """Test using a connection string instead of engine."""
        connection_string = f"sqlite:///{temp_db}"

        df = ppd.DataFrame(
            {
                "id": [1, 2, 3],
                "value": [10, 20, 30],
            }
        )

        df.to_sql(
            "from_string",
            connection_string,
            if_exists="replace",
            primary_key="id",
        )

        # Verify with engine
        engine = create_engine(connection_string)
        result = ppd.read_sql("SELECT * FROM from_string", engine)
        assert len(result) == 3

    def test_engine_object(self, temp_db):
        """Test using an engine object."""
        engine = create_engine(f"sqlite:///{temp_db}")

        df = ppd.DataFrame(
            {
                "id": [1, 2, 3],
                "value": [10, 20, 30],
            }
        )

        df.to_sql(
            "from_engine",
            engine,
            if_exists="replace",
            primary_key="id",
        )

        result = ppd.read_sql("SELECT * FROM from_engine", engine)
        assert len(result) == 3


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not installed")
class TestToSQLBatchOperations:
    """Test to_sql with batch operations."""

    def test_multiple_appends(self, temp_db):
        """Test multiple append operations."""
        engine = create_engine(f"sqlite:///{temp_db}")

        # Initial data
        df1 = ppd.DataFrame({"id": [1], "value": [10]})
        df1.to_sql("batch_test", engine, if_exists="replace", primary_key="id")

        # Append multiple times
        for i in range(2, 6):
            df = ppd.DataFrame({"id": [i], "value": [i * 10]})
            df.to_sql("batch_test", engine, if_exists="append", primary_key="id")

        # Verify all data
        result = ppd.read_sql("SELECT * FROM batch_test ORDER BY id", engine)
        assert len(result) == 5
        assert list(result["id"]) == [1, 2, 3, 4, 5]

    def test_replace_after_append(self, temp_db):
        """Test replace after multiple appends."""
        engine = create_engine(f"sqlite:///{temp_db}")

        # Create and append
        df1 = ppd.DataFrame({"id": [1, 2], "value": [10, 20]})
        df1.to_sql("replace_test", engine, if_exists="replace", primary_key="id")

        df2 = ppd.DataFrame({"id": [3, 4], "value": [30, 40]})
        df2.to_sql("replace_test", engine, if_exists="append", primary_key="id")

        # Verify 4 rows
        result = ppd.read_sql("SELECT COUNT(*) as count FROM replace_test", engine)
        assert result["count"][0] == 4

        # Replace with new data
        df3 = ppd.DataFrame({"id": [100], "value": [1000]})
        df3.to_sql("replace_test", engine, if_exists="replace", primary_key="id")

        # Verify only 1 row
        result = ppd.read_sql("SELECT COUNT(*) as count FROM replace_test", engine)
        assert result["count"][0] == 1
