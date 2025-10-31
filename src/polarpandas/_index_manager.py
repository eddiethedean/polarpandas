"""
Index management utilities for PolarPandas.

This module provides utilities for managing DataFrame index preservation
across operations, centralizing logic that would otherwise be duplicated.
"""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import polars as pl

if TYPE_CHECKING:
    from polarpandas.frame import DataFrame


class IndexManager:
    """
    Manager for DataFrame index operations.

    Centralizes index preservation logic used across many DataFrame methods
    to avoid code duplication and ensure consistent behavior.
    """

    @staticmethod
    def preserve_index(
        source_df: "DataFrame", result_df: pl.DataFrame, preserve_name: bool = True
    ) -> "DataFrame":
        """
        Preserve index when creating a new DataFrame from a Polars DataFrame.

        Parameters
        ----------
        source_df : DataFrame
            Source DataFrame with index to preserve
        result_df : pl.DataFrame
            Resulting Polars DataFrame (may have different shape/columns)
        preserve_name : bool, default True
            Whether to preserve the index name

        Returns
        -------
        DataFrame
            New DataFrame with preserved index

        Examples
        --------
        >>> result_pl = df._df.filter(...)
        >>> result = IndexManager.preserve_index(df, result_pl)
        """
        from polarpandas.frame import DataFrame

        result = DataFrame(result_df)
        # Preserve index if source has one and shapes match
        if source_df._index is not None and len(result_df) == len(source_df._index):
            result._index = source_df._index
            if preserve_name:
                result._index_name = source_df._index_name
        return result

    @staticmethod
    def preserve_index_inplace(df: "DataFrame", new_df: pl.DataFrame) -> None:
        """
        Update DataFrame in place while preserving index.

        Parameters
        ----------
        df : DataFrame
            DataFrame to update in place
        new_df : pl.DataFrame
            New Polars DataFrame to replace internal _df

        Examples
        --------
        >>> IndexManager.preserve_index_inplace(self, filtered_df)
        """
        # Only preserve index if shapes match
        if df._index is not None and len(new_df) == len(df._index):
            # Index preserved automatically if lengths match
            df._df = new_df
        else:
            df._df = new_df
            # Reset index if shape changed significantly
            if len(new_df) != len(df._index) if df._index else True:
                df._index = None
                df._index_name = None

    @staticmethod
    def extract_index_for_rows(
        df: "DataFrame", row_indices: List[int]
    ) -> Optional[List[Any]]:
        """
        Extract index values for specific row indices.

        Parameters
        ----------
        df : DataFrame
            Source DataFrame
        row_indices : List[int]
            Row indices to extract

        Returns
        -------
        Optional[List[Any]]
            Index values for specified rows, or None if no index
        """
        if df._index is None:
            return None
        return [df._index[i] for i in row_indices if 0 <= i < len(df._index)]

    @staticmethod
    def create_index_from_columns(
        df: pl.DataFrame, columns: List[str]
    ) -> Tuple[List[Any], Optional[str]]:
        """
        Create index from DataFrame columns.

        Parameters
        ----------
        df : pl.DataFrame
            Polars DataFrame to extract from
        columns : List[str]
            Column names to use for index

        Returns
        -------
        Tuple[List[Any], Optional[str]]
            Tuple of (index_values, index_name)
        """
        if len(columns) == 1:
            index_values = df[columns[0]].to_list()
            index_name: Optional[str] = columns[0]
        else:
            # Multi-level index - create tuples
            index_values = list(zip(*[df[col].to_list() for col in columns]))
            index_name = None  # Multi-index names handled separately
        return index_values, index_name

    @staticmethod
    def validate_index_length(index: Optional[List[Any]], data_length: int) -> bool:
        """
        Validate that index length matches data length.

        Parameters
        ----------
        index : Optional[List[Any]]
            Index to validate
        data_length : int
            Expected data length

        Returns
        -------
        bool
            True if index is None or length matches
        """
        if index is None:
            return True
        return len(index) == data_length
