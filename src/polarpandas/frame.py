"""
A DataFrame object that behaves like a pandas DataFrame
but using polars DataFrame to do all the work.
"""
import polars as pl


class DataFrame(pl.DataFrame):
    def __setitem__(self, column: str, values) -> None:
        ...
        #self.with_columns([(values).alias(column)])