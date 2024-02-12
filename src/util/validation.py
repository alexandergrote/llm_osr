import pandas as pd
from typing import Iterable, Optional

class DataFrameValidator:

    @classmethod
    def assert_non_zero_dataframe(cls, data: pd.DataFrame, n_rows: Optional[int] = None, columns: Iterable[str] = [], strict_columns: bool = True, identifier: Optional[str] = None):
        
        identifier = f"{identifier}: " if identifier else ""

        assert isinstance(data, pd.DataFrame), f"{identifier}Data is not a dataframe."
        cls._check_columns(data, columns, strict_columns, identifier=identifier)
        cls._check_rows(data, n_rows, identifier=identifier)

    @classmethod
    def _check_rows(cls, data: pd.DataFrame, n_rows: Optional[int], identifier: str):

        n_rows_df = len(data)

        if n_rows:
            assert n_rows > 0, f"{identifier}Number of rows must be positive."
            msg = f"{identifier}Output data does not contain the correct number of rows. Expected {n_rows}, got {data.shape[0]}."
            assert n_rows_df == n_rows, msg

        assert n_rows_df > 0, f"{identifier}Output data is empty."
        

        
    @classmethod
    def _check_columns(cls, data: pd.DataFrame, columns: Iterable[str], strict_columns: bool, identifier: str):
        
        if strict_columns:
            assert len(set(data.columns) - set(columns)) == 0, f"{identifier}Output data does not exclusively contain the specified columns."

        for column in data.columns:
            assert column in columns, f"{identifier}Dataframe does not contain column {column}."
