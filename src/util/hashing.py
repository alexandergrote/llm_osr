import joblib
import hashlib

import pandas as pd

from typing import Any, Union
from pydantic import BaseModel


class Hash(BaseModel):

    @staticmethod
    def hash_dataframe(data: Union[pd.DataFrame, pd.Series]) -> str:
        return joblib.hash(data)

    @staticmethod
    def hash_string(string: str) -> str:
        return hashlib.sha1(str.encode(string)).hexdigest()

    @staticmethod
    def hash(obj: Any) -> str:

        if isinstance(obj, pd.DataFrame):
            return Hash.hash_dataframe(obj)

        if isinstance(obj, pd.Series):
            return Hash.hash_dataframe(obj)

        if isinstance(obj, str):
            return Hash.hash_string(obj)

        if obj is None:
            return Hash.hash_string('None')

        if isinstance(obj, int):
            return str(obj.__hash__())

        if isinstance(obj, float):
            return str(obj.__hash__())

        try:
            obj_str = str(obj)
        except Exception:
            obj_str = ''

        raise ValueError(f"Object {obj_str} cannot be hashed")