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

        raise ValueError(f"Object {obj} cannot be hashed")
    
    @staticmethod
    def hash_dict(dictionary: dict) -> str:  

        items = {k: v for k, v in dictionary.items()}

        return Hash.hash_recursive(**items)

    @staticmethod
    def hash_list(lst: list) -> str:
        return Hash.hash_recursive(*lst)
    
    @staticmethod
    def hash_tuple(tup: tuple) -> str:
        return Hash.hash_recursive(*tup)

    @staticmethod
    def hash_recursive(*args, **kwargs) -> str:
        
        hash_list = []
        
        for obj in args:

            if isinstance(obj, dict):
                hash_list.append(Hash.hash_dict(obj))
                continue

            if isinstance(obj, list):
                hash_list.append(Hash.hash_list(obj))
                continue

            if isinstance(obj, tuple):
                hash_list.append(Hash.hash_tuple(obj))
                continue

            hash_list.append(Hash.hash(obj))
        
        for key, value in kwargs.items():

            if isinstance(value, dict):
                hash_list.append(Hash.hash_dict(value))
                continue

            if isinstance(value, list):
                hash_list.append(Hash.hash_list(value))
                continue

            if isinstance(value, tuple):
                hash_list.append(Hash.hash_tuple(value))
                continue

            hash_list.append(Hash.hash(key))
            hash_list.append(Hash.hash(value))

        hash_str = ''.join(hash_list)
        
        return Hash.hash(hash_str)