import importlib
import yaml
from pydantic import BaseModel
from pydantic.v1.utils import deep_update
from typing import Iterable, Tuple, Union

from src.util.constants import YamlField


def convert(value, type_):

    # see https://stackoverflow.com/questions/7402573/use-type-information-to-cast-values-stored-as-strings

    
    try:
        # Check if it's a builtin type
        module = importlib.import_module('builtins')
        cls = getattr(module, type_)

    except AttributeError:
        # if not, separate module and class
        module, type_ = type_.rsplit(".", 1)
        module = importlib.import_module(module)
        cls = getattr(module, type_)
        
    return cls(value)


def count_keys(dictionary):
    count = 0
    for _, value in dictionary.items():
        count += 1  # Increment count for each key encountered
        if isinstance(value, dict):
            count += count_keys(value)  # Recursively count keys in nested dictionaries
    return count


class DictExtraction(BaseModel):

    @staticmethod
    def get_class_obj_and_params(dictionary: dict) -> Tuple[str, dict]:

        class_obj = dictionary[YamlField.CLASS_NAME.value]
        class_params = dictionary.get(YamlField.PARAMS.value)

        if class_params is None:
            return class_obj, {}

        return class_obj, class_params

    @staticmethod
    def get_class_obj_and_params_from_yaml(filename: str, overriding: Iterable[str] = ()):

        """
        args:
            - filename: str 
            - overriding: Tuple[str], 
                must contain a = "=" seperating the key value pair
                can also be an empty list
        """

        with open(filename) as f:
            dictionary = yaml.safe_load(f)

        # number of keys before update
        n_keys_before = count_keys(dictionary=dictionary)

        # go through all overriding values
        for el in overriding:

            # seperate value from hierarchy
            hierarchy, value = tuple(el.split('='))

            # format value
            value, dtype = tuple(value.split(':'))

            value = convert(value, dtype)

            # build hierachy and save value in dict
            tree_dict: Union[str, dict] = value
            for key in reversed(hierarchy.split('.')):
                tree_dict = {key: tree_dict}

            dictionary = deep_update(dictionary, tree_dict)

            n_keys_after = count_keys(dictionary=dictionary)

            if n_keys_before != n_keys_after:
                raise ValueError(f"Number of keys differ: {n_keys_before} != {n_keys_after} ")

        return DictExtraction.get_class_obj_and_params(
            dictionary=dictionary
        )
        