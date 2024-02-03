from typing import Optional

from pydantic import BaseModel

from src.util.dict_extraction import DictExtraction


class DynamicImport(BaseModel):
    
    @staticmethod
    def _get_class_obj(name: str):
        # index of last dot
        dot_index = name.rfind(".")
        class_name = name[dot_index + 1 :]

        mod = __import__(name[:dot_index], fromlist=[class_name])
        class_object = getattr(mod, class_name)

        return class_object

    @staticmethod
    def init_class(name: Optional[str], params: Optional[dict]):

        if name is None:
            return None

        class_object = DynamicImport._get_class_obj(name=name)

        if params is None:
            return class_object()

        return class_object(**params)
    
    @staticmethod
    def init_class_from_dict(dictionary: dict):

        args = DictExtraction.get_class_obj_and_params(
            dictionary=dictionary
        )

        return DynamicImport.init_class(*args)
    
    @staticmethod
    def init_class_from_yaml(filename: str, **kwargs):

        args = DictExtraction.get_class_obj_and_params_from_yaml(
        filename=filename, **kwargs)

        return DynamicImport.init_class(*args)