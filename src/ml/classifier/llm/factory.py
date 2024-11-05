from pydantic import BaseModel, model_validator
from typing import Union, Callable

from src.ml.classifier.llm.base import AbstractClassifierLLM
from src.ml.classifier.llm.util.rest import LLM as RestLLM
from src.ml.classifier.llm.util.request import RequestInputData
from src.util.dynamic_import import DynamicImport


class LLMClassifierFactory(BaseModel):

    """
    This class is used to create a LLM classifier.
    Its secondary purpose is to provide type checking for the model_config used in any yaml file
    """

    name: str
    request_output_formatter: Union[Callable, dict]
    request_input: Union[RequestInputData, dict]

    function_key: str = 'function'
    function_key_params: str = 'params'

    @model_validator(mode='after')
    def _init_output_formatter(self):

        key = 'request_output_formatter'

        if not isinstance(self.request_output_formatter, dict):
            raise ValueError(f"{key} must be a dict")
        
        self.request_output_formatter = DynamicImport.get_attribute_from_class(
            self.request_output_formatter[self.function_key]
        )

        key = 'request_input'

        if not isinstance(self.request_input, dict):
            raise ValueError(f"{key} must be a dict")

        function_input = DynamicImport.get_attribute_from_class(
            self.request_input[self.function_key]
        )

        self.request_input = function_input(**self.request_input[self.function_key_params])

        return self

    def create(self) -> AbstractClassifierLLM:

        llm = RestLLM(
            name=self.name,
            request_output_formatter=self.request_output_formatter,
            request_input=self.request_input
        )
        
        return llm
