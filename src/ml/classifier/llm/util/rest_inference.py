from pydantic import BaseModel, field_validator
from typing import List, Type, Union
from langchain import pydantic_v1

from src.ml.classifier.llm.util.rate_limit import RateLimitError
from src.ml.classifier.llm.util.logprob import LogProbScore
from src.ml.classifier.llm.util.rest import StructuredRequestLLM, AbstractLLM

class InferenceHandler(BaseModel, AbstractLLM):

    llms: List[Union[StructuredRequestLLM, str]]

    @field_validator("llms")
    @classmethod
    def _init_llms(cls, v):

        if not isinstance(v, list):
            raise ValueError("llms must be a list")
        
        result = []

        for llm in v:
            if isinstance(llm, StructuredRequestLLM):
                result.append(llm)
            elif isinstance(llm, str):
                result.append(StructuredRequestLLM.create_from_yaml_file(llm))
            else:
                raise ValueError("llms must be a list of strings or StructuredRequestLLM")

        return result

    def __call__(self, text: str, pydantic_model: Type[pydantic_v1.BaseModel], use_cache: bool = False, **kwargs) -> LogProbScore:

        result = None  # placeholder

        for llm in self.llms:

            if not isinstance(llm, StructuredRequestLLM):
                raise ValueError("llms must be a list of StructuredRequestLLM")
            
            try:
                
                result = llm(text, pydantic_model, use_cache, **kwargs)
            
            except RateLimitError:
                pass

            except Exception as e:
                raise e
            
        assert isinstance(result, LogProbScore)
        return result
