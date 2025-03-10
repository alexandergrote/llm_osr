import sys
import random

from pydantic import BaseModel, field_validator, model_validator
from typing import List, Type, Union
from typing import Optional

from src.util.logger import console
from src.ml.classifier.llm.util.rate_limit import Action
from src.util.error import RateLimitException
from src.ml.classifier.llm.util.logprob import LogProbScore
from src.ml.classifier.llm.util.rest import StructuredRequestLLM, AbstractLLM


class InferenceHandler(BaseModel, AbstractLLM):

    free_llms: Optional[List[Union[StructuredRequestLLM, str]]] = None
    paid_llms: Optional[List[Union[StructuredRequestLLM, str]]] = None

    shuffle_free_llms: bool = False
    shuffle_paid_llms: bool = False

    action: Action = Action.EXIT

    @field_validator("action")
    @classmethod
    def _init_action(cls, v):

        if v != Action.EXIT:
            raise NotImplementedError("Currently, only exit action is supported")

        return v

    @model_validator(mode="after")
    def _check_llms(self):

        # check if both are None
        if (self.free_llms is None) and (self.paid_llms is None):
            raise ValueError("At least one llm must be provided")

        return self


    @field_validator("free_llms", "paid_llms")
    @classmethod
    def _init_llms(cls, v):

        if v is None:
            return None

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

    def __call__(self, text: str, pydantic_model: Type[BaseModel], use_cache: bool = False, **kwargs) -> LogProbScore:

        result = None  # placeholder

        # shuffle list
        llms = []

        if self.free_llms is not None:
            llms = self.free_llms.copy()

            llms = [el for el in llms if isinstance(el, StructuredRequestLLM) and el.check_rate_limit()]
            
            if self.shuffle_free_llms:
                random.shuffle(llms)
        
        if self.paid_llms is not None:
            paid_llms_copy = self.paid_llms.copy()

            paid_llms_copy = [el for el in paid_llms_copy if isinstance(el, StructuredRequestLLM) and el.check_rate_limit()]

            if self.shuffle_paid_llms:
                random.shuffle(paid_llms_copy)

            llms += paid_llms_copy

         # check if both are None
        if (self.free_llms is None) and (self.paid_llms is None):
            raise ValueError("At least one llm must be provided")


        for idx, llm in enumerate(llms):

            if not isinstance(llm, StructuredRequestLLM):
                raise ValueError("llms must be a list of StructuredRequestLLM")
            
            try:
                
                kwargs["llm_name"] = llm.name

                result = llm(text, pydantic_model, use_cache, **kwargs)
                assert isinstance(result, LogProbScore)
                break

            except RateLimitException:

                if idx != len(llms) - 1:
                    continue
                    
                console.log("All rate limit restrictions have been used. Exiting...")
                sys.exit(1)

            except Exception as e:
                raise e
            
        assert isinstance(result, LogProbScore)

        return result
