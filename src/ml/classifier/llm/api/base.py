import requests  # type: ignore
import json
import copy
from pydantic import BaseModel


from abc import abstractmethod, ABC
from typing import Any, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM



class BaseRemoteLLM(ABC):
    
    @abstractmethod
    def __call__(self, *, prompt: str, **kwargs) -> str:

        raise NotImplementedError()


class LangchainWrapper(LLM):

    name: str
    custom_model: Any

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

       return self.custom_model(prompt=prompt, stop=stop, run_manager=run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return self.name
    

class RequestInput(BaseModel):

    url: str
    headers: dict = dict()
    data: dict = dict()
    prompt_key: str


class RequestOutput(BaseModel):
    
    output_key: str

    @staticmethod
    def get_response_from_single_obj(x: Any, output_key: str, **kwargs) -> str:

        assert isinstance(x, dict), "Response should be a list of dictionaries"
        assert output_key in x, f"{output_key} is required for this function"

        result = x[output_key]

        assert isinstance(result, str), "Response should be a string"

        return result
    
    @staticmethod
    def get_response_from_list_of_obj(x: Any, output_key: str, **kwargs) -> str:

        assert isinstance(x, list), "Response should be a list"
        assert len(x) == 1, "Response should be a list of length 1"
        
        el = x[-1]

        text = RequestOutput.get_response_from_single_obj(x=el, output_key=output_key) 

        return text

    def format_response(self, x: Any, **kwargs) -> str:

        if isinstance(x, list):
            return self.get_response_from_list_of_obj(x=x, output_key=self.output_key, **kwargs)

        if isinstance(x, dict):
            return self.get_response_from_single_obj(x=x, output_key=self.output_key, **kwargs)
        
        raise ValueError("Response should be a list or a dictionary")
    

class LLamaRequestOutput(RequestOutput):

    def format_response(self, x: Any, **kwargs) -> str:
        
        raw_result = super().format_response(x, **kwargs)

        # extract json_str from the answer
        # Extract the JSON part from the text
        start_index = raw_result.rfind('{')
        end_index = raw_result.rfind('}') + 1

        return raw_result[start_index:end_index]
    


class LangchainWrapperRemote(LLM):

    name: str
    request_input: RequestInput
    request_output: RequestOutput

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        
        # work on copy
        data = copy.copy(self.request_input.data)

        # update data       
        data[self.request_input.prompt_key] = prompt
        
        response = requests.post(self.request_input.url, headers=self.request_input.headers, data=json.dumps(data))

        json_data = response.json()

        return self.request_output.format_response(x=json_data, **self.request_input.model_dump())


    @property
    def _llm_type(self) -> str:
        return self.name