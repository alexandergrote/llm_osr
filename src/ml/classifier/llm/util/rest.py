import json
import requests  # type: ignore
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional
from pydantic import BaseModel

from src.util.caching import JsonCache
from src.util.hashing import Hash
from src.ml.classifier.llm.util.backoff import BackoffMixin
from src.ml.classifier.llm.util.request import RequestInputData, RequestOutput


class AbstractLLM(ABC):
    
    @abstractmethod
    def __call__(self, *, prompt: str, **kwargs) -> RequestOutput:

        raise NotImplementedError()


class LLM(BaseModel, AbstractLLM, BackoffMixin):

    name: str
    request_input: RequestInputData
    request_output_formatter: Callable
    request_output: Optional[RequestOutput] = None

    class Config:
        arbitrary_types_allowed = True

    def __call__(self, *, prompt: str, **kwargs) -> RequestOutput:
        
        request_kwargs = self.request_input.model_dump(exclude={'data', 'data_modifying_function'})
        request_data = self.request_input.data.copy()
        
        if self.request_input.data_modifying_function is None:
            raise ValueError("Data modifying function not set")

        request_data = self.request_input.data_modifying_function(
            data=request_data,
            prompt_value=prompt
        )
    
        # send request if not cached
        hash_filename = Path(self.name) / (Hash.hash_string(prompt) + '.json')
        
        cache = JsonCache(filepath=hash_filename)

        json_data = cache.read()

        if json_data is None:

            response = self.completion_with_backoff(
                requests.post,
                data=json.dumps(request_data),
                **request_kwargs
            )

            json_data = response.json()

            # cache the response
            cache.write(json_data)

        return self.request_output_formatter(json_data)


if __name__ == '__main__':

    from src.ml.classifier.llm.util.request import RequestInputData, RequestOutput
    from src.util.constants import LLMModels, RESTAPI_URLS

    llama = LLM(
        name='llama-3b',
        request_input=RequestInputData.create_hf_llama_request_input(
            url=RESTAPI_URLS[LLMModels.LLAMA_3_8B_Remote_HF]
        ),
        request_output_formatter=RequestOutput.from_llama_hf_request
    )

    gpt35 = LLM(
        name='gpt-3.5-turbo',
        request_input=RequestInputData.create_openai_request_input(
            name='gpt-3.5-turbo-0125'
        ),
        request_output_formatter=RequestOutput.from_openai_request
    )

    for model in [llama, gpt35]:

        prompt = 'hi, who are you? answer with one word with this json schema: {"answer": <your reply>}'

        result = model(prompt=prompt)

        print(result)

