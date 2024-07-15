import json
import requests
from pathlib import Path
from typing import Callable, Optional, List, Tuple
from pydantic import BaseModel

from src.util.caching import JsonCache
from src.util.hashing import Hash
from src.util.types import LogProb
from src.ml.classifier.llm.util.backoff import BackoffMixin
from src.ml.classifier.llm.util.request_utils import RequestInputData, RequestOutput
from src.ml.classifier.llm.api.base import AbstractLLM

class LLM(BaseModel, BackoffMixin, AbstractLLM):

    name: str
    request_input: RequestInputData
    request_output_formatter: Callable
    request_output: Optional[RequestOutput] = None

    def __call__(self, *, prompt: str, **kwargs) -> Tuple[str, List[LogProb]]:
        

        # prepare request data
        assert hasattr(self.request_input, 'prompt_key'), "Prompt key should be present in the request input"
        assert hasattr(self.request_input, 'data'), "Data should be present in the request input"
        assert isinstance(self.request_input.data, dict), "Data should be a dictionary" 
        
        request_kwargs = self.request_input.dict(exclude={'prompt_key', 'data'})
        request_data = self.request_input.data.copy()
        request_data[self.request_input.prompt_key] = prompt
    

        # send request if not cached
        hash_filename = Path(self.name) / (Hash.hash_string(prompt) + '.json')
        cache = JsonCache(filepath=hash_filename)

        cache_result = cache.read()

        if cache_result:

            self.request_output = RequestOutput(
                **cache_result
            )

            return self.request_output


        response = self.completion_with_backoff(
            requests.post,
            data=json.dumps(request_data),
            **request_kwargs
        )

        json_data = response.json()

        # cache the response
        cache.write(json_data)

        # parse to output object
        self.request_output = self.request_output_formatter(json_data)

        return self.request_output.text, self.request_output.logprobas
