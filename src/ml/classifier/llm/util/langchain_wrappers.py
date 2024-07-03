import requests  # type: ignore
import json
import copy

from typing import Any, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

from src.ml.classifier.llm.util.backoff import BackoffMixin
from src.ml.classifier.llm.util.request_utils import RequestInput, RequestOutput

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
    


class LangchainWrapperRemote(BackoffMixin, LLM):

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


        kwargs = {
            'url': self.request_input.url,
            'headers': self.request_input.headers,
            'data': json.dumps(data)
        }

        response = self.completion_with_backoff(
            requests.post,
            **kwargs
        )

        json_data = response.json()

        return self.request_output.format_response(x=json_data, **self.request_input.model_dump())


    @property
    def _llm_type(self) -> str:
        return self.name