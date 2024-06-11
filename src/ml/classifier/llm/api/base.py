import requests  # type: ignore
import json

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

class LangchainWrapperRemote(LLM):

    name: str
    url: str

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
       
       data = {
            "prompt": prompt
       }
       
       response = requests.post(self.url, data=json.dumps(data))

       return response.json()["response"]


    @property
    def _llm_type(self) -> str:
        return self.name