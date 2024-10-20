import json
import requests  # type: ignore
from abc import ABC, abstractmethod
from typing import Callable, Optional
from pydantic import BaseModel
from pydantic.config import ConfigDict
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from src.util.hashing import Hash
from src.ml.util.backoff import BackoffMixin
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_hash_id(self, prompt: str) -> str:
        return f"{self.name}_{Hash.hash_string(prompt)}"

    
    def __call__(self, prompt, parser: Optional[PydanticOutputParser] = None, **kwargs) -> RequestOutput:

        request_kwargs = self.request_input.model_dump(
            exclude={'data', 'data_modifying_function'}
        )

        request_data = self.request_input.data.copy()

        if self.request_input.data_modifying_function is None:
            raise ValueError("Data modifying function not set")

        request_data = self.request_input.data_modifying_function(
            data=request_data,
            prompt_value=prompt
        )
    
        # send request if not cached
        hash_filename = self.get_hash_id(prompt)

        request_kwargs['data'] = json.dumps(request_data)

        save = True if parser is None else False

        response = self.completion_with_backoff_and_queue(
            function=requests.post,
            job_id=hash_filename,
            save=save,
            **request_kwargs
        )

        try:

            result = self.request_output_formatter(response.request_output)

            if not isinstance(result, RequestOutput):
                raise ValueError("Results needs to be a Request Output")
            
            # check if output is parsable and then save it
            if parser is not None:
                
                parser.parse(text=result.text)
                
                if not response.exists:
                    response.save()

        except OutputParserException as e:

                print('-'*10)
                print("Output parser failed: {}".format(str(e)))
                print(result.text)
                print('-'*10)
            
        except Exception as e:
            print(f"Error formatting request output: {e}")

        return result

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

