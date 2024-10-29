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
from src.ml.util.job_queue import JobStatus


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

        # we need the parser to conduct an addtional test on the output
        # only if this test is passed, we can then save the output
        save = True if parser is None else False

        # rest api response
        response = self.completion_with_backoff_and_queue(
            function=requests.post,
            job_id=hash_filename,
            save=save,
            **request_kwargs
        )

        if response.status == JobStatus.failed:

            return RequestOutput(
                text=response.error_description,
                error=True,
                logprobas=[]
            )


        # parse response
        # first parsing: request output will be parsed to a pydantic object
        # second parsing: output needs to follow the parser object
        # the first step is needed to manually exclude parts of the output that lead to errors with the pydantic output parser
        first_parsed_response = None   
        
        try:

            first_parsed_response: RequestOutput = self.request_output_formatter(response.request_output)

        except Exception as e:

            error_message = f"Exception {str(e)};\n\nError parsing request output: {response.error_description}"

            return RequestOutput(
                text=error_message,
                error=True,
                logprobas=[]
            )

        if not isinstance(first_parsed_response, RequestOutput):
            raise ValueError("Results needs to be a Request Output")
            
        # check if output is parsable and then save it
        try:

            if parser is not None:
                
                parser.parse(text=first_parsed_response.text)
                
                if not response.exists:
                    response.save()

            return first_parsed_response

        except OutputParserException as e:

            error_message = f"Output parser failed: {first_parsed_response.text}"
            
            return RequestOutput(
                text=error_message,
                error=True,
                logprobas=[]
            )
            
        except Exception as e:

            error_message = f"Error formatting request output: {e}"

            return RequestOutput(
                text=error_message,
                error=True,
                logprobas=[]
            )


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

