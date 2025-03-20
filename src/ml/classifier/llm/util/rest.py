import json
import requests  # type: ignore
from abc import ABC, abstractmethod
from typing import Callable, Optional, Any, Union, Type
from pydantic import BaseModel, model_validator, field_validator
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from langchain.output_parsers import PydanticOutputParser
from pathlib import Path

from src.util.hashing import Hash
from src.ml.classifier.llm.util.request import RequestOutput, RequestInput
from src.ml.classifier.llm.util.logprob import LogProbScore
from src.ml.classifier.llm.util.prediction import Prediction
from src.ml.util.job_queue import Job, JobStatus
from src.ml.classifier.llm.util.rate_limit import RateLimitManager
from src.ml.classifier.llm.util.tokenizer import LlamaTokenizer
from src.util.dynamic_import import DynamicImport
from src.util.logger import log_pydantic_error, delete_error_log, console, get_logging_fun
from src.util.error import LLMError, RateLimitException, UnknownAPIException
from src.util.constants import Directory
from src.util.caching import JsonCache


def traced_request(*args, **kwargs):
    return requests.post(*args, **kwargs)


class AbstractLLM(ABC):
    
    @abstractmethod
    def __call__(self, text: str, pydantic_model: Type[BaseModel], use_cache: bool = False, **kwargs) -> LogProbScore:

        raise NotImplementedError()
    
    @classmethod
    def create_from_yaml_file(cls, filename: str) -> "AbstractLLM":

        filepath = Directory.CONFIG / "llm" 
        filepath = filepath / filename

        if not filepath.exists():
            raise LLMError(f"File {filepath} does not exist")
        
        obj = DynamicImport.init_class_from_yaml(
            filename=str(filepath),
        )

        return obj


class StructuredRequestLLM(BaseModel, AbstractLLM):

    name: str
    rest_api_model_name: str
    url: str
    payload: dict = {}

    rate_limit_manager: Optional[Union[RateLimitManager, str]] = None

    # functions to call formatting options
    request_input_classmethod: Union[Callable, str]
    request_output_classmethod: Union[Callable, str]
    request_input_data_extraction: Union[Callable, str]

    @field_validator("request_input_classmethod", "request_input_data_extraction")
    @classmethod
    def _init_input(cls, v: Any):

        if not isinstance(v, str):
            msg = "Initial value must be a string, will be transformed into callable later"
            raise ValueError(msg)
        
        if not hasattr(RequestInput, v):
            raise ValueError(f"Attribute {v} not found in RequestInput")
        
        return getattr(RequestInput, v)

    @field_validator("request_output_classmethod")
    @classmethod
    def _init_output(cls, v: Any):

        if not isinstance(v, str):
            msg = "Initial value must be a string, will be transformed into callable later"
            raise ValueError(msg)
        
        if not hasattr(RequestOutput, v):
            raise ValueError(f"Attribute {v} not found in RequestOutput")
        
        return getattr(RequestOutput, v)

    @model_validator(mode='before')
    @classmethod
    def _init_rate_limits(cls, values: Any):

        if not isinstance(values, dict):
            raise ValueError("Values must be a dictionary")    

        key = "rate_limit_manager"
        rate_limit_manager_params = values.get(key)

        if rate_limit_manager_params is None:
            return values
        
        is_str = isinstance(rate_limit_manager_params, str)
        is_rlm = isinstance(rate_limit_manager_params, RateLimitManager)

        if is_str:

            obj = RateLimitManager.create_from_config_file(
                filename=rate_limit_manager_params,
                init_from_disk=True
            )

        elif is_rlm:

            obj = rate_limit_manager_params
            obj.load()
            obj.save()

        else:
            raise ValueError(f"Value of {key} must be a string or a rate limit manager directly")

        assert isinstance(obj, RateLimitManager)
        
        values[key] = obj
        
        return values

    def _get_hash_id(self, prompt: str) -> str:
        return f"{self.name}_{Hash.hash_string(prompt)}"
    
    def _get_log_filename(self, prompt: str) -> str:
        return f"{self._get_hash_id(prompt=prompt)}.json"

    def check_rate_limit(self) -> bool:

        if self.rate_limit_manager is None:
            return True
        
        if isinstance(self.request_input_data_extraction, str):
            raise ValueError("Request input class method is not callable")

        if not isinstance(self.rate_limit_manager, RateLimitManager):
            raise ValueError("Rate limit manager is not a RateLimitManager object")

        self.rate_limit_manager.load()

        rate_limits = self.rate_limit_manager.rate_limits

        for (_, rate_limit) in rate_limits.items():

            if not rate_limit.check():
                return False

        return True 

    def _check_rate_limit_based_on_request(self, request_dict: dict, **kwargs):

        if self.rate_limit_manager is None:
            return 

        if not isinstance(self.rate_limit_manager, RateLimitManager):
            raise ValueError("Rate limit manager is not a RateLimitManager object")
        
        if isinstance(self.request_input_data_extraction, str):
            raise ValueError("Request input class method is not callable")
        
        self.rate_limit_manager.load()

        prompt = self.request_input_data_extraction(request_dict)

        # simple heuristic to estimate the number of tokens used
        # +20 because the simple tokenizer underestimates the real tokens
        tokenizer = LlamaTokenizer()
        num_tokens = len(tokenizer.encode(prompt)) + 20

        self.rate_limit_manager.check_execution(
            num_request_tokens=num_tokens,
            **kwargs
        )
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5), reraise=True, retry=retry_if_exception_type(UnknownAPIException))
    def _backoff(self, job: Job, request_fun: Callable, request_dict: dict, **kwargs) -> Job:

        """
        This function ensures that the api request is successful.
        The output parsing will take place later
        """
        self._check_rate_limit_based_on_request(request_dict, **kwargs)

        request_dict_copy = request_dict.copy()

        request_dict_copy['data'] = json.dumps(request_dict_copy['data'])

        # see more info on timeouts: https://requests.readthedocs.io/en/latest/user/advanced/#timeouts
        response = request_fun(timeout=(3.05, 20), **request_dict_copy)

        status_code = response.status_code

        if status_code == 200:

            assert hasattr(response, "json"), "Response object does not have a json attribute"

            job.status = JobStatus.success
            job.request_output = response.json()
            job.error_description = None

        elif status_code == 429:

            job.status = JobStatus.failed
            job.error_description = f"{self.name} - Rate Limit reached"
            logging_fun = get_logging_fun(**kwargs)
            logging_fun(f"{job.error_description}")

            raise RateLimitException(job.error_description)

        else:

            job.status = JobStatus.failed

            if hasattr(response, "text"):
                job.error_description = response.text

            console.log(f"{self.name} - unknown failure: {job.error_description}")
            raise UnknownAPIException(job.error_description)
        
        return job

    def _pre_request(self, prompt: str, **kwargs) -> RequestInput:
        
        if isinstance(self.request_input_classmethod, str):
            raise ValueError("Request input class method is not callable")

        output = self.request_input_classmethod(
            prompt=prompt,
            payload=self.payload,
            url=self.url
        )

        assert isinstance(output, RequestInput)

        return output

    def _make_request(self, request_input: RequestInput, job_id: str, use_cache: bool = True, **kwargs) -> Job:

        if isinstance(self.request_input_data_extraction, str):
                raise ValueError("Request input class method is not callable")

        # create request job object, which will later store the result of the request
        job = Job(
            job_id=job_id,
            rest_model_name=self.rest_api_model_name, 
            function=requests.post, 
            request_dict=request_input.data,
        )

        # check if cached
        if job.filepath.exists() and use_cache:

            job = Job.from_json_file(job.filepath)

            if job.is_success:
                return job

        try:

            job = self._backoff(job=job, request_fun=traced_request, request_dict=request_input.data, **kwargs)

        except RateLimitException as e:
            raise e

        except Exception as e:

            job.status = JobStatus.failed
            job.error_description = str(e)

            prompt = self.request_input_data_extraction(request_input.data)

            error = LLMError(
                prompt=prompt,
                response=job.error_description,
                error_message=str(job.error_description)
            )

            log_pydantic_error(
                filename=self._get_log_filename(prompt=prompt),
                error=error,
                **kwargs
            )

            raise e
        
        return job

    def _post_request(self, request_job: Job, **kwargs) -> RequestOutput:

        """
        This function ensures that the job output can be transformed into a request output object.
        """

        req_output = None

        if not request_job.is_success:
            raise Exception(request_job.error_description)
        
        if isinstance(self.request_input_data_extraction, str):
                raise ValueError("Request input class method is not callable")

        if isinstance(self.request_output_classmethod, str):
                raise ValueError("Request output class method is not callable")

        try:

            req_output = self.request_output_classmethod(request_job.request_output, is_prefilled=True)

            assert isinstance(req_output, RequestOutput)

            if self.rate_limit_manager is None:
                return req_output
        
            if not isinstance(self.rate_limit_manager, RateLimitManager):
                raise ValueError("Rate limit manager is not a RateLimitManager object")

            # if job has already been saved, it means that the request was successful and the output was cached
            # no need to update the rate limit
            if not request_job.filepath.exists():

                self.rate_limit_manager.update(
                    tokens=req_output.num_tokens,
                    tokens_only=True
                )

        except Exception as e:

            prompt = self.request_input_data_extraction(request_job.request_dict)

            error = LLMError(
                prompt=prompt,
                response=str(request_job.request_output),
                error_message=str(e)
            )

            log_pydantic_error(
                filename=self._get_log_filename(prompt=prompt),
                error=error,
                **kwargs
            )

            assert isinstance(req_output, RequestOutput)

        return req_output

    def _get_parsed_output(self, prompt: str, request_output: RequestOutput, pydantic_model: Type[BaseModel], **kwargs) -> LogProbScore:

        try:

            parser = PydanticOutputParser(pydantic_object=pydantic_model)

            parsed_response: Prediction = parser.parse(text=request_output.text)

            if not hasattr(parsed_response, "valid_labels"):
                raise Exception("Response does not have a valid_labels attribute")

            Prediction.valid_labels = parsed_response.valid_labels
            
            result = LogProbScore(
                answer=Prediction(**parsed_response.dict()),
                logprobs=request_output.logprobas
            )

            return result

        except Exception as e:

            error = LLMError(
                prompt=prompt,
                response=request_output.text,
                error_message=str(e)
            )
                
            log_pydantic_error(
                filename=self._get_log_filename(prompt=prompt),
                error=error,
                **kwargs
            )

            raise e

    def __call__(self, text: str, pydantic_model: Type[BaseModel], use_cache: bool = False, **kwargs) -> LogProbScore:
        
        cache = JsonCache(
            filepath=Path(self.rest_api_model_name) / f"{Hash.hash(text)}.json"
        )

        result = None

        # check if cache is enabled
        if use_cache:
            result = cache.read()
            
        if result is not None:

            assert isinstance(result, dict)

            result = LogProbScore(**result)

            return result

        # verify pydantic model
        key = "valid_labels"

        if not hasattr(pydantic_model, key):
            raise ValueError("Pydantic model must have valid_labels")
        
        labels = getattr(pydantic_model, key)

        assert isinstance(labels, list)
        assert all([isinstance(el, str) for el in labels])

        if 'pbar' in kwargs:
            pbar = kwargs['pbar']

            if hasattr(pbar, 'write'):
                pbar.write(f"Using {self.name}")


        # set id for request and cache
        hash_filename = self._get_hash_id(text)

        # get standardized input data
        request_input = self._pre_request(prompt=text, **kwargs)

        # carry out request
        request_job = self._make_request(job_id=hash_filename, request_input=request_input, use_cache=use_cache, **kwargs)

        # reformat output of request
        request_output = self._post_request(request_job=request_job, **kwargs)
        
        # parse output based on standardized request output
        logprobscore = self._get_parsed_output(
            request_output=request_output,
            pydantic_model=pydantic_model,
            prompt=text,
            **kwargs
        )

        if use_cache:

            # avoids calling the same llm with a query twice
            # and it is useful for debugging
            # even though this cache is not useful for the inference handler
            # each job gets formatted depending on the individual llm after the rest api call 
            request_job.save() 

            cache.write(
                obj=logprobscore.model_dump()
            ) 

        # delete potential error log file
        delete_error_log(
            filename=self._get_log_filename(prompt=text)
        )

        return logprobscore


if __name__ == '__main__':

    from src.ml.classifier.llm.util.request import RequestOutput
    from src.util.constants import LLMModels, RESTAPI_URLS

    prompt_template = """
    Please classify this sentence: {}

    You must comply will the following instructions:

    - There are two possible labels and you have to decide which one to use: "question" and "answer".
    - Fill in your label and your reasoning using this format:
        {{"reasoning": "<your reasoning>", "label": "<your label>"}}  
    - Reply only with valid json
    """

    Prediction.valid_labels = ["question", "answer"]

    model = StructuredRequestLLM(
        name="ollama-phi4-14b",
        rest_api_model_name="phi4-14b",
        url='http://localhost:11434/api/chat',
        payload={
            "model": "phi4:latest",
        },
        request_input_classmethod="create_ollama_request_input",
        request_output_classmethod="from_ollama_request",
        request_input_data_extraction="get_prompt_from_ollama_data",
        rate_limit_manager=None
    )

    prompt = prompt_template.format("What is the meaning of life?")
    result = model(prompt, use_cache=False, pydantic_model=Prediction)

    print(result.answer)