import numpy as np
import os

from copy import copy
from abc import abstractmethod
from tqdm import tqdm
from pathlib import Path
from langchain.output_parsers import PydanticOutputParser, RetryOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, model_validator
from numpy import ndarray
from typing import Any, Optional, List, Union, Tuple
from langchain_core.prompts import PromptTemplate
import concurrent.futures

from src.ml.classifier.base import BaseClassifier
from src.ml.classifier.llm.cls.util.prediction import Prediction
from src.ml.classifier.llm.util.mapping import LLM_Mapping
from src.util.constants import UnknownClassLabel, LLMModels
from src.util.caching import PickleCacheHandler
from src.util.logging import console



# prompts taken from retry output parser from langchain
NAIVE_COMPLETION_RETRY = """Prompt:
{prompt}
Completion:
{completion}

Above, the Completion did not satisfy the constraints given in the Prompt.
Please try again:"""

NAIVE_RETRY_PROMPT = PromptTemplate.from_template(NAIVE_COMPLETION_RETRY)



class BaseLLM(BaseModel, BaseClassifier):

    model: Any
    model_str: str

    # placeholders, values will be set later
    x_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    classes: Optional[np.ndarray] = None
    parser: Optional[Any] = None

    last_prompt: Optional[str] = None
    logprobs: Optional[dict] = None
    
    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='before')
    def _init_model(data: dict):

        data['model'] = LLM_Mapping[LLMModels(data['model_str'])]

        return data
    
    def _retry(self, model: Any, parser: RetryOutputParser, prompt: str, retries: int = 3) -> Optional[Prediction]:

        # work on copy
        prompt_copy = copy(prompt)

        for _ in range(retries):

            parsed_data = None

            try:
                completion = model._call(prompt=prompt_copy)
                parsed_data = parser.parser.parse(completion)

            except OutputParserException:

                prompt_copy = NAIVE_RETRY_PROMPT.format(prompt=prompt_copy, completion=completion)

            if parsed_data:
                return parsed_data
            
        return None

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        **kwargs
    ):

       self.y_train = np.concatenate([
           y_train, y_valid
       ]) 

       self.x_train = np.concatenate([
           x_train, x_valid
       ])

       self.classes = np.unique(self.y_train)

       # Example usage
       valid_labels = [UnknownClassLabel.UNKNOWN_STR.value] + list(self.classes)
       Prediction.valid_labels = valid_labels
       
       self.parser = PydanticOutputParser(pydantic_object=Prediction)

    @abstractmethod
    def _single_predict(self, text: str, output_reasoning: bool = False) -> Union[str, Tuple[str, str]]:
        raise NotImplementedError("Method must be implemented in subclass")
    
    def _batch_single_predict(self, texts: List[str], n_jobs: int) -> List[str]:

        results = []
        
        # Use ThreadPoolExecutor to send requests in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            # Create a list of futures
            futures = {executor.submit(self._single_predict, prompt): prompt for prompt in texts}

            # Process the results as they complete
            for future in concurrent.futures.as_completed(futures):
                prompt = futures[future]
                try:
                    response = future.result()

                    if isinstance(response, tuple):
                        response = response[0]

                    results.append((prompt, response))
                except Exception as exc:
                    print(f'{prompt} generated an exception: {exc}')

        results_formatted = [el[1] for el in results]

        return results_formatted
    
    def predict_batch(self, x: List[str], **kwargs) -> np.ndarray:

        # divide this list into k chunks
        cpu_count = os.cpu_count()

        if cpu_count is None:
            k = 1
        else:
            k = 2 * cpu_count + 1

        x_batch = [x[i:i + k] for i in range(0, len(x), k)]

        results = []

        for el in x_batch:
            results += self._batch_single_predict(texts=el, n_jobs=k)

        return np.array(results)


    def predict(self, x: np.ndarray, include_outlierscore: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        if include_outlierscore:
            raise ValueError("Outlier score not implemented yet")

        result_list = []

        cache_handler = PickleCacheHandler(
            filepath= Path(self.__class__.__name__) / f"{self.model_str}.pkl"
        )

        cache = cache_handler.read()

        if cache is None:
            cache = {}   

        assert isinstance(cache, dict), "cache must be a dictionary"     

        for el in tqdm(x, desc="LLM Prediction"):

            el_str = el[0]
            
            result = cache.get(el_str)

            if result is not None:
                result_list.append(result)
                continue

            try:

                result = self._single_predict(text=el_str)

            except Exception as e:

                console.log(f"Error processing text: {el_str}")

                # in case of an error write cache to not lose progress
                cache_handler.write(cache)

                raise e

        
            cache[el_str] = result

            result_list.append(result)

        cache_handler.write(cache)

        
        return np.array(result_list)
    
    def predict_proba(self, x: ndarray, **kwargs) -> ndarray:
        pass