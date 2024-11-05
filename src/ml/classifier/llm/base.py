import numpy as np
import pandas as pd

from abc import abstractmethod
from tqdm import tqdm
from copy import copy
from pydantic import BaseModel
from pydantic.config import ConfigDict
from numpy import ndarray
from typing import Optional, Union, Tuple, List, Dict
from langchain.output_parsers import PydanticOutputParser

from src.ml.classifier.llm.util.request import RequestOutput
from src.ml.classifier.llm.util.prediction import Prediction, PredictionV1
from src.ml.util.job_queue import Job
from src.util.hashing import Hash
from src.ml.classifier.llm.util.rest import AbstractLLM
from src.ml.classifier.llm.util.logprob import LogProbScore
from src.util.constants import ErrorValues
from src.ml.classifier.base import BaseClassifier
from src.util.logger import log_pydantic_error, delete_error_log
from src.util.error import LLMError
from src.ml.classifier.llm.util.cosine_selector import CosineSelector
from src.util.constants import DatasetColumn

class AbstractClassifierLLM(BaseModel, BaseClassifier):

    # placeholders, values will be set later
    x_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    x_valid: Optional[np.ndarray] = None
    y_valid: Optional[np.ndarray] = None
    classes: Optional[np.ndarray] = None

    # fewshot selection of data points
    selector: Optional[Union[dict, CosineSelector]] = None
    n_classes: Optional[int] = 3
    n_datapoints_per_class: Optional[int] = 5


    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, **kwargs):
        raise NotImplementedError("Method must be implemented in subclass")

    @abstractmethod
    def _single_predict(self, text: str) -> Tuple[str, float]:
        raise NotImplementedError("Method must be implemented in subclass")
    
    def _get_ood_examples(self, query: str = 'input', answer: str = 'output') -> List[Dict[str, str]]:

        if self.y_train is None:
            raise ValueError("Model is not fitted")
        
        if self.y_valid is None:
            raise ValueError("Model is not fitted")
        
        if self.x_train is None:
            raise ValueError("Model is not fitted")
        
        if self.x_valid is None:
            raise ValueError("Model is not fitted")
        
        if self.classes is None:
            raise ValueError("Model is not fitted")
        
        # draw a random example from each known class
        examples = []

        rng = np.random.default_rng(42)

        for unique_class in self.classes:
             
            y_mask = self.y_train == unique_class

            x_sub = self.x_train[y_mask]

            x_chosen = rng.choice(x_sub)

            examples.append({query: x_chosen, answer: "false"})

        # draw example of unknown class
        unkwown_classes = np.setdiff1d(self.y_valid, self.y_train)

        for unkwown_class in unkwown_classes:
            
            y_mask = self.y_valid == unkwown_class
            x_sub = self.x_valid[y_mask]
            x_chosen = rng.choice(x_sub)

            examples.append({query: x_chosen, answer: "true"})

        return examples


    def _get_examples(self, text_to_classify: str, query: str = 'input', answer: str = 'output') -> List[Dict[str, str]]:

        if self.y_train is None:
            raise ValueError("Model is not fitted")
        
        if self.y_valid is None:
            raise ValueError("Model is not fitted")
        
        if self.x_train is None:
            raise ValueError("Model is not fitted")
        
        if self.x_valid is None:
            raise ValueError("Model is not fitted")
        
        if self.classes is None:
            raise ValueError("Model is not fitted")

        # draw a random example from each class
        examples = []

        if self.selector is None:

            rng = np.random.default_rng(42)

            for selected_class in self.classes:

                y_mask = self.y_train == selected_class

                x_sub = self.x_train[y_mask]

                x_chosen = rng.choice(x_sub)

                examples.append({query: x_chosen, answer: selected_class})

        else:

            data = pd.DataFrame({
                DatasetColumn.LABEL: self.y_train,
                DatasetColumn.TEXT: self.x_train.reshape(-1,)
            })

            n_classes = len(self.classes) if self.n_classes is None else self.n_classes
            n_datapoints_per_class = 1 if self.n_datapoints_per_class is None else self.n_datapoints_per_class

            if not isinstance(self.selector, CosineSelector):
                raise ValueError("Selector must be of type CosineSelector")

            dataframe = self.selector.get_most_similar_datapoints_for_n_classes(
                query=text_to_classify,
                data=data,
                n_classes=n_classes,
                n=n_datapoints_per_class
            )

            for _, row in dataframe.iterrows():
                examples.append({
                    query: row[DatasetColumn.TEXT], 
                    answer: row[DatasetColumn.LABEL]
                })


        return examples

    def predict(self, x: np.ndarray, include_outlierscore: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        # assert if array is 2D
        if len(x.shape) != 2:
            raise ValueError("Input should be 2D array")

        result_text_list= []
        result_score_list = []    

        for el in tqdm(x, desc="LLM Prediction"):

            el_str = el[0]

            result_text = ErrorValues.PARSING_STR.value
            result_score = float(ErrorValues.PARSING_NUM.value)

            result = None

            try:

                result = self._single_predict(text=el_str)

            except Exception as e:

                tqdm.write(f"Error processing text: {el_str}")
                tqdm.write(f"Error: {e}")

                pass

            if result is not None:
                result_text, result_score = result

            result_text_list.append(result_text)
            result_score_list.append(result_score)

            
        if include_outlierscore:
            return np.array(result_text_list), np.array(result_score_list)
        
        return np.array(result_text_list)
    
    def predict_proba(self, x: ndarray, **kwargs) -> ndarray:
        pass


class LLMClassifierMixin:

    def _get_log_filename(self, prompt: str) -> str:
        return Hash.hash(prompt) + ".json"

    def _get_parsed_output(self, model: AbstractLLM, parser: PydanticOutputParser, prompt: str, retries: int = 5) -> Optional[LogProbScore]:

        # work on copy
        prompt_copy = copy(prompt)
        
        for _ in range(retries):

            log_filename = self._get_log_filename(prompt=prompt_copy) 

            try:

                completion: Job = model(prompt=prompt_copy)

                if not completion.is_success:
                    raise ValueError(completion.error_description)
            
            except Exception as e:

                error = LLMError(
                    prompt=prompt,
                    response=completion.error_description,
                    error_message=str(e)
                )

                log_pydantic_error(
                    filename=log_filename,
                    error=error
                )

            # parse response
            # first parsing: request output will be parsed to a pydantic object
            # second parsing: output needs to follow the parser object
            # the first step is needed to manually exclude parts of the output that lead to errors with the pydantic output parser

            try:

                first_parsed_response: RequestOutput = model.format_output(job=completion)
            
            except Exception as e:

                error = LLMError(
                    prompt=prompt,
                    response=str(completion.request_output),
                    error_message=str(e)
                )

                log_pydantic_error(
                    filename=log_filename,
                    error=error
                )

                continue

            if first_parsed_response is None:
                raise ValueError("No parsed response")

            try:

                second_parsed_response: PredictionV1 = parser.parse(text=first_parsed_response.text)

                Prediction.valid_labels = second_parsed_response.valid_labels
                
                result = LogProbScore(
                    answer=Prediction(**second_parsed_response.dict()),
                    logprobs=first_parsed_response.logprobas
                )

                completion.save()

                # delete potential error log file
                delete_error_log(
                    filename=self._get_log_filename(prompt=prompt_copy)
                )

                return result

            except Exception as e:

                error = LLMError(
                    prompt=prompt,
                    response=first_parsed_response.text,
                    error_message=str(e)
                )
                
                log_pydantic_error(
                    filename=log_filename,
                    error=error
                )

        return None