import numpy as np

from copy import copy
from abc import abstractmethod
from tqdm import tqdm
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, model_validator
from pydantic.config import ConfigDict
from numpy import ndarray
from typing import Optional, Union, Tuple
from langchain_core.prompts import PromptTemplate

from src.util.logger import console
from src.util.constants import ErrorValues
from src.ml.classifier.base import BaseClassifier
from src.ml.classifier.llm.util.prediction import PredictionV1, Prediction
from src.ml.classifier.llm.util.request import RequestOutput
from src.ml.classifier.llm.util.logprob import LogProbScore
from src.ml.classifier.llm.util.mapping import LLM_Mapping
from src.util.constants import UnknownClassLabel, LLMModels
from src.ml.classifier.llm.util.rest import AbstractLLM


# prompts taken from retry output parser from langchain
NAIVE_COMPLETION_RETRY = """Prompt:
{prompt}
Completion:
{completion}

Above, the Completion did not satisfy the constraints given in the Prompt.
Please try again:"""

NAIVE_RETRY_PROMPT = PromptTemplate.from_template(NAIVE_COMPLETION_RETRY)



class AbstractClassifierLLM(BaseModel, BaseClassifier):

    model: AbstractLLM
    clf_str: str

    # placeholders, values will be set later
    x_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    classes: Optional[np.ndarray] = None
    parser: Optional[PydanticOutputParser] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='before')
    def _init_model(data: dict):

        data['model'] = LLM_Mapping[LLMModels(data['clf_str'])]

        return data
    
    def _get_parsed_output(self, model: AbstractLLM, parser: PydanticOutputParser, prompt: str, retries: int = 3) -> Optional[LogProbScore]:

        # work on copy
        prompt_copy = copy(prompt)

        result = None

        for _ in range(retries):

            try:
                completion = model(prompt=prompt_copy)

                if not isinstance(completion, RequestOutput):
                    raise ValueError("Model did not return a RequestOutput object")
                
                parsed_data = parser.parse(text = completion.text)

                Prediction.valid_labels = parsed_data.valid_labels
                
                result = LogProbScore(
                    answer=Prediction(**parsed_data.dict()),
                    logprobs=completion.logprobas
                )


            except OutputParserException:

                prompt_copy = NAIVE_RETRY_PROMPT.format(prompt=prompt_copy, completion=completion.text)

            if result:
                return result
            
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
       PredictionV1.valid_labels = valid_labels
       
       self.parser = PydanticOutputParser(pydantic_object=PredictionV1)

    @abstractmethod
    def _single_predict(self, text: str) -> Optional[LogProbScore]:
        raise NotImplementedError("Method must be implemented in subclass")
    
    def predict(self, x: np.ndarray, include_outlierscore: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        # assert if array is 2D
        if len(x.shape) != 2:
            raise ValueError("Input should be 2D array")

        result_text_list= []
        result_score_list = []    

        for el in tqdm(x, desc="LLM Prediction"):

            el_str = el[0]

            try:

                result = self._single_predict(text=el_str)

                result_text = ErrorValues.PARSING_STR.value
                result_score = 0.0

                if result is not None:

                    result_text = result.answer.label
                    result_score = result.unknown_score

                result_text_list.append(result_text)
                result_score_list.append(result_score)

            except Exception as e:

                console.log(f"Error processing text: {el_str}")

                raise e
            
        if include_outlierscore:
            return np.array(result_text_list), np.array(result_score_list)
        
        return np.array(result_text_list)
    
    def predict_proba(self, x: ndarray, **kwargs) -> ndarray:
        pass