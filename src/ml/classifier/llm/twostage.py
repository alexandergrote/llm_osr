import numpy as np
import pandas as pd

from copy import copy
from typing import Dict, List, Optional, Union, Tuple
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import model_validator, ConfigDict
from langchain_core.prompts import PromptTemplate

from src.util.dynamic_import import DynamicImport
from src.ml.classifier.llm.util.cosine_selector import CosineSelector
from src.ml.classifier.llm.util.prediction import PredictionV1, Prediction
from src.ml.classifier.llm.util.request import RequestOutput
from src.ml.classifier.llm.util.logprob import LogProbScore, LogProb
from src.ml.classifier.llm.base import AbstractClassifierLLM
from src.util.constants import DatasetColumn, LLMModels
from src.ml.classifier.llm.util.mapping import LLM_Mapping
from src.util.constants import UnknownClassLabel
from src.ml.classifier.llm.util.rest import AbstractLLM
from src.ml.classifier.llm.util.logprob_extraction import LogProbExtractor


# prompts taken from retry output parser from langchain
NAIVE_COMPLETION_RETRY = """Prompt:
{prompt}
Completion:
{completion}

Above, the Completion did not satisfy the constraints given in the prompt.
Please try again:"""

NAIVE_RETRY_PROMPT = PromptTemplate.from_template(NAIVE_COMPLETION_RETRY)

# two types of uncertainty prediction
# 1) semantic entropy - common baseline, 
# 2) epistemic uncertainty - state of the art, k = 5, https://arxiv.org/pdf/2402.10189
# 3) reflection with bs detector https://www.nature.com/articles/s41586-024-07421-0


class TwoStageLLM(AbstractClassifierLLM):

    model: AbstractLLM
    clf_str: str

    selector: Optional[Union[dict, CosineSelector]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='before')
    def _init_model(data: dict):

        data['model'] = LLM_Mapping[LLMModels(data['clf_str'])]

        data_selector = data.get('selector')

        if isinstance(data_selector, dict):
            data['selector'] = DynamicImport.init_class_from_dict(data_selector)

        return data
    
    
    def _get_parsed_output(self, model: AbstractLLM, parser: PydanticOutputParser, prompt: str, retries: int = 5) -> Optional[LogProbScore]:

        # work on copy
        prompt_copy = copy(prompt)

        result = None

        for _ in range(retries):

            try:
                completion = model(prompt=prompt_copy)

                if not isinstance(completion, RequestOutput):
                    raise ValueError("Model did not return a RequestOutput object")
                
                parsed_data = parser.parse(text =completion.text)

            except OutputParserException:

                prompt_copy = NAIVE_RETRY_PROMPT.format(prompt=prompt_copy, completion=completion.text)

            try:

                Prediction.valid_labels = parsed_data.valid_labels
                
                result = LogProbScore(
                    answer=Prediction(**parsed_data.dict()),
                    logprobs=completion.logprobas
                )

            except Exception as e:
                print(e)
                continue

            if result:
                return result
            
        return None


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


    def _get_examples(self, query: str = 'input', answer: str = 'output') -> List[Dict[str, str]]:

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

        rng = np.random.default_rng(42)

        for selected_class in self.classes:

            y_mask = self.y_train == selected_class

            x_sub = self.x_train[y_mask]

            x_chosen = rng.choice(x_sub)

            examples.append({query: x_chosen, answer: selected_class})

        return examples
    
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        **kwargs
    ):

       self.y_train = y_train
       self.y_valid = y_valid

       self.x_train = x_train
       self.x_valid = x_valid

       self.classes = np.unique(self.y_train)

       if self.selector is not None:
           pass
               

    def _detect_unknown_class(self, text: str) -> LogProbScore:

        if self.y_train is None:
            raise ValueError("Not fitted")
        
        if self.x_train is None:
            raise ValueError("Not fitted")
        
        if self.classes is None:
            raise ValueError("Not fitted")
        
        # Example usage
        valid_labels = ["true", "false"]
        PredictionV1.valid_labels = valid_labels
        
        parser = PydanticOutputParser(pydantic_object=PredictionV1)

        examples = self._get_examples()

        examples_msg = '\n'.join([f"{example['input']} -> {example['output']}" for example in examples])


        prompt = f"""\n
        You are an Out-Of-Distribution detector for an open set recognition problem. 
        Your task is to decide if "{text}" is similar in meaning to these examples and their corresponding classes:

        {examples_msg} 
        
        First, think step by step about why the abstract concept behind the incoming data point might be similar to the concepts of the examples.
        Second, think step by step about why the the abstract concept the incoming data point might differ significantly from the examples.
        Third, compare the two scenarios and answer with the more probable answer.
        
        Provide the reasoning containing your thought process first and then the label.
        The label must be 'true' if the overall concept of the incoming data point is significantly different to the examples and 'false' if the concept is similar.
 
        For your answer, you must use this json format: 
        {{
           "reasoning": <your reasoning>,
           "label": <your final label>
        }}.
        
        """
            
        logprob_score = self._get_parsed_output(model=self.model, parser=parser, prompt=prompt, retries=3)

        if logprob_score is None:
            raise Exception("logprob score should not be None")
        
        return logprob_score
    

    def _classify_known_classes(self, text: str) -> LogProbScore:

        if self.y_train is None:
            raise ValueError("Not fitted")
        
        if self.x_train is None:
            raise ValueError("Not fitted")
        
        if self.classes is None:
            raise ValueError("Not fitted")
        
        # Example usage
        valid_labels = list(self.classes)
        PredictionV1.valid_labels = valid_labels
        
        parser = PydanticOutputParser(pydantic_object=PredictionV1)

        classes_msg = '\n'.join(valid_labels)

        examples = self._get_examples()

        examples_msg = '\n'.join([f"{example['input']} -> {example['output']}" for example in examples])

        prompt = f"""\n
        You are a classifier that classifies an incoming data point into predefined classes. 
        Your task is to decide if "{text}" belongs to these classes: \n{classes_msg}\n

        Here's a list of examples associated with these classes:\n 
        
        {examples_msg}

        First, think step by step of the overall concept behind the incoming data point.
        Second, think step by step about the concepts of each class.
        Third, compare the overall concept of the incoming data point with the concepts of each class and answer with the more probable answer.
        
        Provide the reasoning containing your thought process first and then the label.
        The label must be one of those values:
        {classes_msg}
 
        For your answer, you must use this json format: 
        {{
           "reasoning": <your reasoning>,
           "label": <your final label>
        }}.
        """

        logprob_score = self._get_parsed_output(model=self.model, parser=parser, prompt=prompt, retries=3)

        if logprob_score is None:
            raise Exception("logprob score should not be None")
        
        return logprob_score


    def _single_predict(self, text: str) -> Tuple[str, float]:

        if self.y_train is None:
            raise ValueError("Not fitted")
        
        if self.x_train is None:
            raise ValueError("Not fitted")
        
        if self.classes is None:
            raise ValueError("Not fitted")

        unknown_prediction = self._detect_unknown_class(text=text)

        logprob_candidates = LogProbExtractor.get_target_logprobas(
            log_sequences=unknown_prediction.logprobs,
            prior_sequence=["label", '":'],
            end_sequence=[]
        )

        assert len(logprob_candidates) > 0

        logprob_value = None

        logprob_true_values = LogProbExtractor.get_specific_logprobas(text='true', log_sequences=logprob_candidates)       
        logprob_false_values = LogProbExtractor.get_specific_logprobas(text='false', log_sequences=logprob_candidates)

        if all([len(logprob_false_values) > 0, len(logprob_true_values) > 0]):
            raise Exception("Ambiguous Logprobs")

        if len(logprob_true_values) == 1:
            logprob_value = logprob_true_values[0]

        if len(logprob_false_values) == 1:
            logprob_value = logprob_false_values[0]

        if logprob_value is None:
            raise Exception("No logprob value found")
        
        assert isinstance(logprob_value, LogProb)

        unknown_class: str = logprob_value.text.lower()
        unknown_score: float = LogProbScore.calculate_linear_prob(logprob_value.logprob)

        if "true" in unknown_class:            
            return UnknownClassLabel.UNKNOWN_STR.value, unknown_score
        
        known_prediction = self._classify_known_classes(text=text)

        # recalculate the unknown score
        # if model was confidenct in their prediction and an in distribution example was detected, the anomaly score needs to be low
        unknown_score = 1 - unknown_score
        
        return known_prediction.answer.label, unknown_score
        


if __name__ == '__main__':

    import numpy as np

    data_train = pd.DataFrame({
        DatasetColumn.FEATURES: ["Ich heiße Alex", "Auf Wiedersehen!", "Hallo"],
        DatasetColumn.LABEL: ['Greeting', 'Farewell', "Greeting"]
    })

    data_valid = pd.DataFrame({
        DatasetColumn.FEATURES: ["Ich bin ein Mensch", "Ich war noch nie in Berlin"],
        DatasetColumn.LABEL: ['Mensch', 'Berlin']
    })

    llm = TwoStageLLM(
        clf_str=LLMModels.LLAMA_3_70B_Remote_HF.value
    )

    llm.fit(
        x_train=data_train[DatasetColumn.FEATURES].values,
        y_train=data_train[DatasetColumn.LABEL].values,
        x_valid=data_valid[DatasetColumn.FEATURES].values,
        y_valid=data_valid[DatasetColumn.LABEL].values,
    )

    result = llm._single_predict(text="Hello")

    print(result)

    result2 = llm.predict(
        x=np.array([["Hola"], ["Hallo du"], ["Ich möchte einen Tee bestellen"], ['Ich wohne in Bayern'], ["I like trains"]], dtype=np.object_),
        include_outlierscore=True
    )

    print(result2)

    

    
    

