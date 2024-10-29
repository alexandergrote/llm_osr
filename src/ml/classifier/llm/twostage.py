import numpy as np
import pandas as pd

from copy import copy
from pydantic import BaseModel
from typing import Dict, List, Optional, Union, Tuple, Callable
from langchain.output_parsers import PydanticOutputParser
from pydantic import model_validator, ConfigDict, field_validator
from langchain_core.prompts import PromptTemplate

from src.util.dynamic_import import DynamicImport
from src.ml.classifier.llm.util.cosine_selector import CosineSelector
from src.ml.classifier.llm.util.prediction import PredictionV1, Prediction
from src.ml.classifier.llm.util.request import RequestOutput, RequestInputData
from src.ml.classifier.llm.util.logprob import LogProbScore, LogProb
from src.ml.classifier.llm.base import AbstractClassifierLLM
from src.ml.classifier.llm.util.rest import LLM as RestLLM
from src.util.constants import UnknownClassLabel, Directory, DatasetColumn
from src.ml.classifier.llm.util.rest import AbstractLLM
from src.ml.classifier.llm.util.logprob_extraction import LogProbExtractor
from src.util.logger import log_error
from src.util.hashing import Hash
from src.util.constants import ErrorValues


# prompts taken from retry output parser from langchain
NAIVE_COMPLETION_RETRY = """Prompt:
{prompt}
Completion:
{completion}

Above, the Completion did not satisfy the constraints given in the prompt.
The error message is: 
{error_message}

Please try again:"""

NAIVE_RETRY_PROMPT = PromptTemplate.from_template(NAIVE_COMPLETION_RETRY)

# two types of uncertainty prediction
# 1) semantic entropy - common baseline, 
# 2) epistemic uncertainty - state of the art, k = 5, https://arxiv.org/pdf/2402.10189
# 3) reflection with bs detector https://www.nature.com/articles/s41586-024-07421-0


class LLMClassifierFactory(BaseModel):

    """
    This class is used to create a LLM classifier.
    Its secondary purpose is to provide type checking for the model_config used in any yaml file
    """

    name: str
    request_output_formatter: Union[Callable, dict]
    request_input: Union[RequestInputData, dict]

    function_key: str = 'function'
    function_key_params: str = 'params'

    @model_validator(mode='after')
    def _init_output_formatter(self):

        key = 'request_output_formatter'

        if not isinstance(self.request_output_formatter, dict):
            raise ValueError(f"{key} must be a dict")
        
        self.request_output_formatter = DynamicImport.get_attribute_from_class(
            self.request_output_formatter[self.function_key]
        )

        key = 'request_input'

        if not isinstance(self.request_input, dict):
            raise ValueError(f"{key} must be a dict")

        function_input = DynamicImport.get_attribute_from_class(
            self.request_input[self.function_key]
        )

        self.request_input = function_input(**self.request_input[self.function_key_params])

        return self

    def create(self) -> AbstractClassifierLLM:

        llm = RestLLM(
            name=self.name,
            request_output_formatter=self.request_output_formatter,
            request_input=self.request_input
        )
        
        return llm


class TwoStageLLM(AbstractClassifierLLM):

    unknown_detection_model: Union[AbstractLLM, dict]
    unknown_detection_prompt: str = "ood.txt"

    classifier_model: Union[AbstractLLM, dict]
    classifier_prompt: str = "multiclass.txt"

    # skip unknown detection
    skip_unknown_detection: bool = False

    # fewshot selection of data points
    selector: Optional[Union[dict, CosineSelector]] = None
    n_classes: Optional[int] = 3
    n_datapoints_per_class: Optional[int] = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='before')
    def _init_models(data: dict):

        model_keys = ["unknown_detection_model", "classifier_model"]

        for model in model_keys:
            data[model] = LLMClassifierFactory(**data[model]).create()
        
        data_selector = data.get('selector')

        if isinstance(data_selector, dict):
            data['selector'] = DynamicImport.init_class_from_dict(data_selector)

        return data
    
    @model_validator(mode='after')
    def _init_model_prompts(self):

        for prompt_attr_name in ["unknown_detection_prompt", "classifier_prompt"]:

            prompt_filename = getattr(self, prompt_attr_name)

            with open(Directory.PROMPT_DIR / prompt_filename, 'r') as f:
                setattr(self, prompt_attr_name, f.read())

        return self
    
    
    def _get_parsed_output(self, model: AbstractLLM, parser: PydanticOutputParser, prompt: str, retries: int = 5) -> Optional[LogProbScore]:

        # work on copy
        prompt_copy = copy(prompt)

        result = None
        completion = None

        for i in range(retries):

            try:

                completion: RequestOutput = model(prompt=prompt_copy, parser=parser)

                if completion.error:
                    raise ValueError(completion.text)
                
                parsed_data = parser.parse(text=completion.text)

                Prediction.valid_labels = parsed_data.valid_labels
                
                result = LogProbScore(
                    answer=Prediction(**parsed_data.dict()),
                    logprobs=completion.logprobas
                )

            except Exception as e:

                #prompt_copy = NAIVE_RETRY_PROMPT.format(
                #    prompt=prompt,
                #)

                log_error(
                    filename=Hash.hash(prompt) + ".json",
                    json_dict={
                        "prompt": prompt,
                        "response": completion.text,
                        "error_message": str(e)
                    }
                )

            if result is not None:
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
               

    def _detect_unknown_class(self, text: str) -> Optional[LogProbScore]:

        # todo: the output of a function must either be a string with the error message or a pydantic object

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
        instructions = parser.get_format_instructions()

        examples = self._get_examples(
            text_to_classify=text,
        )

        examples_msg = '\n'.join([f"{example['input']} -> {example['output']}" for example in examples])

        prompt = self.unknown_detection_prompt.format(
            examples_msg=examples_msg,
            text=text,
            instructions=instructions
        )
    
        logprob_score = self._get_parsed_output(model=self.unknown_detection_model, parser=parser, prompt=prompt, retries=5)
        
        return logprob_score
    
    def _get_unknown_class_logprob_value(self, logprobs: List[LogProb], last_n: Optional[int] = None, fallback: bool = False) -> LogProb:

        try:

            logprob_candidates = logprobs

            if last_n is not None:
                logprob_candidates = logprob_candidates[-last_n:]

            assert len(logprob_candidates) > 0  

            logprob_true_values = LogProbExtractor.get_specific_logprobas(text='true', log_sequences=logprob_candidates)       
            logprob_false_values = LogProbExtractor.get_specific_logprobas(text='false', log_sequences=logprob_candidates)

            if all([len(logprob_false_values) > 0, len(logprob_true_values) > 0]):
                raise Exception("Ambiguous Logprobs")

            if len(logprob_true_values) == 1:
                return logprob_true_values[0]

            if len(logprob_false_values) == 1:
                return logprob_false_values[0]

            if fallback:

                for logprob in logprobs[::-1]:
                    if 'true' in logprob.text.lower():
                        return logprob
                        
                    if 'false' in logprob.text.lower():
                        return logprob
                
            raise Exception("No Logprob found")

        except Exception as e:

            log_probs = [logprob.text for logprob in logprobs]

            text = ''.join(log_probs)

            log_error(
                filename=Hash.hash(text) + '.json',
                json_dict={
                    "error": str(e),
                    "logprobs": log_probs,
                }
            )

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
        instructions = parser.get_format_instructions()

        classes_msg = '\n'.join(valid_labels)

        examples = self._get_examples(
            text_to_classify=text,
        )

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
        The label must be one of those values: {classes_msg}
        
        {instructions}

        """

        logprob_score = self._get_parsed_output(model=self.classifier_model, parser=parser, prompt=prompt, retries=5)

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
        
        unknown_score = 0.0
        
        if not self.skip_unknown_detection:
            
            unknown_prediction = self._detect_unknown_class(text=text)

            if unknown_prediction is None:
                return ErrorValues.PARSING_STR.value, float(ErrorValues.PARSING_NUM.value)

            logprob_value = self._get_unknown_class_logprob_value(
                logprobs=unknown_prediction.logprobs,
                last_n=10,
                fallback=True
            )
            
            if logprob_value is None:
                return ErrorValues.PARSING_STR.value, float(ErrorValues.PARSING_NUM.value)
            
            unknown_class: str = logprob_value.text.lower()
            unknown_score: float = LogProbScore.calculate_linear_prob(logprob_value.logprob)

            if "true" in unknown_class:            
                return UnknownClassLabel.UNKNOWN_STR.value, unknown_score
        
        known_prediction = self._classify_known_classes(text=text)

        if known_prediction is None:
            return ErrorValues.PARSING_STR.value, float(ErrorValues.PARSING_NUM.value)

        # recalculate the unknown score
        # if model was confidenct in their prediction and an in distribution example was detected, the anomaly score needs to be low
        unknown_score = 1 - unknown_score
        
        return known_prediction.answer.label, unknown_score
        


if __name__ == '__main__':

    import numpy as np
    from typing import cast
    from src.util.load_hydra import get_hydra_config

    key = "ml__classifier"

    config = get_hydra_config(
        overrides=[
            f"{key}=two_stage_llm_llama"
        ]
    )

    llm = DynamicImport.init_class_from_dict(
        dictionary=config[key],
    )

    # Explicitly cast llm to TwoStageLLM
    llm = cast(TwoStageLLM, llm)

    
    data_train = pd.DataFrame({
        DatasetColumn.FEATURES: ["Ich heiße Alex", "Auf Wiedersehen!", "Hallo", "Wo kann ich Kekse kaufen?", "Die Apfelsaftschorle finde ich wo?", "Das Essen ist sehr lecker", "München ist eine große Stadt."],
        DatasetColumn.LABEL: ['Greeting', 'Farewell', "Greeting", "Food", "Food", "Food", "City"]
    })

    data_valid = pd.DataFrame({
        DatasetColumn.FEATURES: ["Ich bin ein Mensch", "Ich war noch nie in Berlin"],
        DatasetColumn.LABEL: ['Mensch', 'City']
    })

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
