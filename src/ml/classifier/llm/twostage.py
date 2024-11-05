import numpy as np
import pandas as pd

from typing import List, Optional, Union, Tuple
from langchain.output_parsers import PydanticOutputParser
from pydantic import model_validator, ConfigDict
from langchain_core.prompts import PromptTemplate

from src.util.dynamic_import import DynamicImport
from src.ml.classifier.llm.util.prediction import PredictionV1
from src.ml.classifier.llm.util.logprob import LogProbScore, LogProb
from src.ml.classifier.llm.base import AbstractClassifierLLM
from src.util.constants import UnknownClassLabel, Directory, DatasetColumn
from src.ml.classifier.llm.util.rest import AbstractLLM
from src.ml.classifier.llm.util.logprob_extraction import LogProbExtractor
from src.util.logger import log_pydantic_error
from src.util.error import LogProbError
from src.util.constants import ErrorValues
from src.ml.classifier.llm.base import LLMClassifierMixin
from src.ml.classifier.llm.factory import LLMClassifierFactory

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


class TwoStageLLM(LLMClassifierMixin, AbstractClassifierLLM):

    unknown_detection_model: Union[AbstractLLM, dict]
    unknown_detection_prompt: str = "ood.txt"

    classifier_model: Union[AbstractLLM, dict]
    classifier_prompt: str = "multiclass.txt"

    # skip unknown detection
    skip_unknown_detection: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='before')
    def _init_models(data: dict):

        model_keys = ["unknown_detection_model", "classifier_model"]

        for model in model_keys:
            data[model] = LLMClassifierFactory(**data[model]).create()
            assert hasattr(data[model], "format_output"), f"Model {model} must have a predict method"

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


    def _detect_unknown_class(self, text: str) -> Optional[LogProbScore]:

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

        if not isinstance(self.unknown_detection_model, AbstractLLM):
            raise ValueError("Unknown detection model must be of type AbstractLLM")
    
        logprob_score = self._get_parsed_output(model=self.unknown_detection_model, parser=parser, prompt=prompt, retries=5)
        
        return logprob_score
    
    def _get_unknown_class_logprob_value(self, logprobs: List[LogProb], last_n: Optional[int] = None, fallback: bool = False) -> Optional[LogProb]:

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

            error = LogProbError(
                error_message=str(e),
                logprobs=log_probs,
            )

            log_pydantic_error(
                filename=self._get_log_filename(prompt=text),
                error=error
            )

            return None

    def _classify_known_classes(self, text: str) -> Optional[LogProbScore]:

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

        examples = self._get_examples(
            text_to_classify=text,
        )

        classes_msg = '\n'.join(set(example["output"] for example in examples))
        examples_msg = '\n'.join([f"{example['input']} -> {example['output']}" for example in examples])

        prompt = self.classifier_prompt.format(
            examples_msg=examples_msg,
            text=text,
            instructions=instructions,
            classes_msg=classes_msg
        )

        if not isinstance(self.classifier_model, AbstractLLM):
            raise ValueError("Classifier model must be an AbstractLLM")

        logprob_score = self._get_parsed_output(model=self.classifier_model, parser=parser, prompt=prompt, retries=5)

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
            unknown_score = LogProbScore.calculate_linear_prob(logprob_value.logprob)

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
