import numpy as np
import pandas as pd

from typing import Optional, Union, Tuple, List, Dict
from langchain.output_parsers import PydanticOutputParser
from pydantic import model_validator, ConfigDict
from omegaconf.dictconfig import DictConfig

from src.util.dynamic_import import DynamicImport
from src.ml.classifier.llm.util.prediction import PredictionV1
from src.ml.classifier.llm.util.logprob import LogProbScore
from src.ml.classifier.llm.base import AbstractClassifierLLM
from src.util.constants import UnknownClassLabel, Directory, DatasetColumn
from src.ml.classifier.llm.util.rest import AbstractLLM, StructuredRequestLLM
from src.ml.classifier.llm.util.rest_inference import InferenceHandler
from src.util.constants import ErrorValues
from src.ml.classifier.llm.base import LLMClassifierMixin


class TwoStageLLM(LLMClassifierMixin, AbstractClassifierLLM):

    unknown_detection_model: Union[InferenceHandler, Dict[str, List[str]]]
    unknown_detection_prompt: str = "ood.txt"

    classifier_model: Union[InferenceHandler, Dict[str, List[str]]]
    classifier_prompt: str = "multiclass.txt"

    # skip unknown detection
    skip_unknown_detection: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='before')
    def _init_models(data: dict):

        model_keys = ["unknown_detection_model", "classifier_model"]

        for model in model_keys:

            free_llms = []
            paid_llms = []

            for llm_category, llm_str_list in data[model].items():

                for llm_str in llm_str_list:

                    llm = StructuredRequestLLM.create_from_yaml_file(llm_str)

                    if llm_category == 'free_llms':
                        free_llms.append(llm)
                    elif llm_category == 'paid_llms':
                        paid_llms.append(llm)
                    else:
                        raise ValueError(f"Unrecognized llm_category {llm_category}")
                    

            data[model] = InferenceHandler(
                free_llms=free_llms,
                paid_llms=paid_llms
            )
            
        data_selector = data.get('selector')

        if data_selector is None:
            raise ValueError("No selector specified")
        
        if isinstance(data_selector, DictConfig):
            data_selector = dict(data_selector)
        
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

    def _detect_unknown_class(self, text: str, use_cache: bool = False) -> Optional[LogProbScore]:

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
    
        logprob_score = self._get_parsed_output(
            model=self.unknown_detection_model,
            valid_labels=PredictionV1.valid_labels,
            text=prompt, 
            retries=5,
            use_cache=use_cache
        )
        
        return logprob_score
    
    def _classify_known_classes(self, text: str, use_cache: bool = False) -> Optional[LogProbScore]:

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

        logprob_score = self._get_parsed_output(
            model=self.classifier_model,
            valid_labels=PredictionV1.valid_labels,
            text=prompt,
            use_cache=use_cache,
            retries=5 
        )

        return logprob_score

    def _single_predict(self, text: str, use_cache: bool = False, **kwargs) -> Tuple[str, float]:

        if self.y_train is None:
            raise ValueError("Not fitted")
        
        if self.x_train is None:
            raise ValueError("Not fitted")
        
        if self.classes is None:
            raise ValueError("Not fitted")

        if not self.skip_unknown_detection:
            
            unknown_prediction = self._detect_unknown_class(text=text, use_cache=use_cache)

            if unknown_prediction is None:
                return ErrorValues.PARSING_STR.value, float(ErrorValues.PARSING_NUM.value)

            if unknown_prediction.answer.label == "true":            
                return UnknownClassLabel.UNKNOWN_STR.value, 1
        
        known_prediction = self._classify_known_classes(text=text, use_cache=use_cache)

        if known_prediction is None:
            return ErrorValues.PARSING_STR.value, float(ErrorValues.PARSING_NUM.value)
        
        return known_prediction.answer.label, 0.0
        


if __name__ == '__main__':

    import numpy as np
    from typing import cast
    from src.util.load_hydra import get_hydra_config

    key = "ml__classifier"

    config = get_hydra_config(
        overrides=[
            f"{key}=two_stage_groq_llama_8"
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
