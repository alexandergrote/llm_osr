import numpy as np
import pandas as pd
import os

from typing import Union, Tuple, List, Dict
from langchain.output_parsers import PydanticOutputParser
from pydantic import model_validator, ConfigDict
from omegaconf.dictconfig import DictConfig
from pathlib import Path

from src.util.dynamic_import import DynamicImport
from src.ml.classifier.llm.util.cosine_selector import CosineSelector
from src.ml.classifier.llm.util.prediction import Prediction
from src.ml.classifier.llm.base import AbstractClassifierLLM, LLMClassifierMixin
from src.util.constants import UnknownClassLabel
from src.ml.classifier.llm.util.rest import AbstractLLM, StructuredRequestLLM
from src.ml.classifier.llm.util.rest_inference import InferenceHandler
from src.ml.classifier.llm.util.prompt import PromptCreator, PROMPT_SCENARIOS
from src.util.constants import ErrorValues, DatasetColumn
from src.util.caching import PickleCacheHandler
from src.util.hashing import Hash


class OneStageLLM(LLMClassifierMixin, AbstractClassifierLLM):

    osr_model: Union[InferenceHandler, Dict[str, List[str]]]
    
    shuffle_free_llms: bool = False
    shuffle_paid_llms: bool = False

    # fewshot selection of data points
    selector: Union[CosineSelector, dict]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='before')
    def _init_models(data: dict):

        model_keys = ["osr_model"]

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

            shuffle_free_llms = data.get("shuffle_free_llms", False)

            assert isinstance(shuffle_free_llms, bool)

            data[model] = InferenceHandler(
                free_llms=free_llms,
                paid_llms=paid_llms,
                shuffle_free_llms=shuffle_free_llms
            )
        
        data_selector = data.get('selector')

        if data_selector is None:
            raise ValueError("No selector specified")
        
        if isinstance(data_selector, DictConfig):
            data_selector = dict(data_selector)
        
        if isinstance(data_selector, dict):
            data['selector'] = DynamicImport.init_class_from_dict(data_selector)

        return data
    
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

    def get_valid_labels(self) -> List[str]:

        if self.classes is None:
            raise ValueError("Classes not set")
        
        valid_labels = list(self.classes)
        result = valid_labels + [UnknownClassLabel.UNKNOWN_STR.value]
        assert all(isinstance(label, str) for label in result)
        return result

    def get_parser(self) -> PydanticOutputParser:
        
        Prediction.valid_labels = self.get_valid_labels()

        parser = PydanticOutputParser(pydantic_object=Prediction)

        return parser

    def get_prompt(self, text: str) -> str:

        examples = self._get_examples(text_to_classify=text)
        parser = self.get_parser()

        try:
            outlier_examples = self._get_outlier_examples(
                outlier_value=UnknownClassLabel.UNKNOWN_STR.value
            )

        except ValueError:
            outlier_examples = None

        prompt_creator = PROMPT_SCENARIOS[self.unknown_detection_scenario]

        if not isinstance(prompt_creator, PromptCreator):
            raise ValueError("Prompt creator must be an instance of PromptCreator")

        prompt = prompt_creator.create(
            text_to_classify=text,
            parser=parser,
            examples=examples,
            outlier_examples=outlier_examples,
        )

        return prompt

    def _single_predict(self, text: str, use_cache: bool = False, is_prompt: bool = False, **kwargs) -> Tuple[str, float]:

        if not isinstance(self.osr_model, AbstractLLM):
            raise ValueError("Classifier model must be an AbstractLLM")

        exp_name = kwargs.get('experiment_name', 'None')
        config = kwargs.get('config', 'None')
        seed = 'None'

        if isinstance(config, dict):
            seed = config.get('random_seed', 'None')

        base_filepath = Path(os.path.join(exp_name, str(seed)))
        full_filepath = base_filepath / Hash.hash(text)

        cache_handler = PickleCacheHandler(
            filepath=full_filepath
        )

        if self.use_cache:
            
            result = cache_handler.read()

            if result is not None:

                text, score = result
                # minor adjustment due to late corrections to label space
                text = text.replace('reverted_card_payment?', 'reverted_card_payment')

                return text, score

        
        valid_labels = self.get_valid_labels()

        if is_prompt:

            prompt = text

        else:

            if self.y_train is None:
                raise ValueError("Not fitted")
            
            if self.x_train is None:
                raise ValueError("Not fitted")
            
            if self.classes is None:
                raise ValueError("Not fitted")
            
            prompt = self.get_prompt(text=text)

        prediction = self._get_parsed_output(model=self.osr_model, valid_labels=valid_labels, use_cache=use_cache,  text=prompt, retries=5, **kwargs)

        if prediction is None:
            return ErrorValues.PARSING_STR.value, float(ErrorValues.PARSING_NUM.value)

        prediction_label, unknown_score = self.get_result_from_logprobscore(prediction)

        result = (prediction_label, unknown_score)

        cache_handler.write(
            result
        )

        return result

        
if __name__ == '__main__':

    import mlflow
    import numpy as np

    from typing import cast
    from src.util.load_hydra import get_hydra_config

    mlflow.set_experiment("Onestage")
    mlflow.tracing.enable()

    key = "ml__classifier"

    config = get_hydra_config(
        overrides=[
            f"{key}=one_stage_llama_8",
            #f"{key}.params.osr_model.paid_llms=['nebius-llama-70b.yaml']"
        ]
    )

    llm = DynamicImport.init_class_from_dict(
        dictionary=config[key],
    )

    llm.use_cache = False

    # Explicitly cast llm to TwoStageLLM
    llm = cast(OneStageLLM, llm)

    
    data_train = pd.DataFrame({
        DatasetColumn.FEATURES: ["Ich heiße Marvin", "Auf Wiedersehen!", "Hallo", "Wo kann ich Kekse kaufen?", "Die Apfelsaftschorle finde ich wo?", "Das Essen ist sehr lecker", "München ist eine große Stadt."],
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

    result = llm._single_predict(text="Hello", use_cache=False)

    print(result)

    result2 = llm.predict(
        x=np.array([["Hola"], ["Hallo du"], ["Ich möchte einen Tee bestellen"], ['Ich wohne in Bayern'], ["I like trains"]], dtype=np.object_),
        include_outlierscore=True,
    )

    print(result2)
