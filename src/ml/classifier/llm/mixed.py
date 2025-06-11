import numpy as np
import pandas as pd
import os

from typing import Optional, Union, Tuple, List, Dict
from langchain.output_parsers import PydanticOutputParser
from pydantic import model_validator, ConfigDict
from omegaconf.dictconfig import DictConfig
from pathlib import Path

from src.util.dynamic_import import DynamicImport
from src.ml.classifier.llm.util.prediction import Prediction
from src.ml.classifier.llm.util.logprob import LogProbScore
from src.ml.classifier.llm.base import AbstractClassifierLLM
from src.util.constants import UnknownClassLabel, DatasetColumn
from src.ml.classifier.llm.util.outlier import OutlierValue
from src.ml.classifier.llm.util.rest import AbstractLLM, StructuredRequestLLM
from src.ml.classifier.llm.naive import RandomLLM
from src.ml.classifier.llm.util.rest_inference import InferenceHandler
from src.util.constants import ErrorValues
from src.ml.classifier.llm.base import LLMClassifierMixin
from src.util.caching import PickleCacheHandler
from src.util.hashing import Hash
from src.ml.classifier.llm.util.prompt import PromptCreator, PromptScenarioName, PROMPT_SCENARIOS

random_llm = RandomLLM(
    selector=dict(),
    unknown_detection_scenario=PromptScenarioName.EXPLICIT_WITH_LABELS,
    unknown_detection_model_name="random",
    fixed_random_seed=False
)

class PromptFirstRandomSecond(LLMClassifierMixin, AbstractClassifierLLM):

    unknown_detection_model: Union[InferenceHandler, Dict[str, List[str]]]
    
    classifier_model: RandomLLM = random_llm

    # shuffle free llm apis to equal use
    shuffle_free_llms: bool = False
    shuffle_paid_llms: bool = False
    use_classes_in_examples: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='before')
    def _init_models(data: dict):

        model_keys = ["unknown_detection_model"]

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
            shufle_paid_llms = data.get("shuffle_paid_llms", False)

            assert isinstance(shuffle_free_llms, bool)
            assert isinstance(shufle_paid_llms, bool)

            data[model] = InferenceHandler(
                free_llms=free_llms,
                paid_llms=paid_llms,
                shuffle_free_llms=shuffle_free_llms,
                shuffle_paid_llms=shufle_paid_llms
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

       self.classifier_model.x_train = self.x_train
       self.classifier_model.y_train = self.y_train
       self.classifier_model.x_valid = self.x_valid
       self.classifier_model.y_valid = self.y_valid
       self.classifier_model.classes = self.classes


    def _detect_unknown_class(self, text: str, use_cache: bool = False, **kwargs) -> Optional[LogProbScore]:

        if self.y_train is None:
            raise ValueError("Not fitted")
        
        if self.x_train is None:
            raise ValueError("Not fitted")
        
        if self.classes is None:
            raise ValueError("Not fitted")

        if 'pbar' in kwargs:
            pbar = kwargs['pbar']

            if hasattr(pbar, 'write'):
                pbar.write("Unknown Detection")


        # Example usage
        Prediction.valid_labels = OutlierValue.list()
        
        parser = PydanticOutputParser(pydantic_object=Prediction)
        
        examples = self._get_examples(
            text_to_classify=text,
        )

        outlier_examples = self._get_outlier_examples(
            outlier_value=OutlierValue.OUTLIER.value
        )

        prompt_creator = PROMPT_SCENARIOS[self.unknown_detection_scenario]

        if not isinstance(prompt_creator, PromptCreator):
            raise ValueError("Prompt creator must be an instance of PromptCreator")

        prompt = prompt_creator.create(
            text_to_classify=text,
            parser=parser,
            examples=examples,
            outlier_examples=outlier_examples,
        )

        if not isinstance(self.unknown_detection_model, AbstractLLM):
            raise ValueError("Unknown detection model must be of type AbstractLLM")

        logprob_score = self._get_parsed_output(
            model=self.unknown_detection_model,
            valid_labels=Prediction.valid_labels,
            text=prompt, 
            retries=5,
            use_cache=use_cache,
            **kwargs
        )
        
        return logprob_score

    def _single_predict(self, text: str, use_cache: bool = False, is_prompt: bool = False, **kwargs) -> Tuple[str, float]:

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
                return result[0], result[1]
        
        # checks for first model
        if self.y_train is None:
            raise ValueError("Not fitted")
        
        if self.x_train is None:
            raise ValueError("Not fitted")
        
        if self.classes is None:
            raise ValueError("Not fitted")

        # checks for second model
        if self.classifier_model.y_train is None:
            raise ValueError("Not fitted")
        
        if self.classifier_model.x_train is None:
            raise ValueError("Not fitted")
        
        if self.classifier_model.classes is None:
            raise ValueError("Not fitted")

        unknown_prediction = self._detect_unknown_class(text=text, use_cache=use_cache, **kwargs)

        if unknown_prediction is None:
            return ErrorValues.PARSING_STR.value, float(ErrorValues.PARSING_NUM.value)

        if unknown_prediction.answer.label == OutlierValue.OUTLIER.value:

            result = (UnknownClassLabel.UNKNOWN_STR.value, 1)

            cache_handler.write(
                result
            )

            return result

        result = self.classifier_model.single_predict(text=text, use_cache=use_cache, **kwargs)

        cache_handler.write(
            result
        )

        return result
        

if __name__ == '__main__':

    import mlflow
    import numpy as np
    from typing import cast
    from src.util.load_hydra import get_hydra_config

    mlflow.set_experiment("PromptFirstRandomSecond")
    mlflow.tracing.enable()

    key = "ml__classifier"

    config = get_hydra_config(
        overrides=[
            f"{key}=mixed_llama_8"
        ]
    )

    llm = DynamicImport.init_class_from_dict(
        dictionary=config[key],
    )

    # Explicitly cast llm
    llm = cast(PromptFirstRandomSecond, llm)
    
    data_train = pd.DataFrame({
        DatasetColumn.FEATURES: ["Erdbeeren sind lecker", "Auf Wiedersehen!", "Hallo", "Wo kann ich Kekse kaufen?", "Die Apfelsaftschorle finde ich wo?", "Das Essen ist sehr lecker", "München ist eine große Stadt."],
        DatasetColumn.LABEL: ['Food', 'Farewell', "Greeting", "Food", "Food", "Food", "City"]
    })

    data_valid = pd.DataFrame({
        DatasetColumn.FEATURES: ["Kartoffeln sind was feines", "Ich war noch nie in Berlin", "Ich spiele gerne Fußball"],
        DatasetColumn.LABEL: ['Food', 'City', 'Outlier']
    })

    llm.fit(
        x_train=data_train[DatasetColumn.FEATURES].values,
        y_train=data_train[DatasetColumn.LABEL].values,
        x_valid=data_valid[DatasetColumn.FEATURES].values,
        y_valid=data_valid[DatasetColumn.LABEL].values,
    )

    x_test = np.array([['Pfirsiche sind lecker und der Apfel auch'], ["Ich morgen in die Schule"], ["Ich möchte einen Tee bestellen"], ['Ich mag Züge'], ["I like trains"]], dtype=np.object_)
    y_test = np.array(["food", "unknown", "unknown", "unknown", "unknown"])

    for use_class in [True, False]:

        llm.use_cache = False
        llm.use_classes_in_examples = use_class

        result = llm._single_predict(text="Hello")

        print(result)

        result2 = llm.predict(
            x=x_test,
            include_outlierscore=True
        )

        y_pred = result2[0]
        
        print(y_pred)
        
