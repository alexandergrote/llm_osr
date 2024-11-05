import numpy as np
import pandas as pd

from copy import copy
from typing import Optional, Union, Tuple
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import model_validator, ConfigDict
from langchain_core.prompts import PromptTemplate

from src.util.dynamic_import import DynamicImport
from src.ml.classifier.llm.util.cosine_selector import CosineSelector
from src.ml.classifier.llm.util.prediction import PredictionV1, Prediction
from src.ml.classifier.llm.util.request import RequestOutput
from src.ml.classifier.llm.util.logprob import LogProbScore
from src.ml.classifier.llm.base import AbstractClassifierLLM, LLMClassifierMixin
from src.ml.classifier.llm.util.prompt import FewshotPromptCreator
from src.util.constants import DatasetColumn, LLMModels
from src.ml.classifier.llm.util.mapping import LLM_Mapping
from src.util.constants import UnknownClassLabel
from src.ml.classifier.llm.util.rest import AbstractLLM
from src.ml.classifier.llm.factory import LLMClassifierFactory
from src.util.constants import Directory, ErrorValues

# prompts taken from retry output parser from langchain
NAIVE_COMPLETION_RETRY = """Prompt:
{prompt}
Completion:
{completion}

Above, the Completion did not satisfy the constraints given in the Prompt.
Please try again:"""

NAIVE_RETRY_PROMPT = PromptTemplate.from_template(NAIVE_COMPLETION_RETRY)


class FewShotLLM(AbstractClassifierLLM):

    model: AbstractLLM
    clf_str: str

    parser: Optional[PydanticOutputParser] = None
    selector: Optional[Union[dict, CosineSelector]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='before')
    def _init_model(data: dict):

        data['model'] = LLM_Mapping[LLMModels(data['clf_str'])]

        data_selector = data.get('selector')

        if isinstance(data_selector, dict):
            data['selector'] = DynamicImport.init_class_from_dict(data_selector)

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

       if self.selector is not None:
           
           pass
               
       # Example usage
       valid_labels = [UnknownClassLabel.UNKNOWN_STR.value] + list(self.classes)
       PredictionV1.valid_labels = valid_labels
       
       self.parser = PydanticOutputParser(pydantic_object=PredictionV1)

    def _single_predict(self, text: str, **kwargs) -> Tuple[str, float]:

        if self.y_train is None:
            raise ValueError("Not fitted")
        
        if self.x_train is None:
            raise ValueError("Not fitted")
        
        if self.classes is None:
            raise ValueError("Not fitted")
        
        if self.parser is None:
            raise ValueError("Not fitted")
        
        classes_msg = '\n'.join(list(self.classes))
        system_msg = "You are an Open Set Recognition Classifier. Your goal is to 1) reject unknown classes and 2) classifier known classes into their corresponding classes"
        task_desc = f"You are asked to classify the following text: {{{FewshotPromptCreator.get_chain_input_field_name()}}}"
        cls_prompt = f"You must reply with one of these classes: \n{classes_msg}\nIf the query does not belong to any of these classes, answer with 'unknown'."

        prefix_prompt = f"{system_msg}\n{task_desc}\n{cls_prompt}\nHere are some examples:"

        suffix_prompt = f"You must give the answer by following these instructions {{{FewshotPromptCreator.get_format_instruction_field()}}}"

        prompt_creator = FewshotPromptCreator(
            text_to_classify=text,
            examples=self._get_examples(text_to_classify=text),
            prefix_prompt=prefix_prompt,
            suffix_prompt=suffix_prompt,
            parser=self.parser
        )

        prompt = prompt_creator.create()
            
        logprob_score = self._get_parsed_output(model=self.model, parser=self.parser, prompt=prompt, retries=3)

        if logprob_score is None:
            raise ValueError("Result cannot be None")
        
        return logprob_score, logprob_score.unknown_score

class OneStageLLM(LLMClassifierMixin, AbstractClassifierLLM):

    osr_model: Union[AbstractLLM, dict]
    osr_prompt: str = "osr.txt"

    # fewshot selection of data points
    selector: Optional[Union[dict, CosineSelector]] = None
    n_classes: Optional[int] = 3
    n_datapoints_per_class: Optional[int] = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='before')
    def _init_models(data: dict):

        model_keys = ["osr_model"]

        for model in model_keys:
            data[model] = LLMClassifierFactory(**data[model]).create()
        
        data_selector = data.get('selector')

        if isinstance(data_selector, dict):
            data['selector'] = DynamicImport.init_class_from_dict(data_selector)

        return data
    
    @model_validator(mode='after')
    def _init_model_prompts(self):

        for prompt_attr_name in ["osr_prompt"]:

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

    def _single_predict(self, text: str, **kwargs) -> Tuple[str, float]:

        if self.y_train is None:
            raise ValueError("Not fitted")
        
        if self.x_train is None:
            raise ValueError("Not fitted")
        
        if self.classes is None:
            raise ValueError("Not fitted")
        
        valid_labels = list(self.classes)
        PredictionV1.valid_labels = valid_labels + [UnknownClassLabel.UNKNOWN_STR.value]
        
        parser = PydanticOutputParser(pydantic_object=PredictionV1)
        instructions = parser.get_format_instructions()

        examples = self._get_examples(
            text_to_classify=text,
        )

        classes_msg = '\n'.join(valid_labels)
        examples_msg = '\n'.join([f"{example['input']} -> {example['output']}" for example in examples])

        prompt = self.osr_prompt.format(
            examples_msg=examples_msg,
            text=text,
            instructions=instructions,
            classes_msg=classes_msg,
            unknown_label=UnknownClassLabel.UNKNOWN_STR.value
        )

        if not isinstance(self.osr_model, AbstractLLM):
            raise ValueError("Classifier model must be an AbstractLLM")

        prediction = self._get_parsed_output(model=self.osr_model, parser=parser, prompt=prompt, retries=5)

        if prediction is None:
            return ErrorValues.PARSING_STR.value, float(ErrorValues.PARSING_NUM.value)

        prediction_label = prediction.answer.label

        unknown_score = 0.0 if prediction_label != UnknownClassLabel.UNKNOWN_STR.value else 1.0

        return prediction_label, unknown_score

        
if __name__ == '__main__':

    import numpy as np
    from typing import cast
    from src.util.load_hydra import get_hydra_config

    key = "ml__classifier"

    config = get_hydra_config(
        overrides=[
            f"{key}=one_stage_llm_llama"
        ]
    )

    llm = DynamicImport.init_class_from_dict(
        dictionary=config[key],
    )

    # Explicitly cast llm to TwoStageLLM
    llm = cast(OneStageLLM, llm)

    
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
