import numpy as np

from tqdm import tqdm
from langchain_core import pydantic_v1
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.prompts.few_shot import FewShotPromptTemplate, PromptTemplate
from langchain.output_parsers import PydanticOutputParser, RetryOutputParser
from langchain_community.cache import SQLiteCache
from pydantic import BaseModel, model_validator
from numpy import ndarray
from typing import Any, Optional, Dict, List

import pandas as pd

from src.ml.classifier.base import BaseClassifier
from src.ml.classifier.util.llm_models import LLM_Mapping, LLMModels
from src.util.constants import DatasetColumn


class Prediction(pydantic_v1.BaseModel):
    reasoning: str = pydantic_v1.Field(description="The reasoning behind the prediction")
    label: str = pydantic_v1.Field(description="The predicted label")

    @pydantic_v1.validator("label", allow_reuse=True)
    def label_must_be_valid(cls, label: str):

        if not hasattr(cls, 'valid_labels'):
            raise Exception("No class labels supplied")
        
        valid_labels = cls.valid_labels

        if label.lower() not in [valid_label.lower() for valid_label in valid_labels]:
            raise ValueError(f"Label must be one of {valid_labels}")
        return label


class LLM(BaseModel, BaseClassifier):

    model: Any
    model_str: str

    # placeholders, values will be set later
    x_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    classes: Optional[np.ndarray] = None
    parser: Optional[Any] = None
    
    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='before')
    def _init_model(data: dict):

        data['model'] = LLM_Mapping[LLMModels(data['model_str'])]

        return data

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
       valid_labels = ["unknown"] + list(self.classes)
       Prediction.valid_labels = valid_labels
       
       self.parser = PydanticOutputParser(pydantic_object=Prediction)

    def _get_examples(self, query: str = 'input', answer: str = 'output') -> List[Dict[str, str]]:

        if self.y_train is None:
            raise ValueError("Model is not fitted")
        
        if self.x_train is None:
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

    def _single_predict(self, text: str) -> str:

        if self.y_train is None:
            raise ValueError("Not fitted")
        
        if self.x_train is None:
            raise ValueError("Not fitted")
        
        if self.classes is None:
            raise ValueError("Not fitted")
        
        if self.parser is None:
            raise ValueError("Not fitted")


        query = f"{text} -> <your response>"

        example_prompt = PromptTemplate(
            template="{input} -> {output}",
            input_variables=["input", "output"]
        )

        system_msg = "You are a classifier for an Open Set Recognition problem."
        classes_msg = '\n'.join(self.classes)
        classes_prompt = f"You must reply with one of these classes: \n{classes_msg}\nIf the query does not belong to any of these classes, answer with 'unknown'"
        task_prompt = "Let's think step by step! Question: {query}"
        prefix_prompt = f"{system_msg}\n{classes_prompt}\n{task_prompt}"
        suffix_prompt = "Provide your final desired output in the following format: {format_instructions}"

        examples = self._get_examples()

        prompt = FewShotPromptTemplate(
            prefix=prefix_prompt,
            suffix=suffix_prompt,
            examples=examples,
            example_prompt=example_prompt,
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
            validate_template=True
        )

        retry_parser = RetryOutputParser.from_llm(parser=self.parser, llm=self.model, max_retries=3)

        completion_chain = prompt | self.model

        main_chain = RunnableParallel(
            completion=completion_chain, prompt_value=prompt
        ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

        try:
            
            cache = SQLiteCache(database_path='.langchain.db')
            look_up = cache.lookup(prompt=query, llm_string=self.model_str)

            if look_up is not None:
                return look_up[0]
            
            answer: Prediction = main_chain.invoke({"query": query})
            answer_final: str = answer.label

            cache.update(
                prompt=query,
                llm_string=self.model_str,
                return_val=[answer.label]
            )

        except Exception as e:

            print(e)

            answer_final = 'ERROR'

        return answer_final

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:

        result = []

        for el in tqdm(x):
            result.append(self._single_predict(text=el))
        
        return np.array(result)
    
    def predict_proba(self, x: ndarray, **kwargs) -> ndarray:
        pass

if __name__ == '__main__':

    data = pd.DataFrame({
        DatasetColumn.FEATURES: ["Ich heiße Alex", "Auf Wiedersehen!"],
        DatasetColumn.LABEL: ['Greeting', 'Goodbye']
    })

    llm = LLM(
        model_str=LLMModels.OAI_GPT3.value
    )

    llm.fit(
        x_train=data[DatasetColumn.FEATURES].values,
        y_train=data[DatasetColumn.LABEL].values,
        x_valid=data[DatasetColumn.FEATURES].values,
        y_valid=data[DatasetColumn.LABEL].values,
        
    )

    data_inf = pd.DataFrame({
        DatasetColumn.FEATURES: ['Eine Katze läuft über die Straße']
    })

    result = llm.predict(data_inf[DatasetColumn.FEATURES].values)

    print(result)

