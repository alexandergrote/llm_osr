import numpy as np

from langchain_core import pydantic_v1
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.prompts.few_shot import FewShotPromptTemplate, PromptTemplate
from langchain.output_parsers import PydanticOutputParser, RetryOutputParser
from langchain_community.cache import SQLiteCache
from pydantic import BaseModel, model_validator
from numpy import ndarray
from typing import Any

import pandas as pd

from src.ml.classifier.base import BaseClassifier
from src.ml.classifier.util.llm_models import LLM_Mapping, LLMModels


class Prediction(pydantic_v1.BaseModel):
    reasoning: str = pydantic_v1.Field(description="The reasoning behind the prediction")
    label: str = pydantic_v1.Field(description="The predicted label")

    @pydantic_v1.validator("label")
    def label_must_be_valid(cls, label: str):
        if label.lower() not in ["positive", "negative"]:
            raise ValueError("Label must be either 'positive' or 'negative'")
        return label

parser = PydanticOutputParser(pydantic_object=Prediction)

class LLM(BaseModel, BaseClassifier):

    model: Any
    
    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='before')
    def _init_model(data: dict):

        if 'model' not in data:
            data['model'] = LLM_Mapping[LLMModels.OAI_GPT3]

        return data

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        **kwargs
    ):
        pass

    def predict(self, x: np.ndarray, **kwargs) -> pd.DataFrame:

        example_prompt = PromptTemplate(
            template="{input} -> {output}",
            input_variables=["input", "output"]
        )

        reasoning_prompt = "Let's think step by step!\nQuestion:{input}\nProvide your final desired output in the following format: {format_instructions}"

        prompt = FewShotPromptTemplate(
            suffix=reasoning_prompt,
            examples=[{'input': "Ich bin glücklich", 'output': "positive"}, {'input': "Ich bin traurig", 'output': "negative"}],
            example_prompt=example_prompt,
            input_variables=["input"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        retry_parser = RetryOutputParser.from_llm(parser=parser, llm=self.model, max_retries=3)

        completion_chain = prompt | self.model

        main_chain = RunnableParallel(
            completion=completion_chain, prompt_value=prompt
        ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

        try:

            look_up_key = "I am happy --> <your response>"
            cache_key = 'my_llm'

            cache = SQLiteCache(database_path='.langchain.db')
            look_up = cache.lookup(prompt=look_up_key, llm_string=cache_key)

            if look_up is not None:
                return look_up
            
            print('caching')
            
            answer: Prediction = main_chain.invoke({"input": input})
            answer_final: str = answer.label

            cache.update(
                prompt=look_up_key,
                llm_string=cache_key,
                return_val=[answer.label]
            )

        except Exception as e:

            print(e)

            answer_final = 'ERROR'

        return answer_final
    
    def predict_proba(self, x: ndarray, **kwargs) -> ndarray:
        pass

if __name__ == '__main__':

    data = pd.DataFrame({'text': ["Ich bin glücklich", "Ich bin traurig"]})
    llm = LLM()
    result = llm.predict(data)

    print(result)

