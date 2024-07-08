import pandas as pd

from langchain.output_parsers import RetryOutputParser
from typing import Union, Tuple

from src.ml.classifier.llm.cls.base import BaseLLM
from src.ml.classifier.llm.cls.util.prompt import PromptCreator
from src.util.constants import DatasetColumn, LLMModels


class SingleShotLLM(BaseLLM):

    def _single_predict(self, text: str, output_reasoning: bool = False) -> Union[str, Tuple[str, str]]:
        
        if self.classes is None:
            raise ValueError("Not fitted")
        
        if self.parser is None:
            raise ValueError("Not fitted")

        prompt_creator = PromptCreator(
            text=text,
            classes=set(list(self.classes)),
            parser=self.parser
        )

        prompt = prompt_creator.create_zero_shot_prompt()
        prompt_key = PromptCreator.get_chain_input_field_name()
        prompt_str = prompt.format(**{prompt_key: text})

        retry_parser = RetryOutputParser.from_llm(parser=self.parser, llm=self.model, max_retries=3)

        try:
            answer = self._retry(model=self.model, parser=retry_parser, prompt=prompt_str, retries=3)

            if answer is None:
                raise ValueError("No answer found")
            
            answer_final: str = answer.label
            reasoning_final: str = answer.reasoning

        except Exception as e:

            print(e)

            answer_final = 'ERROR'
            reasoning_final = 'ERROR'

        if output_reasoning is True:
            return answer_final, reasoning_final

        return answer_final


if __name__ == '__main__':

    data = pd.DataFrame({
        DatasetColumn.FEATURES: ["Ich heiße Alex", "Auf Wiedersehen!"],
        DatasetColumn.LABEL: ['Greeting', 'Goodbye']
    })

    llm = SingleShotLLM(
        model_str=LLMModels.LLAMA_3_8B_Remote_HF.value
    )

    llm.fit(
        x_train=data[DatasetColumn.FEATURES].values,
        y_train=data[DatasetColumn.LABEL].values,
        x_valid=data[DatasetColumn.FEATURES].values,
        y_valid=data[DatasetColumn.LABEL].values,
        
    )

    result = llm._single_predict(text="Hello")

    print(result)

    result = llm.predict_batch(["Hello", "Goodbye"])
    print(result)

