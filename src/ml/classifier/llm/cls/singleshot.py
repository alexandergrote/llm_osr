import pandas as pd

from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain.output_parsers import RetryOutputParser

from typing import Union, Tuple

from src.ml.classifier.llm.cls.base import BaseLLM
from src.ml.classifier.llm.cls.util.prediction import Prediction
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

        prompt = prompt_creator.create_single_shot_prompt()

        prompt_fmt = prompt.format_prompt(query=text)

        retry_parser = RetryOutputParser.from_llm(parser=self.parser, llm=self.model, max_retries=3)
        
        completion_chain = prompt | self.model


        main_chain = RunnableParallel(
            completion=completion_chain, prompt_value=prompt
        ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

        try:
            chain_input: str = PromptCreator.get_chain_input_field_name()
            answer: Prediction = main_chain.invoke({chain_input: text})
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

