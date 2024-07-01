import numpy as np
import pandas as pd

from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain.output_parsers import RetryOutputParser
from typing import Dict, List, Union, Tuple

from src.ml.classifier.llm.cls.base import BaseLLM
from src.ml.classifier.llm.cls.util.prediction import Prediction
from src.ml.classifier.llm.cls.util.prompt import PromptCreator
from src.util.constants import DatasetColumn, LLMModels
from src.util.environment import PydanticEnvironment

env = PydanticEnvironment()

class FewShotLLM(BaseLLM):

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

    def _single_predict(self, text: str, output_reasoning: bool = False) -> Union[str, Tuple[str, str]]:

        if self.y_train is None:
            raise ValueError("Not fitted")
        
        if self.x_train is None:
            raise ValueError("Not fitted")
        
        if self.classes is None:
            raise ValueError("Not fitted")
        
        if self.parser is None:
            raise ValueError("Not fitted")
        
        prompt_creator = PromptCreator(
            text=text,
            classes=set(list(self.classes)),
            parser=self.parser
        )

        prompt = prompt_creator.create_few_shot_prompt(
            examples=self._get_examples()
        )

        retry_parser = RetryOutputParser.from_llm(parser=self.parser, llm=self.model, max_retries=3)

        completion_chain = prompt | self.model

        main_chain = RunnableParallel(
            completion=completion_chain, prompt_value=prompt
        ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

        try:

            answer: Prediction = main_chain.invoke({"query": text})
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

    import time

    data = pd.DataFrame({
        DatasetColumn.FEATURES: ["Hi, ich heiße Alex", "Auf Wiedersehen!"],
        DatasetColumn.LABEL: ['Greeting', 'Goodbye']
    })

    llm = FewShotLLM(
        model_str=LLMModels.LLAMA_3_8B_Remote_HF.value
    )

    llm.fit(
        x_train=data[DatasetColumn.FEATURES].values,
        y_train=data[DatasetColumn.LABEL].values,
        x_valid=data[DatasetColumn.FEATURES].values,
        y_valid=data[DatasetColumn.LABEL].values,
        
    )

    text_list = ["Buenos dias", "Eine Katze läuft über die Straße", "Go away", "Tschüss"] * 2


    # benchmark with batch predict

    start = time.time()

    result = llm.predict_batch(text_list)

    print("Time taken: ", time.time() - start)

    print(result)

    start = time.time()

    result = [llm._single_predict(text, output_reasoning=True) for text in text_list]
    print("Time taken: ", time.time() - start)

    for text, el in zip(text_list, result):
        print(el[0], text, el[1])

    

    
    

