import numpy as np
import pandas as pd

from typing import Dict, List, Optional

from src.ml.classifier.llm.util.logprob import LogProbScore
from src.ml.classifier.llm.base import AbstractClassifierLLM
from src.ml.classifier.llm.util.prompt import PromptCreator
from src.util.constants import DatasetColumn, LLMModels
from src.util.environment import PydanticEnvironment

env = PydanticEnvironment()

class FewShotLLM(AbstractClassifierLLM):

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


    def _single_predict(self, text: str) -> Optional[LogProbScore]:

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
        prompt_key = PromptCreator.get_chain_input_field_name()
        prompt_str = prompt.format(**{prompt_key: text})
            
        logprob_score = self._get_parsed_output(model=self.model, parser=self.parser, prompt=prompt_str, retries=3)

        if logprob_score is None:
            return None
        
        return logprob_score


if __name__ == '__main__':

    import numpy as np

    data = pd.DataFrame({
        DatasetColumn.FEATURES: ["Ich heiße Alex", "Auf Wiedersehen!", "Hallo"],
        DatasetColumn.LABEL: ['Greeting', 'Goodbye', "Greeting"]
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

    result = llm._single_predict(text="Hello")

    print(result)

    result2 = llm.predict(
        x=np.array([["Hello"]], dtype=np.object_),
        include_outlierscore=True
    )

    print(result2)

    

    
    

