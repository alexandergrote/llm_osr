import numpy as np
import pandas as pd

from typing import Tuple

from src.ml.classifier.llm.base import AbstractClassifierLLM
from src.util.constants import DatasetColumn
from src.ml.classifier.llm.util.logprob import LogProbScore, LogProb
from src.ml.classifier.llm.util.prediction import Prediction



class RandomLLM(AbstractClassifierLLM):

    fixed_random_seed: bool = True
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, **kwargs):
        
        self.y_train = np.concatenate([
           y_train, y_valid
       ])
    
        self.classes = np.unique(self.y_train)


    def _single_predict(self, text: str, use_cache: bool = False, is_prompt: bool = False, **kwargs) -> Tuple[str, float]:

        if self.y_train is None:
            raise ValueError("Not fitted")
        
        
        if self.classes is None:
            raise ValueError("Not fitted")
        
        # set rng generator
        if self.fixed_random_seed:
            rng = np.random.default_rng(42)
        else:
            rng = np.random.default_rng()
        
        # select random answer from y_train
        answer = rng.choice(self.y_train)

        assert isinstance(answer, str), f"Answer must be a string, got {type(answer)}"

        # create random score between 0 and 1
        score = rng.random()
        
        return answer, score

    def single_predict(self, text: str, use_cache: bool = False, **kwargs) -> Tuple[str, float]:
        return self._single_predict(text=text, use_cache=use_cache, **kwargs)


if __name__ == '__main__':

    import numpy as np

    data = pd.DataFrame({
        DatasetColumn.FEATURES: ["Ich heiße Alex", "Auf Wiedersehen!", "Hallo"],
        DatasetColumn.LABEL: ['Greeting', 'Goodbye', "Greeting"]
    })

    llm = RandomLLM()

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

    

    
    

