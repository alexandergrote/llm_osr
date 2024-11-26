import numpy as np
import pandas as pd

from typing import Tuple

from src.ml.classifier.llm.base import AbstractClassifierLLM
from src.util.constants import DatasetColumn
from src.ml.classifier.llm.util.logprob import LogProbScore, LogProb
from src.ml.classifier.llm.util.prediction import Prediction



class RandomLLM(AbstractClassifierLLM):

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, **kwargs):
        
        self.y_train = np.concatenate([
           y_train, y_valid
       ])
    
        self.classes = np.unique(self.y_train)


    def _single_predict(self, text: str, use_cache: bool = False) -> Tuple[str, float]:

        if self.y_train is None:
            raise ValueError("Not fitted")
        
        
        if self.classes is None:
            raise ValueError("Not fitted")
        
        # set rng generator
        rng = np.random.default_rng(42)
        
        # select random answer from y_train
        answer = rng.choice(self.y_train)

        # create random score between 0 and 1
        score = rng.random()

        Prediction.valid_labels = self.classes
        
        logprob_score = LogProbScore(
            answer=Prediction(reasoning="naive", label=answer),
            logprobs=[LogProb.from_prob(text=answer, prob=score)]
        )

        if logprob_score is None:
            return None
        
        return logprob_score.answer.label, score


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

    

    
    

