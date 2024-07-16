import numpy as np

from pydantic import BaseModel
from typing import List

from src.util.constants import UnknownClassLabel
from src.ml.classifier.llm.util.prediction import Prediction
from src.util.types import LogProb   


class LogProbScore(BaseModel):

    answer: Prediction
    logprobs: List[LogProb]

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def calculate_linear_prob(log_prob):
        return np.exp(log_prob)
    
    @property
    def tokens(self) -> List[str]:
        return [log_prob.text for log_prob in self.logprobs]

    @property
    def probs(self) -> List[float]:
        return [self.calculate_linear_prob(log_prob=log_prob.logprob) for log_prob in self.logprobs]
    
    @property
    def entropy(self) -> float:
        probs = np.array(self.probs) 
        return -np.sum(probs * np.log(probs))
    
    @property
    def unknown_score(self) -> float:

        avg_probs = np.mean(self.probs)

        if self.answer.label == UnknownClassLabel.UNKNOWN_STR.value:
            return avg_probs
        
        return 1 - avg_probs

    
if __name__ == '__main__':

    Prediction.valid_labels = ['Paris', 'London', 'Berlin']
    
    prediction = Prediction(
        label='Paris',
        reasoning='Because it is the capital of France'
    )

    log_prob_result = LogProbScore(
        answer=prediction,
        logprobs=[LogProb(text='Paris', logprob=-0.1), LogProb(text='London', logprob=-0.2), LogProb(text='Berlin', logprob=-0.3)]
    )
    
    print(log_prob_result.probs)
    print(log_prob_result.entropy)
    print(log_prob_result.unknown_score)
    