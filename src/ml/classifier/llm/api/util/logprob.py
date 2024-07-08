import numpy as np

from langchain_core import pydantic_v1
from typing import List, Dict, Union

from src.ml.classifier.llm.cls.util.prediction import Prediction   


class OAILogProbResult(pydantic_v1.BaseModel):

    model: str
    query: str
    answer: Prediction
    logprobs: Union[Dict[str, float], List[Dict[str, float]]]

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def calculate_linear_prob(log_prob):
        return np.exp(log_prob)
    
    def _get_probs_from_dict(self, logprobs: Dict[str, float]) -> Dict[str, float]:
        p_linear_dict = {}

        for key, value in logprobs.items():
            p_linear_dict[key] = self.calculate_linear_prob(value)

        return p_linear_dict
    
    def _get_probs_from_list(self, logprobs: List[Dict[str, float]]) -> List[Dict[str, float]]:
        return [self._get_probs_from_dict(d) for d in logprobs]

    @property
    def probs(self) -> Dict[str, float]:

        p_linear_dict = {}

        for key, value in self.logprobs.items():
            p_linear_dict[key] = self.calculate_linear_prob(value)

        return p_linear_dict
    
    @property
    def answer_probability(self) -> float:
        log_prob = self.probs[self.answer.label]
        return self.calculate_linear_prob(log_prob)
    
    @property
    def entropy(self) -> float:
        probs = np.array(list(self.probs.values())) 
        return -np.sum(probs * np.log(probs))    
    
    
if __name__ == '__main__':

    Prediction.valid_labels = ['Paris', 'London', 'Berlin']
    
    prediction = Prediction(
        label='Paris',
        reasoning='Because it is the capital of France'
    )

    log_prob_result = OAILogProbResult(
        model='gpt-3',
        query='What is the capital of France?',
        answer=prediction,
        logprobs={
            'Paris': -0.1,
            'London': -0.2,
            'Berlin': -0.3
        }
    )
    
    print(log_prob_result.probs)
    print(log_prob_result.answer_probability)
    print(log_prob_result.entropy)
    