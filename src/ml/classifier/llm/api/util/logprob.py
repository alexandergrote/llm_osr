import hashlib
import numpy as np
import json

from langchain_core import pydantic_v1
from typing import List, Dict

from src.ml.classifier.llm.cls.util.prediction import Prediction

class LLMLogProbResult(pydantic_v1.BaseModel):

    model: str
    query: str
    answer: Prediction
    logprobs: Dict[str, float]

    class Config:
        arbitrary_types_allowed = True

    @property
    def filename(self) -> str:
        id_str = self.model + '_' + self.query
        return hashlib.md5(id_str.encode()).hexdigest() + '.json'

    @property
    def probs(self) -> Dict[str, float]:

        p_linear_dict = {}

        for key, value in self.logprobs.items():
            p_linear_dict[key] = np.exp(value)

        return p_linear_dict
    
    @property
    def answer_probability(self) -> float:
        log_prob = self.probs[self.answer.label]
        return self.calculate_linear_prob(log_prob)
    
    @property
    def entropy(self) -> float:
        probs = np.array(list(self.probs.values())) 
        return -np.sum(probs * np.log(probs))
    
    @staticmethod
    def calculate_linear_prob(log_prob):
        return np.exp(log_prob)
    
    def write_to_json(self):

        with open(self.filename, 'w') as f:
            json.dump(self.json(), f)

    @classmethod
    def from_json(cls, filename: str):

        with open(filename, 'r') as f:
            data = json.load(f)         
    
    
if __name__ == '__main__':

    Prediction.valid_labels = ['Paris', 'London', 'Berlin']
    
    prediction = Prediction(
        label='Paris',
        reasoning='Because it is the capital of France'
    )

    log_prob_result = LLMLogProbResult(
        model='gpt-3',
        query='What is the capital of France?',
        answer=prediction,
        logprobs={
            'Paris': -0.1,
            'London': -0.2,
            'Berlin': -0.3
        }
    )
    
    print(log_prob_result.filename)
    print(log_prob_result.json())
    print(log_prob_result.probs)
    print(log_prob_result.answer_probability)
    print(log_prob_result.entropy)

    log_prob_result.write_to_json()
    