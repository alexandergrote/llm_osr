from pydantic import BaseModel
from openai import OpenAI
import numpy as np

from typing import List, Dict, Tuple

from src.util.types import LogProb
from src.ml.classifier.llm.api.base import AbstractLLM
from src.ml.classifier.llm.util.backoff import BackoffMixin
from src.util.environment import PydanticEnvironment

env = PydanticEnvironment()


def get_completion(
    messages: List[Dict[str, str]],
    model: str,
    max_tokens=1000,
    temperature=0,
    stop=None,
    seed=123,
    tools=None,
    logprobs=True,  # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message..
    top_logprobs=2,
) -> str:

    # taken from https://cookbook.openai.com/examples/using_logprobs

    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }

    if tools:
        params["tools"] = tools

    client = OpenAI(api_key=env.openai_key)
    completion = client.chat.completions.create(**params)

    return completion


class OpenAIWrapper(AbstractLLM, BackoffMixin, BaseModel):

    name: str
    
    def __call__(self, *, prompt: str, **kwargs) -> Tuple[str, List[LogProb]]:
        

        completion = self.completion_with_backoff(
            get_completion,
            messages=[{"role": "user", "content": prompt}],
            model=self.name,
            **kwargs
        )

        assert hasattr(completion, 'choices'), 'Completion does not have choices'
        assert len(completion.choices) == 1, 'Completion does not have one choice'
        assert hasattr(completion.choices[0], 'message'), 'Completion choice does not have message'
        assert hasattr(completion.choices[0].message, 'content'), 'Completion choice message does not have content'

        text = completion.choices[0].message.content
        logprob_list = [LogProb(token=k, logprob=v) for k, v in completion.choices[0].logprobs.content]
        
        return text, logprob_list
    

if __name__ == '__main__':

    model = OpenAIWrapper(name='gpt-3.5-turbo-0125')

    prompt = """you are an open set recognition classifier.\n"""
    prompt += """you are given a non-exhaustive list of examples for each class.\n"""
    prompt += """you are asked to classify a new example and assign it a class. match it by meaning and not by exact match \n"""
    prompt += """if the example is not in the set of classes, you should return 'unknown'.\n"""
    prompt += """classes: ['0', '1', '2']\n"""
    prompt += """examples: 
    - '0': 'meow', 'purr', ...
    - '1': 'bark', 'woof', ...
    - '2': 'tweet', 'chirp', ... \n"""
    prompt += """incoming query: 'riding a bike'\n"""
    prompt += """answer with valid json: {'class': <class>, 'reasoning': <reasoning>}"""

    prompt = 'say hello three times'

    print(model(prompt=prompt))
