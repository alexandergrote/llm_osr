from pydantic import BaseModel
from langchain_openai import OpenAI

from src.ml.classifier.llm.api.base import BaseRemoteLLM
from src.ml.classifier.llm.util.backoff import BackoffMixin

class OpenAIWrapper(BaseRemoteLLM, BackoffMixin, BaseModel):

    name: str
    
    def __call__(self, *, prompt: str, **kwargs):

        stop = kwargs.get("stop")
        
        model = OpenAI(name=self.name, temperature=0)

        kwargs = {
            "prompt": prompt,
            "stop": stop
        }

        answer = self.completion_with_backoff(model, **kwargs)

        return answer
    

if __name__ == '__main__':

    model = OpenAIWrapper(name='gpt-3.5-turbo-0125')

    prompt = """hi, who are you?"""

    print(model(prompt=prompt))
