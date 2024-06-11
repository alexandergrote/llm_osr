from pydantic import BaseModel
from langchain_openai import OpenAI

from src.ml.classifier.llm.api.base import BaseRemoteLLM

class OpenAIWrapper(BaseRemoteLLM, BaseModel):

    name: str
    
    def __call__(self, *, prompt: str, **kwargs):

        stop = kwargs.get("stop")
        
        model = OpenAI(name=self.name, temperature=0)

        answer = model(prompt=prompt, stop=stop)

        return answer