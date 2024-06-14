from typing import Any, Optional
from pydantic import BaseModel, model_validator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

from src.ml.classifier.llm.api.base import BaseRemoteLLM
from src.ml.classifier.llm.util.hf_login import login_to_hf


class HFWrapper(BaseRemoteLLM, BaseModel):

    name: str
    model: Optional[Any] = None

    @model_validator(mode='before')
    def _init_model(data: dict):

        data['model'] = HFWrapper._get_hf_auto_model(data['name'])

        return data

    @staticmethod
    def _get_hf_auto_model(model_id: str, auth: bool = False) -> Any:

        if auth:
            login_to_hf()

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer
        )
        model = HuggingFacePipeline(pipeline=pipe)

        return model
    
    def __call__(self, *, prompt: str, **kwargs):

        if self.model is None:
            raise ValueError("Model is not initialized")

        return self.model(prompt)