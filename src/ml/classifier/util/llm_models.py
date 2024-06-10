from requests import ConnectionError  # type: ignore

from typing import Dict, Type, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from huggingface_hub import login
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

from src.util.constants import LLMModels
from src.util.logging import console
from src.util.environment import PydanticEnvironment
from src.ml.classifier.util.llm_api import Llama, OpenAIWrapper


try:

    env = PydanticEnvironment()

    login(
        token=env.hf_token
    )
except ConnectionError:
    console.log("Could not connect to HF due to missing Internet connection")

def get_hf_auto_model(model_id: str) -> Any:

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer
    )
    model = HuggingFacePipeline(pipeline=pipe)

    return model


class LangchainWrapper(LLM):

    name: str
    custom_model: Any

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

       return self.custom_model(prompt=prompt, stop=stop, run_manager=run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return self.custom_model.name


LLM_Mapping: Dict[LLMModels, Type[LLM]] = {
    LLMModels.OAI_GPT4: LangchainWrapper(name='gpt-3.5-turbo-0125', custom_model=OpenAIWrapper()),
    LLMModels.OAI_GPT3: LangchainWrapper(name='gpt-3.5-turbo-0125', custom_model=OpenAIWrapper()),
    LLMModels.OAI_GPT2: LangchainWrapper(name='gpt2', custom_model=get_hf_auto_model('gpt2')),
    LLMModels.LLAMA_3B: LangchainWrapper(name='llama-3b', custom_model=Llama()),
}


__all__ = ['LLM_Mapping']
