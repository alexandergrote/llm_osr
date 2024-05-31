import torch
import transformers

from typing import Dict, Type, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_openai import OpenAI
from huggingface_hub import login
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

from src.util.constants import LLMModels
from src.util.environment import PydanticEnvironment

env = PydanticEnvironment()

login(
    token=env.hf_token
)

def get_hf_auto_model(model_id: str) -> Any:

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer
    )
    model = HuggingFacePipeline(pipeline=pipe)

    return model


class OAIWrapper(LLM):

    name: str

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        model = OpenAI(name=self.name, temperature=0, **kwargs)

        answer = model(prompt=prompt, stop=stop)

        return answer
    
    @property
    def _llm_type(self) -> str:
        return self.name

class LLamaWrapper(LLM):

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        

        pipeline = transformers.pipeline(
            "text-generation",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",
        )
    
        system_msg = "You are a helpful assistant"
        

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.001,
            top_p=0.9        
        )

        answer = outputs[0]["generated_text"][len(prompt):]

        return answer


    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "llama3b"


LLM_Mapping: Dict[LLMModels, Type[LLM]] = {
    LLMModels.OAI_GPT4: OpenAI(name='gpt-3.5-turbo-0125'),
    LLMModels.OAI_GPT3: OAIWrapper(name='gpt-3.5-turbo-0125'),
    LLMModels.OAI_GPT2: get_hf_auto_model('gpt2'),
    #LLMModels.LLAMA_3B: LLamaWrapper(),
}


__all__ = ['LLM_Mapping']