import torch
from pydantic import BaseModel
from typing import List, Optional, Any, Union
from transformers import pipeline, Pipeline

from src.ml.classifier.util.llm_api.base import BaseRemoteLLM

class Llama(BaseRemoteLLM, BaseModel):

    _name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    _system_msg: str = "You are a helpful assistant"
    _device: str = "cuda"

    pipeline: Optional[Pipeline] = None
    terminators: List[Union[None, int]] = []

    cache: dict = dict()

    class Config:
        arbitrary_types_allowed=True

    def setup(self):

        self.pipeline = pipeline(
            "text-generation",
            model=self._name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=self._device,
        )

        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    
    def __call__(self, *, prompt: str, **kwargs):

        if self.pipeline is None:
            self.setup()

        if self.pipeline is None:
            raise ValueError("Pipeline is not initialized")
        
        if hasattr(self.pipeline, "tokenizer") is False:
            raise ValueError("Tokenizer is not initialized")
        

        messages = [
            {"role": "system", "content": self._system_msg},
            {"role": "user", "content": prompt},
        ]
        
        prompt = self.pipeline.tokenizer.apply_chat_template(  
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        if prompt in self.cache:
            return self.cache[prompt]
        
        outputs = self.pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.001,
            top_p=0.9        
        )

        answer = outputs[0]["generated_text"][len(prompt):]

        try:

            # extract json_str from the answer
            # Extract the JSON part from the text
            start_index = answer.find('{')
            end_index = answer.rfind('}') + 1
            answer = answer[start_index:end_index]

        except Exception as e:
            print(e)

        self.cache[prompt] = answer

        return answer

    def batch(self, prompts: List[str], **kwargs: Any) -> List[str]:
        
        if self.pipeline is None:
            self.setup()

        if self.pipeline is None:
            raise ValueError("Pipeline is not initialized")
        
        if hasattr(self.pipeline, "tokenizer") is False:
            raise ValueError("Tokenizer is not initialized")

        messages_list = [
            [
                {"role": "system", "content": self._system_msg},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]

        tokenized_prompts = [
            self.pipeline.tokenizer.apply_chat_template(  
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            for messages in messages_list
        ]

        # check for cached responses
        cached_responses = [self.cache.get(prompt) for prompt in tokenized_prompts]

        # filter out cached responses
        uncached_prompts = [
            prompt for prompt, response in zip(tokenized_prompts, cached_responses) if response is None
        ]

        if len(uncached_prompts) > 0:

            # execute the model on the uncached prompts
            outputs = self.pipeline(  
                uncached_prompts,
                max_new_tokens=256,
                eos_token_id=self.terminators,
                do_sample=True,
                temperature=0.001,
                top_p=0.9        
            )

            # Extract and return answers
            answers = [
                output[0]["generated_text"][len(prompt):]
                for output, prompt in zip(outputs, uncached_prompts)
            ]

            # cache the responses
            for prompt, answer in zip(uncached_prompts, answers):
                self.cache[prompt] = answer

        # combine the cached and uncached responses
        all_answers = [
            response if response is not None else self.cache[prompt]
            for prompt, response in zip(tokenized_prompts, cached_responses)
        ]

        return all_answers