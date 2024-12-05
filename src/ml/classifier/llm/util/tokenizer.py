from pydantic import BaseModel
from abc import abstractmethod
from typing import List
from transformers import LlamaTokenizerFast

class BaseTokenizer(BaseModel):

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

class LlamaTokenizer(BaseTokenizer):

    _tokenizer: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer", legacy=False)

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text) 
        