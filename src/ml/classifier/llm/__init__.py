from .fewshot import FewShotLLM as LLM
from .naive import RandomLLM as RandomLLM

__all__ = [
    "LLM",
    "RandomLLM",
]