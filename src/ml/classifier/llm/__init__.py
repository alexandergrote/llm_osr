from .fewshot import FewShotLLM as LLM
from .fewshot import OneStageLLM
from .naive import RandomLLM 
from .twostage import TwoStageLLM 

__all__ = [
    "LLM",
    "RandomLLM",
    "TwoStageLLM",
    "OneStageLLM"
]