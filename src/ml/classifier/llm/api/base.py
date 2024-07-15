from abc import abstractmethod, ABC
from typing import List, Tuple

from src.util.types import LogProb


class AbstractLLM(ABC):
    
    @abstractmethod
    def __call__(self, *, prompt: str, **kwargs) -> Tuple[str, List[LogProb]]:

        raise NotImplementedError()

