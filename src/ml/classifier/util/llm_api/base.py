from abc import abstractmethod, ABC


class BaseRemoteLLM(ABC):
    
    @abstractmethod
    def __call__(self, *, prompt: str, **kwargs) -> str:

        raise NotImplementedError()
