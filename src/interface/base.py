from abc import ABC, abstractmethod


class BaseExecutionBlock(ABC):

    @abstractmethod
    def execute(self, **kwargs) -> dict:
        raise NotImplementedError()