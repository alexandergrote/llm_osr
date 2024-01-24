from abc import ABC, abstractmethod


class BaseExecutionBlock(ABC):

    @abstractmethod
    def execute(self, *args, **kwargs) -> dict:
        raise NotImplementedError()