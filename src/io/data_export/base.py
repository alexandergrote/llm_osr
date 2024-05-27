from abc import ABC, abstractmethod


class BaseExporter(ABC):

    @abstractmethod
    def export(self) -> dict:
        raise NotImplementedError("Not Implemented Yet")

    def execute(self, **kwargs) -> dict:
        
        self.export(**kwargs)

        return kwargs