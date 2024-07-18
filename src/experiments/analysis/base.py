import pandas as pd
from abc import abstractmethod, ABC


class BaseAnalyser(ABC):

    @abstractmethod
    def analyse(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError()