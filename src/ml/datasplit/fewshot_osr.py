import pandas as pd

from src.ml.datasplit.osr import DataSplitter
from src.util.constants import DatasetColumn
from src.util.types import TripleDataFrameTuple, MLDataFrame

class FewShotDataSplitter(DataSplitter):

    n: int 
    replace: bool = False

    @staticmethod
    def fewshot_split(data: pd.DataFrame, n: int, random_seed: int, replace: bool = False) -> pd.DataFrame:
        """
        Select 'n' examples of each class and return as new dataframe
        """

        data_list = []

        classes = data[DatasetColumn.LABEL].unique()

        for label in classes:

            data_sub: pd.DataFrame = data[data[DatasetColumn.LABEL] == label]

            data_list.append(
                data_sub.sample(n, random_state=random_seed, replace=replace)
            )

        return pd.concat(data_list)

    def _split_data(self, data: pd.DataFrame, random_seed: int, **kwargs) -> TripleDataFrameTuple:
        
        train, valid, test = super()._split_data(
            data=data, random_seed=random_seed, **kwargs
        )

        train_df_sub = FewShotDataSplitter.fewshot_split(
            data=train.data, n=self.n, random_seed=random_seed, replace=self.replace
        )

        train_sub = MLDataFrame.from_raw_pandas_dataframe(train_df_sub)

        return train_sub, valid, test
