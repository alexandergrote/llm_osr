import pandas as pd

from typing import Optional

from src.ml.datasplit.osr import DataSplitter
from src.util.constants import DatasetColumn
from src.util.types import TripleDataFrameTuple, MLDataFrame

class FewShotDataSplitter(DataSplitter):

    n: int 
    replace: bool = False

    subset_test: Optional[int] = None

    @staticmethod
    def fewshot_split(data: pd.DataFrame, n: int, random_seed: int, replace: bool = False) -> pd.DataFrame:
        """
        Select 'n' examples of each class and return as new dataframe
        """

        data_list = []

        classes = data[DatasetColumn.LABEL].unique()

        for label in classes:

            data_sub: pd.DataFrame = data[data[DatasetColumn.LABEL] == label]

            if len(data_sub) <= n:
                drawn_samples = data_sub
            else:
                drawn_samples = data_sub.sample(n, random_state=random_seed, replace=replace)

            data_list.append(
                drawn_samples
            )

        return pd.concat(data_list)

    def _split_data(self, data: pd.DataFrame, random_seed: int, **kwargs) -> TripleDataFrameTuple:
        
        train, valid, test = super()._split_data(
            data=data, random_seed=random_seed, **kwargs
        )

        train_df = FewShotDataSplitter.fewshot_split(
            data=train.data, n=self.n, random_seed=random_seed, replace=self.replace
        )

        valid_df = FewShotDataSplitter.fewshot_split(
            data=valid.data, n=self.n, random_seed=random_seed, replace=self.replace
        )

        test_df = test.data

        # enable subset for development purposes
        if self.subset_test is not None:
            test_df = test_df.sample(self.subset_test, random_state=random_seed)

        train = MLDataFrame.from_raw_pandas_dataframe(train_df)
        valid = MLDataFrame.from_raw_pandas_dataframe(valid_df)
        test = MLDataFrame.from_raw_pandas_dataframe(test_df)

        return train, valid, test


if __name__ == '__main__':

    import os
    import yaml
    import numpy as np

    from src.util.constants import Directory
    from src.io.data_import.Clinc150 import Clinc150Dataset

    config_name = Directory.CONFIG / os.path.join("io__import", "banking.yaml")

    filepath = Directory.CONFIG / os.path.join('io__import', 'clinc.yaml')
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)['params']

    dataset = Clinc150Dataset(**config)

    filepath = Directory.CONFIG / os.path.join('ml__datasplit', 'fewshot_osr.yaml')
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)['params']

    splitter = FewShotDataSplitter(
        **config
    )

    data = dataset._load()

    df_train, df_valid, df_test = splitter._split_data(data=data, random_seed=0)    

    classes_train = np.unique(df_train.target())
    classes_valid = np.unique(df_valid.target())
    classes_test = np.unique(df_test.target())

    # get unknown classes
    unknown_classes = set(classes_test) - set(classes_train)
    print(f"Unknown classes: {unknown_classes}")
