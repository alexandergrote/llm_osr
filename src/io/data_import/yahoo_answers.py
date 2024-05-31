from datasets import load_dataset
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, model_validator

from src.util.constants import DatasetColumn, Directory
#https://www.kaggle.com/datasets/soumikrakshit/yahoo-answers-dataset?resource=download

# custom code
from .base import BaseDataset

class YahooAnswersDataset(BaseDataset, BaseModel):
    data_home: Path = Directory.INPUT_DIR / "yahoo"
    
    @model_validator(mode='after')
    def init_data_dir(self):
        self.data_home.mkdir(parents=True, exist_ok=True)
    
    
    def _load(self, **kwargs) -> pd.DataFrame:
        # fetch data
        # Load both train and test data and concatenate them
        
        filename = self.data_home / "yahoo_answers.parquet"

        if filename.exists():
            data = pd.read_parquet(filename)
            return data
        
        dataset = load_dataset("yahoo_answers_topics")

        train_data = dataset["train"].to_pandas()
        test_data = dataset["test"].to_pandas()

        # map integer labels to string labels
        # train data
        mapping = {idx: value for idx, value in enumerate(dataset['train'].features['topic'].names)}
        train_data['topic'] = train_data['topic'].map(mapping)

        # test data
        mapping = {idx: value for idx, value in enumerate(dataset['test'].features['topic'].names)}
        test_data['topic'] = test_data['topic'].map(mapping)

        data = pd.concat([train_data, test_data], ignore_index=True)

        # dataframe columns should be named text and label
        data = pd.DataFrame({
            DatasetColumn.TEXT: data['question_title'] + ' ' + data['question_content'], 
            DatasetColumn.LABEL: data['topic']
        })

        assert len(data[DatasetColumn.LABEL].unique()) == 10

        data.to_parquet(filename)
  
        return data