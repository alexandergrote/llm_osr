from datasets import load_dataset
from pathlib import Path
from typing import Literal, Dict, List
import pandas as pd

from pydantic import BaseModel, model_validator

from src.util.constants import DatasetColumn, Directory

from .base import BaseDataset
# https://huggingface.co/datasets/clinc_oos/tree/main


class Clinc150Dataset(BaseDataset, BaseModel):

    integer_mapping: Dict[int, str]
    domain_mapping: Dict[str, List[str]]

    exclude_oos: bool = True
    
    data_home: Path = Directory.INPUT_DIR / "Clinic150"
    kind: Literal['imbalanced', 'small', 'plus'] = 'plus'
        
    @model_validator(mode='after')
    def init_data_dir(self):
        self.data_home.mkdir(parents=True, exist_ok=True)
    
    def _load(self, **kwargs) -> pd.DataFrame:
        
        # fetch data
        
        filename = self.data_home / f"clinc150_{self.kind}.parquet"

        if filename.exists():
            data = pd.read_parquet(filename)
            if self.exclude_oos:
                data = data[data[DatasetColumn.LABEL]!= "domain_unknown__subcategory_oos"]
            data.reset_index(drop=True, inplace=True)

            return data
        

        dataset = load_dataset("clinc_oos", self.kind)
        train_data = dataset["train"].to_pandas()
        test_data = dataset["test"].to_pandas()
        validation_data = dataset["validation"].to_pandas()
        
        data = pd.concat([train_data, test_data, validation_data], ignore_index=True)

        data = pd.DataFrame({
            DatasetColumn.TEXT: data['text']    , 
            DatasetColumn.LABEL: data['intent']
        })

        # rename labels
        reversed_domain_mapping = {
            item: key for key, values in self.domain_mapping.items() for item in values
        }

        new_integer_mapping = {key: f"domain_{reversed_domain_mapping.get(value, 'unknown')}__subcategory_{value}" for key, value in self.integer_mapping.items()}

        data[DatasetColumn.LABEL] = data[DatasetColumn.LABEL].map(new_integer_mapping)

        assert len(data[DatasetColumn.LABEL].unique()) == 151

        data.to_parquet(filename)

        if self.exclude_oos:
            data = data[data[DatasetColumn.LABEL]!= "domain_unknown__subcategory_oos"]
            data.reset_index(drop=True, inplace=True)

        return data