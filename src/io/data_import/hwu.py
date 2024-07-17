import requests  # type: ignore

from pathlib import Path
from typing import Optional, List

import pandas as pd
from pydantic import BaseModel, model_validator

from src.util.constants import DatasetColumn, Directory

from src.io.data_import.base import BaseDataset

# https://huggingface.co/datasets/banking77/tree/main


class HWUDataset(BaseDataset, BaseModel):

    data_home: Path = Directory.INPUT_DIR / "HWU64"
    filter: Optional[List[str]] = None
    
    @model_validator(mode='after')
    def init_data_dir(self):
        self.data_home.mkdir(parents=True, exist_ok=True)
    

    def _load(self, **kwargs) -> pd.DataFrame:

        # fetch data

        filename = self.data_home / "hwu64.parquet"

        if filename.exists():
            data = pd.read_parquet(filename)
            return data

        # Modified URL to point to the raw content
        base_url = 'https://raw.githubusercontent.com/jianguoz/Few-Shot-Intent-Detection/main/Datasets/HWU64/'
        folders = ['train', 'test', 'valid']
        url_filenames = ['seq.in', 'label']
        varnames = [DatasetColumn.TEXT, DatasetColumn.LABEL]

        tmp: dict = {DatasetColumn.TEXT: [], DatasetColumn.LABEL: []}

        for folder in folders:

            for varname, url_filename in zip(varnames, url_filenames):

                url = base_url + folder + '/' + url_filename

                response = requests.get(url)

                # Check if the request was successful
                if response.status_code == 200:
                    content_as_str = response.content.decode('utf-8')
                    tmp[varname].extend(content_as_str.split('\n'))
                else:
                    raise Exception(f"Failed to retrieve data from {url}")
                
            data = pd.DataFrame(tmp)
        

        if self.filter is not None:

            mask = data[DatasetColumn.LABEL].isin(self.filter)

            data = data[mask]

        # random seed
        random_seed = kwargs.get('random_seed', 42)

        if not isinstance(random_seed, int):
            raise ValueError("random_seed must be an integer")

        # shuffle data
        data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)    
        
        # filter to only desired categories
        data.to_parquet(filename)

        return data
    

if __name__ == '__main__':

    
    dataset = HWUDataset()
    data = dataset._load()

    print(data)
    