import pandas as pd
import concurrent.futures

from src.ml.preprocessing.rest_embedding import HFEmbeddingPreprocessor
from src.util.load_hydra import get_hydra_config
from src.util.dynamic_import import DynamicImport
from src.io.data_import.base import BaseDataset
from src.util.constants import DatasetColumn as dfc
from src.util.logger import console


if __name__ == "__main__":

    datasets = [
        "clinc", "hwu", "banking"
    ]

    for dataset in datasets:

        config = get_hydra_config(overrides=[f"io__import={dataset}", "ml__preprocessing=rest_embedding"])

        dataloader = DynamicImport.init_class_from_dict(
            dictionary=config['io__import']
        )

        encoder = DynamicImport.init_class_from_dict(
            dictionary=config['ml__preprocessing']
        )

        assert isinstance(dataloader, BaseDataset)
        assert isinstance(encoder, HFEmbeddingPreprocessor)

        data_dict = dataloader.execute()

        assert "data" in data_dict
        assert "all_classes" in data_dict

        console.rule(f"Dataset: {dataset}")

        data = data_dict['data']
        assert isinstance(data, pd.DataFrame)

        texts = data[dfc.TEXT].tolist()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(encoder._api_call, text) for text in texts]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    console.print(e)
