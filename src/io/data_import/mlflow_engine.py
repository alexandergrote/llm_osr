import pandas as pd

from pydantic import BaseModel
from omegaconf import DictConfig
from typing import List
from mlflow.entities import ViewType, Run
from mlflow.store.entities.paged_list import PagedList

from src.util.dict_extraction import get_nested_dict_values
from src.util.mlflow_util import get_mlflow_client
from src.util.mlflow_columns import id_columns
from src.io.data_import.base import BaseResultDataset


client = get_mlflow_client()


class QueryEngine(BaseResultDataset, BaseModel):

    @staticmethod
    def _get_last_n_runs(experiment_id: str, n: int, query: str):

        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=query,
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["attribute.start_time DESC"],
            max_results=n,
        )

        return runs

    @staticmethod
    def _assign_prefix_to_columns(df: pd.DataFrame, prefix: str):

        # work on copy
        df_copy = df.copy(deep=True)

        df_copy.columns = [f"{prefix}.{col}" for col in df_copy.columns]

        return df_copy

    @staticmethod
    def _runs_to_df(runs):

        metrics = [pd.DataFrame(run.data.metrics, index=[0]) for run in runs]
        params = [pd.DataFrame(run.data.params, index=[0]) for run in runs]
        meta = [
            pd.DataFrame(
                {
                    "end_time": run.info.end_time,
                    "run_id": run.info.run_id,
                    "artifact_uri": run.info.artifact_uri,
                },
                index=[0],
            )
            for run in runs
        ]

        metrics_df = pd.concat(metrics, ignore_index=True)
        params_df = pd.concat(params, ignore_index=True)
        meta_df = pd.concat(meta, ignore_index=True)

        dataframes = [
            QueryEngine._assign_prefix_to_columns(df, prefix) 
            for df, prefix in zip([metrics_df, params_df, meta_df], ["metrics", "params", "meta"])
        ]

        return pd.concat(dataframes, axis=1)
    
    @staticmethod
    def get_results_of_single_experiment(experiment_name: str, n: int, filter_str: str = "") -> pd.DataFrame:

        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id: str = experiment.experiment_id

        runs: PagedList[Run] = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_str,
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["attribute.start_time DESC"],
            max_results=n,
        )

        data = QueryEngine._runs_to_df(runs)

        return data
    
    @staticmethod
    def get_results_of_multiple_experiments(experiment_names: list, n: int, filter_str: str = "") -> pd.DataFrame:

        data = pd.concat(
            [
                QueryEngine.get_results_of_single_experiment(experiment_name, n, filter_str)
                for experiment_name in experiment_names
            ],
            ignore_index=True,
        )

        return data

    def _load(self, **kwargs) -> pd.DataFrame:

        experiment_name = kwargs["experiment_name"]
        n = kwargs.get("n", 10)

        data = self.get_results_of_single_experiment(experiment_name, n)

        return data
    
    @staticmethod
    def get_experiment_names() -> List[str]:

        experiments = client.search_experiments()

        return [str(experiment.name) for experiment in experiments]

    
    @staticmethod
    def check_if_experiment_run_exists(config: DictConfig) -> bool:
        
        experiment_name = get_nested_dict_values(list_of_keys=[id_columns.experiment_name.yaml_keys], dictionary=config)[0]

        # check if name has been initialized in database        
        experiment_names = QueryEngine.get_experiment_names()

        if experiment_name not in experiment_names:
            return False

        # check if required run exists
        data = QueryEngine.get_results_of_single_experiment(experiment_name, n=999)

        if len(data) == 0:
            return False
        
        mlflow_columns = id_columns.get_columns()

        for mlflow_column in mlflow_columns:

            if mlflow_column.column_name not in data.columns:
                raise ValueError(f"Column {mlflow_column.column_name} must be present in experiment.")
            
            unique_values = [str(v) for v in data[mlflow_column.column_name].unique()]
            value = get_nested_dict_values(list_of_keys=[mlflow_column.yaml_keys], dictionary=config)[0]

            if str(value) not in unique_values:
                return False

        return True
    

if __name__ == '__main__':

    mlflow_engine = QueryEngine()

    # get results of a single experiment
    data = mlflow_engine.get_results_of_single_experiment(experiment_name="fewshot_naive", n=5)
    
    print(data)
