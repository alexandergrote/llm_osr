import pandas as pd

from pydantic import BaseModel
from typing import List, Optional
from mlflow.entities import ViewType, Run
from mlflow.store.entities.paged_list import PagedList

from src.util.mlflow_util import get_mlflow_client
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
    def _runs_to_df(runs) -> Optional[pd.DataFrame]:

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

        if len(metrics) == 0:
            return None

        metrics_df = pd.concat(metrics, ignore_index=True)
        params_df = pd.concat(params, ignore_index=True)
        meta_df = pd.concat(meta, ignore_index=True)

        dataframes = [
            QueryEngine._assign_prefix_to_columns(df, prefix) 
            for df, prefix in zip([metrics_df, params_df, meta_df], ["metrics", "params", "meta"])
        ]

        return pd.concat(dataframes, axis=1)
    
    @staticmethod
    def get_results_of_single_experiment(experiment_name: str, n: int, filter_str: str = "") -> Optional[pd.DataFrame]:

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

        if data is None:
            return None

        return data
    
    @staticmethod
    def get_results_of_multiple_experiments(experiment_names: list, n: int, filter_str: str = "") -> pd.DataFrame:

        experiment_data = [
            QueryEngine.get_results_of_single_experiment(experiment_name, n, filter_str)
            for experiment_name in experiment_names
        ]

        experiment_data_cleaned = [el for el in experiment_data if el is not None]

        if len(experiment_data_cleaned) == 0:
            return None

        data = pd.concat(experiment_data_cleaned, ignore_index=True)

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
    

if __name__ == '__main__':

    mlflow_engine = QueryEngine()

    # get results of a single experiment
    data = mlflow_engine.get_results_of_single_experiment(experiment_name="fewshot_naive", n=5)
    
    print(data)
