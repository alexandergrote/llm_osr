import pandas as pd
import mlflow

from pydantic import BaseModel
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType, Run
from mlflow.store.entities.paged_list import PagedList

from src.io.data_import.base import BaseResultDataset
from src.util.constants import Directory


tracking_uri = Directory.ROOT / "mlruns"
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()


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
    

if __name__ == '__main__':

    mlflow_engine = QueryEngine()

    # get results of a single experiment
    data = mlflow_engine.get_results_of_single_experiment(experiment_name="fewshot_naive", n=5)
    
    print(data)
