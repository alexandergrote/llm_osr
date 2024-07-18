import typer
import pandas as pd
from pydantic import BaseModel

from src.experiments import Experiment
from src.util.constants import Directory
from src.io.data_import.mlflow_engine import QueryEngine
from src.experiments.analysis.base import BaseAnalyser
from src.util.logging import console

experiments = Experiment.create_experiments_from_yaml(
        path=Directory.CONFIG / "experiments" / "fewshot.yaml"
    )

class FewShotAnalyser(BaseModel, BaseAnalyser):

    def analyse(self, data: pd.DataFrame, **kwargs):

        # work on copy
        data_copy = data.copy(deep=True)

        metric_col = "metrics.f1_avg"
        dataset_col = "params.io__import.class"
        perc_unknown_col = "params.ml__datasplit.params.percentage_unknown_classes"

        all_columns = [metric_col, perc_unknown_col, dataset_col]

        for col in all_columns:
            assert col in data_copy.columns, f"'{col}' must be present in the analysis DataFrame."

        data_copy_grouped = data_copy.groupby([dataset_col, perc_unknown_col])[metric_col].mean().reset_index()

        console.log(data_copy_grouped)
    

def main():

    mlflow_engine = QueryEngine()

    analyser = FewShotAnalyser()   

    for experiment in experiments:

        data = mlflow_engine.get_results_of_single_experiment(experiment_name=experiment.name, n=100)
        
        analyser.analyse(data=data)


if __name__ == '__main__':
    typer.run(main)