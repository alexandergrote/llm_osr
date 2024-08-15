import pandas as pd
from pydantic import BaseModel

from src.util.logging import console
from src.util.mlflow_columns import id_columns, f1_analysis_columns
from src.experiments import Experiment
from src.util.constants import Directory
from src.experiments.analysis.base import BaseAnalyser

experiments = Experiment.create_experiments_from_yaml(
    path=Directory.CONFIG / "experiments" / "fewshot.yaml"
)

class FewShotAnalyser(BaseModel, BaseAnalyser):

    def analyse(self, data: pd.DataFrame, **kwargs):

        # work on copy
        data_copy = data.copy(deep=True)

        metric_col, dataset_col, perc_unknown_col = f1_analysis_columns.f1_avg.column_name, id_columns.dataset.column_name, id_columns.perc_unknown_classes.column_name

        all_columns = [metric_col, perc_unknown_col, dataset_col]

        for col in all_columns:
            assert col in data_copy.columns, f"'{col}' must be present in the analysis DataFrame."

        data_copy_grouped = data_copy.groupby([dataset_col, perc_unknown_col])[metric_col].mean().reset_index()

        console.log(data_copy_grouped)