from typing import List
from pydantic import BaseModel

class MLFlowColumn(BaseModel):

    column_name: str
    yaml_keys: List[str]  # list of keys in yaml dict
    verbose_str: str
    

class IDColumns(BaseModel):

    experiment_name: MLFlowColumn = MLFlowColumn(
        column_name="params.io__export.params.experiment_name",
        yaml_keys=["io__export", "params", "experiment_name"],
        verbose_str="experiment_name"
    )

    perc_unknown_classes: MLFlowColumn = MLFlowColumn(
        column_name="params.ml__datasplit.params.percentage_unknown_classes",
        yaml_keys=["ml__datasplit", "params", "percentage_unknown_classes"],
        verbose_str="perc_unknown_classes"
    )

    dataset: MLFlowColumn = MLFlowColumn(
        column_name='params.io__import.class',
        yaml_keys=["io__import", "class"],
        verbose_str='dataset'
    )

    seed: MLFlowColumn = MLFlowColumn(
        column_name="params.random_seed",
        yaml_keys=["random_seed"],
        verbose_str="seed"
    )

    def get_columns(self) -> List[MLFlowColumn]:
        return [el[1] for el in self]


class F1AnalysisColumns(BaseModel):

    f1_avg: MLFlowColumn = MLFlowColumn(
        column_name="metrics.f1_avg",
        yaml_keys=["metrics", "f1_avg"],
        verbose_str="f1_avg"
    )

id_columns = IDColumns()
f1_analysis_columns = F1AnalysisColumns()


__all__ = [
    "id_columns", 
    "f1_analysis_columns"
]