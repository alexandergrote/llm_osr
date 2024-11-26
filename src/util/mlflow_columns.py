from typing import List
from pydantic import BaseModel

class MLFlowColumn(BaseModel):

    column_name: str
    yaml_keys: List[str]  # list of keys in yaml dict
    verbose_str: str


class ColumnMixin(BaseModel):

    def get_columns(self) -> List[MLFlowColumn]:
        return [el[1] for el in self]
    

class IDColumns(ColumnMixin, BaseModel):

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


class F1AnalysisColumns(ColumnMixin, BaseModel):

    f1_avg: MLFlowColumn = MLFlowColumn(
        column_name="metrics.f1_avg",
        yaml_keys=["metrics", "f1_avg"],
        verbose_str="f1_avg"
    )

class UnknownAUCAnalysisColumns(ColumnMixin, BaseModel):
    auc: MLFlowColumn = MLFlowColumn(
        column_name="metrics.unknown_scores_auc",
        yaml_keys=["metrics", "unknown_scores_auc"],
        verbose_str="unknown_scores_auc"
    )

    f1: MLFlowColumn = MLFlowColumn(
        column_name="metrics.f1_unknown_class_-1",
        yaml_keys=["metrics", "f1_unknown_class_-1"],
        verbose_str="unknown_scores_f1"
    )

id_columns = IDColumns()
f1_analysis_columns = F1AnalysisColumns()
unknown_auc_analysis_columns = UnknownAUCAnalysisColumns()


__all__ = [
    "id_columns", 
    "f1_analysis_columns",
    "unknown_auc_analysis_columns",
    "MLFlowColumn",
]