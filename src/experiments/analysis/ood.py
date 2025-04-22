import pandas as pd
import numpy as np
from pydantic import BaseModel
from pathlib import Path

from src.util.types import MLDataFrame, MLPrediction
from src.util.mlflow_columns import id_columns, unknown_auc_analysis_columns, prompt_columns
from src.experiments.analysis.base import BaseAnalyser
from src.experiments.visualization.spider import SpiderPlot, SpiderDatasetSchema
from src.experiments.util.artifacts import artifact_sanity_check


class OODAnalyser(BaseModel, BaseAnalyser):

    def analyse(self, data: pd.DataFrame, **kwargs):

        # work on copy
        data_copy = data.copy(deep=True)

        # sanity check
        artifact_sanity_check(data_copy=data_copy, dataset_col=id_columns.dataset.column_name)

        dataset_col, perc_unknown_col = id_columns.dataset.column_name, id_columns.perc_unknown_classes.column_name
        unknown_prompt_col, unknown_model_col = prompt_columns.unknown_prompt.column_name, prompt_columns.unknown_prompt_model_name.column_name
        f1_col, recall_col, precision_col = unknown_auc_analysis_columns.f1.column_name, unknown_auc_analysis_columns.recall.column_name, unknown_auc_analysis_columns.precision.column_name
        
        all_metrics = [f1_col, recall_col, precision_col]
        all_columns = [perc_unknown_col, dataset_col, unknown_prompt_col, unknown_model_col] + all_metrics

        for col in all_columns:
            assert col in data_copy.columns, f"'{col}' must be present in the analysis DataFrame."

        # analyse unknown predictions
        data_copy_grouped = data_copy.groupby([dataset_col, perc_unknown_col])[all_metrics].agg(["mean", "std"]).reset_index()

        print(data_copy_grouped)

        column_mapping = {
            dataset_col: SpiderDatasetSchema.dataset,
            unknown_prompt_col: SpiderDatasetSchema.prompt_version,
            unknown_model_col: SpiderDatasetSchema.model,
            f1_col: SpiderDatasetSchema.F1,
            recall_col: SpiderDatasetSchema.recall,
            precision_col: SpiderDatasetSchema.precision,

        }


        data2plot = data_copy[all_columns].rename(columns=column_mapping)
        data2plot[SpiderDatasetSchema.dataset] = data2plot[SpiderDatasetSchema.dataset].str.replace("src.", "")
        data2plot[SpiderDatasetSchema.dataset] = data2plot[SpiderDatasetSchema.dataset].str.replace("Dataset", "")

        # visualize results
        spider_plot = SpiderPlot(
            data=data2plot
        )

        spider_plot.plot()
