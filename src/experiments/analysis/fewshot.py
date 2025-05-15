import pandas as pd
from pydantic import BaseModel, ConfigDict

from src.util.logger import console
from src.experiments.util.naming_conventions import get_model_name_from_exp_name
from src.util.mlflow_columns import id_columns, f1_analysis_columns, unknown_auc_analysis_columns
from src.experiments.analysis.base import BaseAnalyser
from src.experiments.visualization.f1_table import F1ScoreTable
from src.experiments.visualization.regression_plot import RegressionPlot
from src.util.constants import Directory


class FewShotAnalyser(BaseModel, BaseAnalyser):

    def analyse(self, data: pd.DataFrame, **kwargs):

        # work on copy
        data_copy = data.copy(deep=True)

        metric_col, dataset_col, perc_unknown_col = f1_analysis_columns.f1_avg.column_name, id_columns.dataset.column_name, id_columns.perc_unknown_classes.column_name
        unknown_f1_col = unknown_auc_analysis_columns.f1.column_name
        exp_name_col = id_columns.experiment_name.column_name
        
        data_copy.dropna(subset=[exp_name_col], inplace=True)

        all_columns = [metric_col, perc_unknown_col, dataset_col, exp_name_col]

        for col in all_columns:
            assert col in data_copy.columns, f"'{col}' must be present in the analysis DataFrame."

        data_copy_grouped = data_copy.groupby([dataset_col, perc_unknown_col])[metric_col].agg(["mean", "std"]).reset_index()

        console.log(data_copy_grouped)

        # analyse unknown predictions
        data_copy_grouped = data_copy.groupby([dataset_col, perc_unknown_col])[unknown_f1_col].agg(["mean", "std"]).reset_index()

        console.log(data_copy_grouped)

        # add model column for more finegrained insights
        model_col = 'Model'
        data_copy[model_col] = data_copy[exp_name_col].apply(lambda x: get_model_name_from_exp_name(x))

        # mapping dict
        rename_dict = {
            'hyper_contrastnet': 'ContrastNet',
            'hyper_fastfit': 'FastFit',
            'hyper_simpleshot': 'SimpleShot',
            'one_stage_gemma3_27__scenario__implicit_zeroshot': 'Gemma 3 27B',
            'one_stage_llama_70__scenario__implicit_zeroshot': 'Llama 3.3 70B',
            'one_stage_llama_8__scenario__implicit_zeroshot': 'Llama 3.1 8B',
            'one_stage_phi4_14__scenario__implicit_zeroshot': 'Phi4 14B',
            'params.io__import.class': id_columns.dataset.verbose_str,
            'params.ml__datasplit.params.percentage_unknown_classes': 'Openness',
            'src.Clinc150Dataset': 'CLINC150',
            'src.BankingDataset': 'Banking77',
            'src.HWUDataset': 'HWU64',
        }

        # horizontally the datasets and the metrics
        # veritcally the models and degree of unknown classes
        data_copy.replace(rename_dict, inplace=True)
        data_copy.rename(columns=rename_dict, inplace=True)
        data_copy_pivot = data_copy.pivot_table(index=['Openness', model_col], columns=[id_columns.dataset.verbose_str], values=metric_col, aggfunc=['mean', 'std'])                

        table = F1ScoreTable(data=data_copy.groupby([id_columns.dataset.verbose_str, 'Openness', model_col])[metric_col].agg(['mean', 'std']).reset_index())

        table.print()
        
        # Create a dictionary of datasets for the regression plot
        datasets_dict = {}
        for dataset in data_copy[id_columns.dataset.verbose_str].unique():
            dataset_data = data_copy[data_copy[id_columns.dataset.verbose_str] == dataset]
            # Group by Openness and Model to get both mean and std
            plot_data = dataset_data.groupby(['Openness', model_col])[metric_col].agg(['mean', 'std']).reset_index()
            datasets_dict[dataset] = plot_data
        
        # Create and display a single regression plot with all datasets
        regression_plot = RegressionPlot(
            data=datasets_dict,
            x_column='Openness',
            y_column=metric_col,
            hue_column=model_col,
            title='Known F1 Score vs. Openness Degree',
            output_path=str(Directory.OUTPUT_DIR / 'regression_plot_known_all_datasets.png')
        )
        regression_plot.plot()

        # Create a dictionary of datasets for the regression plot
        datasets_dict = {}
        for dataset in data_copy[id_columns.dataset.verbose_str].unique():
            dataset_data = data_copy[data_copy[id_columns.dataset.verbose_str] == dataset]
            # Group by Openness and Model to get both mean and std
            plot_data = dataset_data.groupby(['Openness', model_col])[unknown_f1_col].agg(['mean', 'std']).reset_index()
            datasets_dict[dataset] = plot_data

        regression_plot = RegressionPlot(
            data=datasets_dict,
            x_column='Openness',
            y_column=unknown_f1_col,
            hue_column=model_col,
            title='Known F1 Score vs. Openness Degree',
            output_path=str(Directory.OUTPUT_DIR / 'regression_plot_unknown_all_datasets.png')
        )
        regression_plot.plot()

        # Example usage:
        # df = pd.read_csv("your_data.csv")
        # validated_df: DataFrame[BankingModel] = BankingModel.validate(df)


class BenchmarkAnalyser(FewShotAnalyser):
    pass

class LLMAnalyser(FewShotAnalyser):
    pass
