import pandas as pd
from pydantic import BaseModel, ConfigDict

from src.util.logger import console
from src.experiments.util.naming_conventions import get_model_name_from_exp_name
from src.util.mlflow_columns import id_columns, f1_analysis_columns, unknown_auc_analysis_columns
from src.experiments.analysis.base import BaseAnalyser


class FewShotAnalyser(BaseModel, BaseAnalyser):

    def analyse(self, data: pd.DataFrame, **kwargs):

        # work on copy
        data_copy = data.copy(deep=True)

        metric_col, dataset_col, perc_unknown_col = f1_analysis_columns.f1_avg.column_name, id_columns.dataset.column_name, id_columns.perc_unknown_classes.column_name
        unknown_f1_col = unknown_auc_analysis_columns.f1.column_name
        exp_name_col = id_columns.experiment_name.column_name

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
        
        import pandera as pa
        from pandera import Field
        from pandera.typing import DataFrame
        from typing import Optional
        import pandas as pd


        class MetricData(pa.DataFrameModel):
            dataset: str = Field()
            Openness: float = Field(coerce=True)
            Model: str = Field()
            mean: float = Field(ge=0, le=1)
            std: Optional[float] = Field(nullable=True)

            @pa.check(name="unique_except_mean_std", element_wise=False)
            def unique_except_mean_std(cls, df: pd.DataFrame) -> bool:
                # Check uniqueness of dataset + Openness + Model
                return df[["dataset", "Openness", "Model"]].drop_duplicates().shape[0] == df.shape[0]


        class F1ScoreTable(BaseModel):
            data: DataFrame[MetricData]

            model_config = ConfigDict(arbitrary_types_allowed=True)

            def print(self, **kwargs) -> None:
                data_copy = self.data.copy()
                mean_str = data_copy[MetricData.mean].round(4).astype(str)
                std_str = data_copy[MetricData.std].round(4).astype(str)
                data_copy['metric'] = mean_str + " ± " + std_str
                data_copy_pivot = data_copy.pivot_table(index=[MetricData.Openness, MetricData.Model], columns=[id_columns.dataset.verbose_str], values='metric', aggfunc="first")
                print(data_copy_pivot.to_latex())
        
                

        table = F1ScoreTable(data=data_copy.groupby([id_columns.dataset.verbose_str, 'Openness', model_col])[metric_col].agg(['mean', 'std']).reset_index())

        table.print()

        # Example usage:
        # df = pd.read_csv("your_data.csv")
        # validated_df: DataFrame[BankingModel] = BankingModel.validate(df)


class BenchmarkAnalyser(FewShotAnalyser):
    pass

class LLMAnalyser(FewShotAnalyser):
    pass