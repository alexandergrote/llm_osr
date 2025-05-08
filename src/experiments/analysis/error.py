import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pydantic import BaseModel
from collections import defaultdict

from src.util.types import MLPrediction
from src.util.mlflow_columns import id_columns, f1_analysis_columns, unknown_auc_analysis_columns
from src.util.constants import Directory
from src.experiments.util.naming_conventions import get_model_name_from_exp_name
from src.experiments.util.artifacts import get_artifacts
from src.experiments.analysis.base import BaseAnalyser
from src.experiments.visualization.heatmap import PearsonHeatmap, JaccardHeatmap, McNemarHeatmap


class ErrorAnalyser(BaseModel, BaseAnalyser):

    def analyse(self, data: pd.DataFrame, **kwargs):

        # work on copy
        data_copy = data.copy(deep=True)

        #artifact_sanity_check(data_copy=data_copy, dataset_col=id_columns.dataset.column_name)

        predictions, _, _, data_test_list = get_artifacts(data_copy=data_copy)

        assert len(predictions) == len(data_copy), 'The number of predictions does not match the number of rows in the dataset.'
        assert any(isinstance(prediction, MLPrediction) for prediction in predictions), 'All predictions must be of type MLPrediction.'

        # calculate errors
        errors_list = [prediction.error().values for prediction in predictions]
        raw_text = [mldf.raw_text() for mldf in data_test_list]

        metric_col, dataset_col, perc_unknown_col = f1_analysis_columns.f1_avg.column_name, id_columns.dataset.column_name, id_columns.perc_unknown_classes.column_name
        unknown_f1_col = unknown_auc_analysis_columns.f1.column_name
        exp_name_col = id_columns.experiment_name.column_name

        all_columns = [metric_col, perc_unknown_col, dataset_col]

        for col in all_columns:
            assert col in data_copy.columns, f"'{col}' must be present in the analysis DataFrame."

        data_copy['model_col'] = data_copy[exp_name_col].apply(lambda x: get_model_name_from_exp_name(x))
        model_names = data_copy['model_col'].to_list()


        error_dict = defaultdict(dict)  # type: ignore

        for model, errors, texts in zip(model_names, errors_list, raw_text):
            for i in range(min(len(errors), len(texts))):  # handle length mismatches safely
                text = texts[i]
                error = errors[i]
                error_dict[text][model] = error

        # Now create the DataFrame
        df = pd.DataFrame.from_dict(error_dict, orient='index')
        df.index.name = 'raw_text'
        df.reset_index(inplace=True)
        
        named_errors = df.dropna(axis=0).drop(columns=["raw_text"]).T.apply(list, axis=1).to_dict()

        ## pearson
        pearson = PearsonHeatmap(
            data=named_errors,
            title="Pearson Correlation Matrix",
            filename="pearson_correlation.pdf"
        )

        pearson.plot()

        ## jaccard
        jaccard = JaccardHeatmap(
            data=named_errors,
            title="Jaccard Similarity Matrix",
            filename="jaccard_similarity.pdf"
        )

        jaccard.plot()

        ## mcnemar
        mcnemar_heatmap = McNemarHeatmap(
            data=named_errors,
            title="McNemar's Test Matrix",
            filename="mcnemar_test.pdf"
        )

        mcnemar_heatmap.plot()

        data_copy.reset_index(drop=True, inplace=True)
    
        # analyse f1 scores with boxplot
        plt.figure()
        sns.barplot(
            x='model_col', y=metric_col, data=data_copy,
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(Directory.OUTPUT_DIR / 'f1_scores.pdf')
        plt.close()

        # analyse f1 scores with boxplot
        plt.figure()
        sns.barplot(
            x='model_col', y=unknown_f1_col, data=data_copy,
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(Directory.OUTPUT_DIR / 'f1_scores_unknown.pdf')
        plt.close()
        
        _, recall_col, precision_col = unknown_auc_analysis_columns.f1.column_name, unknown_auc_analysis_columns.recall.column_name, unknown_auc_analysis_columns.precision.column_name
        
        # precision
        plt.figure()
        sns.barplot(
            x='model_col', y=precision_col, data=data_copy,
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(Directory.OUTPUT_DIR / 'precision_scores_unknown.pdf')
        plt.close()

        # recall
        plt.figure()
        sns.barplot(
            x='model_col', y=recall_col, data=data_copy,
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(Directory.OUTPUT_DIR / 'recall_scores_unknown.pdf')
        plt.close()