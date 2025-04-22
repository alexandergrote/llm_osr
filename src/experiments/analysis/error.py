import pandas as pd

from pydantic import BaseModel
from sklearn.metrics import jaccard_score

from src.util.logger import console
from src.util.types import MLPrediction
from src.util.mlflow_columns import id_columns, f1_analysis_columns, unknown_auc_analysis_columns, artifact_columns
from src.experiments.util.artifacts import get_artifacts, artifact_sanity_check
from src.experiments.analysis.base import BaseAnalyser


class ErrorAnalyser(BaseModel, BaseAnalyser):

    def analyse(self, data: pd.DataFrame, **kwargs):

        # work on copy
        data_copy = data.copy(deep=True)

        artifact_sanity_check(data_copy=data_copy, dataset_col=id_columns.dataset.column_name)

        predictions, _, _, _ = get_artifacts(data_copy=data_copy)

        assert len(predictions) == len(data_copy), 'The number of predictions does not match the number of rows in the dataset.'
        assert any(isinstance(prediction, MLPrediction) for prediction in predictions), 'All predictions must be of type MLPrediction.'

        # calculate errors
        errors = [prediction.error().values for prediction in predictions]

        metric_col, dataset_col, perc_unknown_col = f1_analysis_columns.f1_avg.column_name, id_columns.dataset.column_name, id_columns.perc_unknown_classes.column_name
        exp_name_col = id_columns.experiment_name.column_name

        all_columns = [metric_col, perc_unknown_col, dataset_col]

        for col in all_columns:
            assert col in data_copy.columns, f"'{col}' must be present in the analysis DataFrame."

        named_errors = dict(zip(data_copy[exp_name_col].to_list(), errors))

        ## pearson
        error_df = pd.DataFrame(named_errors)
        corr_matrix = error_df.corr(method="pearson")

        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Error Correlation Between Models")
        plt.show()

        ## jaccard
        from sklearn.metrics import jaccard_score
        import numpy as np

        # Assume binary error vectors: 1 = incorrect, 0 = correct
        model_names = list(named_errors.keys())
        n = len(model_names)

        # Initialize empty matrix
        jaccard_matrix = np.zeros((n, n))

        # Fill matrix with pairwise Jaccard scores
        for i in range(n):
            for j in range(n):
                jaccard_matrix[i, j] = jaccard_score(named_errors[model_names[i]], named_errors[model_names[j]])

        # Convert to pandas DataFrame for readability
        jaccard_df = pd.DataFrame(jaccard_matrix, index=model_names, columns=model_names)
        sns.heatmap(jaccard_df, annot=True, cmap='YlGnBu')
        plt.title("Jaccard Similarity of Prediction Errors")
        plt.show()

        ## mcnemar
        from statsmodels.stats.contingency_tables import mcnemar
        import numpy as np

        model_names = list(named_errors.keys())
        n = len(model_names)
        mcnemar_matrix = np.ones((n, n))  # Start with 1s for diagonal

        for i in range(n):
            for j in range(n):
                if i != j:
                    a_errors = named_errors[model_names[i]]
                    b_errors = named_errors[model_names[j]]

                    # Contingency table components
                    both_correct = np.sum((a_errors == 0) & (b_errors == 0))
                    a_correct_b_wrong = np.sum((a_errors == 0) & (b_errors == 1))
                    a_wrong_b_correct = np.sum((a_errors == 1) & (b_errors == 0))
                    both_wrong = np.sum((a_errors == 1) & (b_errors == 1))

                    table = [[both_correct, a_correct_b_wrong],
                            [a_wrong_b_correct, both_wrong]]

                    try:
                        result = mcnemar(table, exact=True, correction=True)
                        mcnemar_matrix[i, j] = result.pvalue
                    except:
                        mcnemar_matrix[i, j] = np.nan  # In case counts are too small
 
        mcnemar_df = pd.DataFrame(mcnemar_matrix, index=model_names, columns=model_names)
        sns.heatmap(mcnemar_df, annot=True, cmap='coolwarm', vmin=0, vmax=1)
        plt.title("McNemar Test P-Values (Error Disagreement Symmetry)")
        plt.show()

