import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pydantic import BaseModel
from collections import defaultdict, Counter

from src.util.types import MLPrediction
from src.util.mlflow_columns import id_columns, f1_analysis_columns, unknown_auc_analysis_columns
from src.util.constants import Directory
from src.experiments.util.naming_conventions import get_model_name_from_exp_name
from src.experiments.util.artifacts import get_artifacts
from src.experiments.analysis.base import BaseAnalyser
from src.experiments.visualization.heatmap import PearsonHeatmap, JaccardHeatmap, McNemarHeatmap
from src.experiments.visualization.f1_table import F1ScoreTable


class ErrorAnalyser(BaseModel, BaseAnalyser):

    def analyse(self, data: pd.DataFrame, **kwargs):

        # work on copy
        data_copy = data.copy(deep=True)

        data_exp_name = data_copy.filter(like="experiment")

        assert data_exp_name.shape[1] == 1, "There should only be one experiment column."
        exp_col_name = data_exp_name.columns[0]

        data_copy = data_copy.dropna(subset=[exp_col_name])

        predictions, _, _, data_test_list = get_artifacts(data_copy=data_copy)

        assert len(predictions) == len(data_copy), 'The number of predictions does not match the number of rows in the dataset.'
        assert any(isinstance(prediction, MLPrediction) for prediction in predictions), 'All predictions must be of type MLPrediction.'

        # calculate errors
        errors_list = [prediction.error().values for prediction in predictions]
        raw_text = [mldf.raw_text() for mldf in data_test_list]

        metric_col, dataset_col, perc_unknown_col = f1_analysis_columns.f1_avg.column_name, id_columns.dataset.column_name, id_columns.perc_unknown_classes.column_name
        unknown_f1_col = unknown_auc_analysis_columns.f1.column_name
        exp_name_col = id_columns.experiment_name.column_name
        random_seed_col = id_columns.seed.column_name

        all_columns = [metric_col, perc_unknown_col, dataset_col]

        for col in all_columns:
            assert col in data_copy.columns, f"'{col}' must be present in the analysis DataFrame."

        model_col = 'Model'

        data_copy[model_col] = data_copy[exp_name_col].apply(lambda x: get_model_name_from_exp_name(x))
    
        
        column_mapping = {
            'hyper_simpleshot': 'SimpleShot',
            'hyper_fastfit': 'FastFit',
            'hyper_contrastnet': 'ContrastNet',
            'one_stage_llama_8__scenario__implicit_zeroshot': 'LLaMA 8',
            'one_stage_llama_70__scenario__implicit_zeroshot': 'LLaMA 70',
            'one_stage_gemma3_27__scenario__implicit_zeroshot': 'GEMMA3 27',
            'one_stage_phi4_14__scenario__implicit_zeroshot': 'Phi4 14'
        }

        data_copy.replace(column_mapping, inplace=True)

        model_names = data_copy[model_col].to_list()
        unknown_scores = data_copy[perc_unknown_col].to_list()
        random_seeds = data_copy[random_seed_col].to_list()
        dataset_names = data_copy[dataset_col].to_list()

        error_dict = defaultdict(lambda: defaultdict(list))  # type: ignore

        # provides overview over all datasets and degrees of openness and random seeds
        # hence, multiple predictions for the same datapoint can occur
        for dataset, model, seed, openness, errors, texts in zip(dataset_names, model_names, random_seeds, unknown_scores, errors_list, raw_text):

            assert len(errors) == len(texts), f"Error and text lists must have the same length for model {model}."

            for error, text in zip(errors, texts):  # handle length mismatches safely

                idx = f"{openness}_{seed}_{text}"
                        
                error_dict[idx][model].append(error)

                if len(error_dict[idx][model]) > 1:
                    #print(f"Multiple predictions: {dataset.split('.')[1]}\t{openness}\t{seed}\t{model}\t{text}")
                    pass

        for text in error_dict.keys():
            for model in error_dict[text].keys():
                
                # get majority vote on model predictions for each text
                counter = Counter(error_dict[text][model])
                most_common_response = counter.most_common(1)[0][0]
                
                error_dict[text][model] = most_common_response

            

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
            x=model_col, y=metric_col, data=data_copy,
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(Directory.OUTPUT_DIR / 'f1_scores.pdf')
        plt.close()

        # analyse f1 scores with boxplot
        plt.figure()
        sns.barplot(
            x=model_col, y=unknown_f1_col, data=data_copy,
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(Directory.OUTPUT_DIR / 'f1_scores_unknown.pdf')
        plt.close()
        
        _, recall_col, precision_col = unknown_auc_analysis_columns.f1.column_name, unknown_auc_analysis_columns.recall.column_name, unknown_auc_analysis_columns.precision.column_name
        
        # precision
        plt.figure()
        sns.barplot(
            x=model_col, y=precision_col, data=data_copy,
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(Directory.OUTPUT_DIR / 'precision_scores_unknown.pdf')
        plt.close()

        # recall
        plt.figure()
        sns.barplot(
            x=model_col, y=recall_col, data=data_copy,
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(Directory.OUTPUT_DIR / 'recall_scores_unknown.pdf')
        plt.close()

        