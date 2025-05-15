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

        # Define model groups
        traditional_ml_models = ['SimpleShot', 'FastFit', 'ContrastNet']
        llm_models = ['LLaMA 8', 'LLaMA 70', 'GEMMA3 27', 'Phi4 14']
        
        # Create separate error dictionaries for each group
        traditional_ml_errors = {model: named_errors[model] for model in traditional_ml_models if model in named_errors}
        llm_errors = {model: named_errors[model] for model in llm_models if model in named_errors}
        
        # Plot correlation matrices for all models
        ## pearson
        pearson = PearsonHeatmap(
            data=named_errors,
            title="Pearson Correlation Matrix - All Models",
            filename="pearson_correlation_all.pdf"
        )
        pearson.plot()

        ## jaccard
        jaccard = JaccardHeatmap(
            data=named_errors,
            title="Jaccard Similarity Matrix - All Models",
            filename="jaccard_similarity_all.pdf"
        )
        jaccard.plot()

        ## mcnemar
        mcnemar_heatmap = McNemarHeatmap(
            data=named_errors,
            title="McNemar's Test Matrix - All Models",
            filename="mcnemar_test_all.pdf"
        )
        mcnemar_heatmap.plot()
        
        # Plot correlation matrices for traditional ML models
        if len(traditional_ml_errors) > 1:
            pearson_trad = PearsonHeatmap(
                data=traditional_ml_errors,
                title="Pearson Correlation Matrix - Traditional ML",
                filename="pearson_correlation_traditional.pdf"
            )
            pearson_trad.plot()

            jaccard_trad = JaccardHeatmap(
                data=traditional_ml_errors,
                title="Jaccard Similarity Matrix - Traditional ML",
                filename="jaccard_similarity_traditional.pdf"
            )
            jaccard_trad.plot()

            mcnemar_trad = McNemarHeatmap(
                data=traditional_ml_errors,
                title="McNemar's Test Matrix - Traditional ML",
                filename="mcnemar_test_traditional.pdf"
            )
            mcnemar_trad.plot()
        
        # Plot correlation matrices for LLM models
        if len(llm_errors) > 1:
            pearson_llm = PearsonHeatmap(
                data=llm_errors,
                title="Pearson Correlation Matrix - LLMs",
                filename="pearson_correlation_llm.pdf"
            )
            pearson_llm.plot()

            jaccard_llm = JaccardHeatmap(
                data=llm_errors,
                title="Jaccard Similarity Matrix - LLMs",
                filename="jaccard_similarity_llm.pdf"
            )
            jaccard_llm.plot()

            mcnemar_llm = McNemarHeatmap(
                data=llm_errors,
                title="McNemar's Test Matrix - LLMs",
                filename="mcnemar_test_llm.pdf"
            )
            mcnemar_llm.plot()
        
        # Calculate cross-group correlations
        if traditional_ml_errors and llm_errors:
            # Combine all models from each group to create aggregate error patterns
            # For each text, if ANY model in the group made an error, count it as an error for the group
            ml_combined_errors = []
            llm_combined_errors = []
            
            # Get all texts that have predictions from both groups
            common_texts = set()
            for model, errors in named_errors.items():
                if model in traditional_ml_models or model in llm_models:
                    if not common_texts:
                        common_texts = set(range(len(errors)))
                    else:
                        common_texts &= set(range(len(errors)))
            
            common_texts = sorted(common_texts)
            
            # Create combined error lists
            for idx in common_texts:
                # For traditional ML models
                ml_error = False
                for model in traditional_ml_errors:
                    if traditional_ml_errors[model][idx]:  # If True, there was an error
                        ml_error = True
                        break
                ml_combined_errors.append(ml_error)
                
                # For LLM models
                llm_error = False
                for model in llm_errors:
                    if llm_errors[model][idx]:  # If True, there was an error
                        llm_error = True
                        break
                llm_combined_errors.append(llm_error)
            
            # Create a dictionary with combined errors
            cross_group_errors = {
                "Traditional ML (combined)": ml_combined_errors,
                "LLMs (combined)": llm_combined_errors
            }
                
            # Plot cross-group correlation matrices
            pearson_cross = PearsonHeatmap(
                data=cross_group_errors,
                title="Pearson Correlation - ML vs LLM (Combined)",
                filename="pearson_correlation_cross.pdf"
            )
            pearson_cross.plot()

            jaccard_cross = JaccardHeatmap(
                data=cross_group_errors,
                title="Jaccard Similarity - ML vs LLM (Combined)",
                filename="jaccard_similarity_cross.pdf"
            )
            jaccard_cross.plot()

            mcnemar_cross = McNemarHeatmap(
                data=cross_group_errors,
                title="McNemar's Test - ML vs LLM (Combined)",
                filename="mcnemar_test_cross.pdf"
            )
            mcnemar_cross.plot()

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

        
