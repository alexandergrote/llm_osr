import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Literal

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
from src.util.logger import console


class ErrorAnalyser(BaseModel, BaseAnalyser):

    def analyse(self, data: pd.DataFrame, **kwargs):

        # work on copy
        data_copy = data.copy(deep=True)

        data_exp_name = data_copy.filter(like="experiment")

        assert data_exp_name.shape[1] == 1, "There should only be one experiment column."
        exp_col_name = data_exp_name.columns[0]

        data_copy = data_copy.dropna(subset=[exp_col_name])

        predictions, data_train_list, _, data_test_list = get_artifacts(data_copy=data_copy)

        assert len(predictions) == len(data_copy), 'The number of predictions does not match the number of rows in the dataset.'
        assert any(isinstance(prediction, MLPrediction) for prediction in predictions), 'All predictions must be of type MLPrediction.'

        known_classes_list = []

        for train_df, test_df in zip(data_train_list, data_test_list):
            train_classes = np.unique(train_df.target())
            known_classes_list.append(
                np.isin(test_df.target(), train_classes)
            )
            
        # calculate errors
        errors_list = [prediction.error().values for prediction in predictions]
        raw_text = [mldf.raw_text() for mldf in data_test_list]

        # test how many errors are due to unknown classes
        known_errors_list = []
        unknown_errors_list = []
        for i, errors in enumerate(errors_list):
            known_errors_list.append(
                np.logical_and(known_classes_list[i], errors)
            )
            unknown_errors_list.append(
                np.logical_not(known_classes_list[i]) & errors
            )
            # calculate the number of errors due to unknown classes
            num_unknown_errors = np.sum(unknown_errors_list[i])
            num_known_errors = np.sum(known_errors_list[i])

        metric_col, dataset_col, perc_unknown_col = f1_analysis_columns.f1_avg.column_name, id_columns.dataset.column_name, id_columns.perc_unknown_classes.column_name
        unknown_f1_col = unknown_auc_analysis_columns.f1.column_name
        exp_name_col = id_columns.experiment_name.column_name
        random_seed_col = id_columns.seed.column_name

        all_columns = [metric_col, perc_unknown_col, dataset_col]

        for col in all_columns:
            assert col in data_copy.columns, f"'{col}' must be present in the analysis DataFrame."

        # check for all id columns
        data_grouped = data_copy.groupby([exp_name_col, random_seed_col]).size()
        
        # identify if there are any duplicate rows based on exp_name_col and random_seed_col
        # and print them
        mask = data_grouped > 1
        if any(mask):
            print(data_grouped[mask].index.to_list()) 
            raise ValueError("Duplicate rows found based on experiment name and random seed. Please remove duplicates before proceeding with the analysis.")

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
        exp_names = data_copy[exp_name_col].to_list()
        unknown_scores = data_copy[perc_unknown_col].to_list()
        random_seeds = data_copy[random_seed_col].to_list()
        dataset_names = data_copy[dataset_col].to_list()
        artifact_uris = data_copy['meta.artifact_uri'].to_list()

        error_dict = defaultdict(lambda: defaultdict(list))  # type: ignore
        known_dict = defaultdict(lambda: defaultdict(list))  # type: ignore

        # provides overview over all datasets and degrees of openness and random seeds
        # hence, multiple predictions for the same datapoint can occur
        for dataset, model, seed, openness, errors, texts, known_classes, exp_name, artifact_uri, num_known_errors, num_unknown_errors in zip(dataset_names, model_names, random_seeds, unknown_scores, errors_list, raw_text, known_classes_list, exp_names, artifact_uris, known_errors_list, unknown_errors_list):

            assert len(errors) == len(texts), f"Error and text lists must have the same length for model {model}."

            for error, text, known_class in zip(errors, texts, known_classes):  # handle length mismatches safely

                idx = f"{dataset}_{openness}_{seed}_{text}"
                        
                error_dict[idx][model].append(error)
                known_dict[idx][model].append(known_class)

                observations = known_dict[idx][model]
                if 1 < len(set(observations)):
                    console.log('conflicting labels for text:', exp_name, seed, text)
                    pass

        # needed for "natural" duplicates in data
        for text in error_dict.keys():
            for model in error_dict[text].keys():
                
                # get majority vote on model predictions for each text
                counter = Counter(error_dict[text][model])
                most_common_response = counter.most_common(1)[0][0]
                error_dict[text][model] = most_common_response

                # counter for known class
                counter = Counter(known_dict[text][model])
                most_common_response = counter.most_common(1)[0][0]
                known_dict[text][model] = most_common_response
        
        # Process different scenarios
        scenarios = ["all", "known", "unknown"]
        
        for folder in scenarios:

            (Directory.OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)
            
            scenario_named_errors = self._get_named_errors(error_dict, known_dict, folder)
            
            self._plot_matrices(scenario_named_errors, folder)

        data_copy.reset_index(drop=True, inplace=True)
    
        # analyse f1 scores with boxplot
        plt.figure()
        sns.barplot(
            x=model_col, y=metric_col, data=data_copy,
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(Directory.OUTPUT_DIR / 'all/f1_scores.pdf')
        plt.savefig(Directory.OUTPUT_DIR / 'all/f1_scores.png', dpi=300)
        plt.close()

        # analyse f1 scores with boxplot
        plt.figure()
        sns.barplot(
            x=model_col, y=unknown_f1_col, data=data_copy,
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(Directory.OUTPUT_DIR / 'all/f1_scores_unknown.pdf')
        plt.savefig(Directory.OUTPUT_DIR / 'all/f1_scores_unknown.png', dpi=300)
        plt.close()
        
        _, recall_col, precision_col = unknown_auc_analysis_columns.f1.column_name, unknown_auc_analysis_columns.recall.column_name, unknown_auc_analysis_columns.precision.column_name
        
        # precision
        plt.figure()
        sns.barplot(
            x=model_col, y=precision_col, data=data_copy,
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(Directory.OUTPUT_DIR / 'all/precision_scores_unknown.pdf')
        plt.savefig(Directory.OUTPUT_DIR / 'all/precision_scores_unknown.png', dpi=300)
        plt.close()

        # recall
        plt.figure()
        sns.barplot(
            x=model_col, y=recall_col, data=data_copy,
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(Directory.OUTPUT_DIR / 'all/recall_scores_unknown.pdf')
        plt.savefig(Directory.OUTPUT_DIR / 'all/recall_scores_unknown.png', dpi=300)
        plt.close()

    @staticmethod
    def _get_named_errors(error_dict: Dict, known_dict: Dict, scenario: Literal['all', 'known', 'unknown']) -> Dict:
        

        # Now create the DataFrame
        df = pd.DataFrame.from_dict(error_dict, orient='index')
        df = df.astype(bool)
        df.index.name = 'raw_text'
        df.reset_index(inplace=True)

        if scenario == "all":

            idx = df.index
        
        else:

            # Create a DataFrame for known/unknown status
            known_df = pd.DataFrame.from_dict(known_dict, orient='index')
            known_df = known_df.astype(bool)
            known_df.index.name = 'raw_text'
            known_df.reset_index(inplace=True)

            assert all(known_df["raw_text"] == df["raw_text"])

            known_flags_df = known_df.drop(columns=["raw_text"]).dropna(axis=0)

            if scenario == "known":
                idx = known_flags_df[known_flags_df.all(axis=1)].index
            else:
                idx = known_flags_df[~known_flags_df.all(axis=1)].index

        # Create named_errors for all data points
        named_errors = df.iloc[idx, :].dropna(axis=0).drop(columns=["raw_text"]).T.apply(list, axis=1).to_dict()
        
        return named_errors

    def _plot_matrices(self, named_errors, folder: str):

        # Define model groups
        traditional_ml_models = ['SimpleShot', 'FastFit', 'ContrastNet']
        llm_models = ['LLaMA 8', 'LLaMA 70', 'GEMMA3 27', 'Phi4 14']
        
        # Create separate error dictionaries for each group
        traditional_ml_errors = {model: named_errors[model] for model in traditional_ml_models if model in named_errors}
        llm_errors = {model: named_errors[model] for model in llm_models if model in named_errors}

        title_mapping = {
            'all': 'All classes',
            'known': 'Known classes',
            'unknown': 'Unknown classes'
        }

        title = title_mapping[folder]
        title = ''

        
        # Plot correlation matrices for all models
        ## pearson
        pearson = PearsonHeatmap(
            data=named_errors,
            title=title,
            filename=os.path.join(folder, "pearson_correlation_all.pdf")
        )
        pearson.plot()
        # Save as PNG with high DPI
        pearson = PearsonHeatmap(
            data=named_errors,
            title=title,
            filename=os.path.join(folder, "pearson_correlation_all.png")
        )
        pearson.plot()

        ## jaccard
        jaccard = JaccardHeatmap(
            data=named_errors,
            title=title,
            filename=os.path.join(folder, "jaccard_similarity_all.pdf")
        )
        jaccard.plot()
        # Save as PNG with high DPI
        jaccard = JaccardHeatmap(
            data=named_errors,
            title=title,
            filename=os.path.join(folder, "jaccard_similarity_all.png")
        )
        jaccard.plot()

        ## mcnemar
        mcnemar_heatmap = McNemarHeatmap(
            data=named_errors,
            title=title,
            filename=os.path.join(folder, "mcnemar_test_all.pdf")
        )
        mcnemar_heatmap.plot()
        # Save as PNG with high DPI
        mcnemar_heatmap = McNemarHeatmap(
            data=named_errors,
            title=title,
            filename=os.path.join(folder, "mcnemar_test_all.png")
        )
        mcnemar_heatmap.plot()
        
        # Plot correlation matrices for traditional ML models
        if len(traditional_ml_errors) > 1:
            pearson_trad = PearsonHeatmap(
                data=traditional_ml_errors,
                title=title,
                filename=os.path.join(folder, "pearson_correlation_traditional.pdf")
            )
            pearson_trad.plot()
            # Save as PNG with high DPI
            pearson_trad = PearsonHeatmap(
                data=traditional_ml_errors,
                title=title,
                filename=os.path.join(folder, "pearson_correlation_traditional.png")
            )
            pearson_trad.plot()

            jaccard_trad = JaccardHeatmap(
                data=traditional_ml_errors,
                title=title,
                filename=os.path.join(folder, "jaccard_similarity_traditional.pdf")
            )
            jaccard_trad.plot()
            # Save as PNG with high DPI
            jaccard_trad = JaccardHeatmap(
                data=traditional_ml_errors,
                title=title,
                filename=os.path.join(folder, "jaccard_similarity_traditional.png")
            )
            jaccard_trad.plot()

            mcnemar_trad = McNemarHeatmap(
                data=traditional_ml_errors,
                title=title,
                filename=os.path.join(folder, "mcnemar_test_traditional.pdf")
            )
            mcnemar_trad.plot()
            # Save as PNG with high DPI
            mcnemar_trad = McNemarHeatmap(
                data=traditional_ml_errors,
                title=title,
                filename=os.path.join(folder, "mcnemar_test_traditional.png")
            )
            mcnemar_trad.plot()
        
        # Plot correlation matrices for LLM models
        if len(llm_errors) > 1:
            pearson_llm = PearsonHeatmap(
                data=llm_errors,
                title=title,
                filename=os.path.join(folder, "pearson_correlation_llm.pdf")
            )
            pearson_llm.plot()
            # Save as PNG with high DPI
            pearson_llm = PearsonHeatmap(
                data=llm_errors,
                title=title,
                filename=os.path.join(folder, "pearson_correlation_llm.png")
            )
            pearson_llm.plot()

            jaccard_llm = JaccardHeatmap(
                data=llm_errors,
                title=title,
                filename=os.path.join(folder, "jaccard_similarity_llm.pdf")
            )
            jaccard_llm.plot()
            # Save as PNG with high DPI
            jaccard_llm = JaccardHeatmap(
                data=llm_errors,
                title=title,
                filename=os.path.join(folder, "jaccard_similarity_llm.png")
            )
            jaccard_llm.plot()

            mcnemar_llm = McNemarHeatmap(
                data=llm_errors,
                title=title,
                filename=os.path.join(folder, "mcnemar_test_llm.pdf")
            )
            mcnemar_llm.plot()
            # Save as PNG with high DPI
            mcnemar_llm = McNemarHeatmap(
                data=llm_errors,
                title=title,
                filename=os.path.join(folder, "mcnemar_test_llm.png")
            )
            mcnemar_llm.plot()
        
        # Calculate cross-group correlations
        if traditional_ml_errors and llm_errors:
            # Combine all models from each group to create aggregate error patterns
            # For each text, use the majority vote of errors across models in each group
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
            
            # Create combined error lists using majority vote
            for idx in common_texts:
                # For traditional ML models - count errors and use majority vote
                ml_votes = [traditional_ml_errors[model][idx] for model in traditional_ml_errors]
                ml_error = sum(ml_votes) > len(ml_votes) / 2  # True if majority made errors
                ml_combined_errors.append(ml_error)
                
                # For LLM models - count errors and use majority vote
                llm_votes = [llm_errors[model][idx] for model in llm_errors]
                llm_error = sum(llm_votes) > len(llm_votes) / 2  # True if majority made errors
                llm_combined_errors.append(llm_error)
            
            # Create a dictionary with combined errors
            cross_group_errors = {
                "Traditional ML (combined)": ml_combined_errors,
                "LLMs (combined)": llm_combined_errors
            }
                
            # Plot cross-group correlation matrices
            pearson_cross = PearsonHeatmap(
                data=cross_group_errors,
                title=title,
                filename=os.path.join(folder, "pearson_correlation_cross.pdf")
            )
            pearson_cross.plot()
            # Save as PNG with high DPI
            pearson_cross = PearsonHeatmap(
                data=cross_group_errors,
                title=title,
                filename=os.path.join(folder, "pearson_correlation_cross.png")
            )
            pearson_cross.plot()

            jaccard_cross = JaccardHeatmap(
                data=cross_group_errors,
                title=title,
                filename=os.path.join(folder, "jaccard_similarity_cross.pdf")
            )
            jaccard_cross.plot()
            # Save as PNG with high DPI
            jaccard_cross = JaccardHeatmap(
                data=cross_group_errors,
                title=title,
                filename=os.path.join(folder, "jaccard_similarity_cross.png")
            )
            jaccard_cross.plot()

            mcnemar_cross = McNemarHeatmap(
                data=cross_group_errors,
                title=title,
                filename=os.path.join(folder, "mcnemar_test_cross.pdf")
            )
            mcnemar_cross.plot()
            # Save as PNG with high DPI
            mcnemar_cross = McNemarHeatmap(
                data=cross_group_errors,
                title=title,
                filename=os.path.join(folder, "mcnemar_test_cross.png")
            )
            mcnemar_cross.plot()
            
            # Perform additional statistical tests on the correlation
            # Chi-square test of independence
            contingency_table = self._create_contingency_table(ml_combined_errors, llm_combined_errors)
            
            self._plot_contingency_table(contingency_table, folder)
            

    def _create_contingency_table(self, ml_errors: List[bool], llm_errors: List[bool]) -> np.ndarray:
        """
        Create a contingency table for chi-square test from two lists of boolean error indicators.
        
        Args:
            ml_errors: List of boolean values indicating errors for traditional ML models
            llm_errors: List of boolean values indicating errors for LLM models
            
        Returns:
            2x2 numpy array contingency table
        """
        # Count occurrences of each combination
        ml_correct_llm_correct = sum(1 for ml, llm in zip(ml_errors, llm_errors) if not ml and not llm)
        ml_correct_llm_error = sum(1 for ml, llm in zip(ml_errors, llm_errors) if not ml and llm)
        ml_error_llm_correct = sum(1 for ml, llm in zip(ml_errors, llm_errors) if ml and not llm)
        ml_error_llm_error = sum(1 for ml, llm in zip(ml_errors, llm_errors) if ml and llm)
        
        # Create the contingency table
        return np.array([
            [ml_correct_llm_correct, ml_correct_llm_error],
            [ml_error_llm_correct, ml_error_llm_error]
        ])
    
    def _calculate_phi_coefficient(self, contingency_table: np.ndarray) -> float:
        """
        Calculate the phi coefficient from a 2x2 contingency table.
        
        Args:
            contingency_table: 2x2 numpy array contingency table
            
        Returns:
            Phi coefficient value
        """
        if contingency_table.shape != (2, 2):
            return 0.0
            
        a, b = contingency_table[0]
        c, d = contingency_table[1]
        
        numerator = (a * d) - (b * c)
        denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator
    
    def _plot_contingency_table(self, contingency_table: np.ndarray, folder: str) -> None:
        """
        Visualize the contingency table as a heatmap.
        
        Args:
            contingency_table: 2x2 numpy array contingency table
        """
        plt.figure(figsize=(8, 6))
        
        # Calculate chi-square and p-value
        chi2, p = None, None

        try:
            chi2, p, _, _ = stats.chi2_contingency(contingency_table)
        except ValueError as e:
            print(f"Error calculating chi-square in folder {folder}: {e}")
        
        # Calculate phi coefficient
        phi = self._calculate_phi_coefficient(contingency_table)
        
        # Calculate total number of samples
        total = contingency_table.sum()
        
        # Calculate percentages
        percentages = contingency_table / total * 100
        
        # Create annotation text with only percentages
        annot_text = np.array([
            [f"{perc:.1f}%" for perc in perc_row]
            for perc_row in percentages
        ])
        
        # Create the heatmap with larger font size
        sns.heatmap(
            contingency_table, 
            annot=annot_text, 
            fmt='',
            cmap='Blues',
            xticklabels=['LLM Correct', 'LLM Error'],
            yticklabels=['ML Correct', 'ML Error'],
            cbar=False,
            annot_kws={"size": 16}  # Größere Schriftgröße für die Annotationen
        )
        
        if p is not None:
            # Add title with statistics (p-value with significance symbols)
            p_formatted = ""
            if p < 0.001:
                p_formatted = "p < 0.001"
            elif p < 0.01:
                p_formatted = "p < 0.01"
            elif p < 0.05:
                p_formatted = "p < 0.05"
            else:
                p_formatted = f"p = {p:.3f}"

        title = f"Contingency Table - ML vs LLM Errors\nPhi = {phi:.2f}"

        if (chi2 is not None) and (p is not None):
            title += f", {p_formatted}, Chi² = {chi2:.2f}"
            
        plt.title(title, fontsize=20)
        plt.xlabel("LLMs", fontsize=18)
        plt.ylabel("Traditional Fewshot Models", fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
        plt.tight_layout()
        plt.savefig(Directory.OUTPUT_DIR / os.path.join(folder, 'contingency_table.pdf'))
        # Save as PNG with high DPI
        plt.savefig(Directory.OUTPUT_DIR / os.path.join(folder, 'contingency_table.png'), dpi=300)
        plt.close()
