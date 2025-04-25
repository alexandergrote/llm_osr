import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from sklearn.metrics import jaccard_score
from statsmodels.stats.contingency_tables import mcnemar

from src.util.constants import Directory


class BaseHeatmap(BaseModel, ABC):

    title: str
    filename: str
    data: Dict[str, List[float]]
    show: bool = False

    def _plot(self, matrix: pd.DataFrame, heatmap_kwargs: Optional[Dict[str, Any]] = None) -> None:

        if heatmap_kwargs is None:
            heatmap_kwargs = {}

        sns.heatmap(matrix, **heatmap_kwargs)
        plt.title(self.title)
        plt.tight_layout()
        plt.savefig(Directory.OUTPUT_DIR / self.filename)
        if self.show:
            plt.show()
        plt.close()

    @abstractmethod
    def plot(self) -> None:
        raise NotImplementedError("Subclasses must implement this method.")


class PearsonHeatmap(BaseHeatmap):

    def plot(self) -> None:

        """
        Plot a heatmap of the Pearson correlation matrix.
    
        """

        error_df = pd.DataFrame(self.data)
        corr_matrix = error_df.corr(method="pearson")

        self._plot(corr_matrix)


class JaccardHeatmap(BaseHeatmap):

    def plot(self) -> None:

        # Assume binary error vectors: 1 = incorrect, 0 = correct
        model_names = list(self.data.keys())
        n = len(model_names)

        # Initialize empty matrix
        jaccard_matrix = np.zeros((n, n))

        # Fill matrix with pairwise Jaccard scores
        for i in range(n):
            for j in range(n):
                jaccard_matrix[i, j] = jaccard_score(self.data[model_names[i]], self.data[model_names[j]])

        # Convert to pandas DataFrame for readability
        jaccard_df = pd.DataFrame(jaccard_matrix, index=model_names, columns=model_names)

        heatmap_kwargs = dict(annot=True, cmap='YlGnBu')

        self._plot(jaccard_df, heatmap_kwargs=heatmap_kwargs)


class McNemarHeatmap(BaseHeatmap):

    def plot(self) -> None:
        model_names = list(self.data.keys())
        n = len(model_names)
        mcnemar_matrix = np.ones((n, n))  # Start with 1s for diagonal

        for i in range(n):
            for j in range(n):

                if i != j:
                    a_errors = self.data[model_names[i]]
                    b_errors = self.data[model_names[j]]

                    if not isinstance(a_errors, list):
                        raise ValueError("Expected list of errors for each model")
                    
                    if not isinstance(b_errors, list):
                        raise ValueError("Expected list of errors for each model")

                    a_errors = np.array(a_errors)
                    b_errors = np.array(b_errors)

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
                    except Exception as e:
                        print(e)
                        mcnemar_matrix[i, j] = np.nan  # In case counts are too small

        mcnemar_df = pd.DataFrame(mcnemar_matrix, index=model_names, columns=model_names)
        heatmap_kwargs = dict(annot=True, cmap='coolwarm', vmin=0, vmax=1)
        
        self._plot(mcnemar_df, heatmap_kwargs=heatmap_kwargs)

if __name__ == '__main__':

    named_errors: Dict[str, List[bool]] = {
        "model_a": [True, False, True, False],   # Alternating errors (50%)
        "model_b": [False, True, False, True],   # Inverted alternation (50%)
        "model_c": [True, True, False, True],    # Clustered high error rate (75%)
        "model_d": [False, False, True, True],   # Late errors (50%, different pattern)
        "model_e": [True, True, True, False],    # Early cluster (75%)
        "model_f": [False, True, True, False],   # Middle cluster (50%) 
        "model_g": [False, False, False, False], # Perfect model
        "model_h": [False, False, False, False] 
    }

    heatmap_kwargs = {
        "cmap": "coolwarm",
        "annot": True,
        "fmt": ".2f",
        "linewidths": .5,
        "square": True,
        "center": 0.5,
        "vmin": -1,
        "vmax": 1
    }

    title = "Error Rate Heatmap"
    filename = "tmp_heatmap.png"

    kwargs = {
        "show": True,
        "data": named_errors,
        "title": title,
        "filename": filename
    }

    PearsonHeatmap(**kwargs).plot()
    JaccardHeatmap(**kwargs).plot()
    McNemarHeatmap(**kwargs).plot()