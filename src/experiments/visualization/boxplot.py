import numpy as np
import pandera as pa
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

from pandera.typing import DataFrame, Series
from pydantic import BaseModel, ConfigDict


class BoxPlotDatasetSchema(pa.DataFrameModel):
    dataset: Series[str] = pa.Field(coerce=True)
    model: Series[str] = pa.Field(coerce=True)
    prompt_version: Series[str] = pa.Field(coerce=True)
    precision: Series[float]
    recall: Series[float]
    F1: Series[float]


class BoxPlot(BaseModel):

    data: DataFrame[BoxPlotDatasetSchema]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_datasets(self) -> List[str]:
        return self.data[BoxPlotDatasetSchema.dataset].unique()

    def get_models(self) -> List[str]:
        return self.data[BoxPlotDatasetSchema.model].unique()

    def get_prompts(self) -> List[str]:
        return self.data[BoxPlotDatasetSchema.prompt_version].unique()

    def get_metrics(self) -> List[str]:
        return ["precision", "recall", "F1"]

    def plot(self):
        metrics = self.get_metrics()
        
        # Set the seaborn style
        sns.set(style="whitegrid")
        
        # Create subplots stacked vertically
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        
        # Add a combined model-prompt column for better visualization
        self.data['model_prompt'] = self.data['model'] + ' (' + self.data['prompt_version'] + ')'
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Create seaborn boxplot
            sns.boxplot(
                x='model_prompt',
                y=metric,
                data=self.data,
                ax=ax,
                palette='viridis',
                width=0.6
            )
            
            # Add individual data points for better visualization
            sns.stripplot(
                x='model_prompt',
                y=metric,
                data=self.data,
                ax=ax,
                color='black',
                alpha=0.5,
                size=4,
                jitter=True
            )
            
            # Set labels and title
            ax.set_title(f"{metric.capitalize()}", fontsize=14)
            ax.set_ylabel("Score", fontsize=12)
            ax.set_xlabel("")
            
            # Rotate x-axis labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
            
            # Add reference lines
            ax.axhline(y=0.7, color='r', linestyle='--', alpha=0.3, label='0.7 threshold')
            ax.axhline(y=0.8, color='g', linestyle='--', alpha=0.3, label='0.8 threshold')
            
            # Add legend for the first plot only
            if i == 0:
                ax.legend(loc='lower left')
                
            # Set y-axis limits for consistency across plots
            ax.set_ylim(0.65, 0.85)
        
        plt.suptitle("Performance Metrics Across Models and Prompt Versions", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig("boxplot.pdf")
        plt.show()


if __name__ == '__main__':

    import pandas as pd
    from itertools import product

    datasets = ["A", "B"]
    models = ["GPT-3.5", "GPT-4", "Llama 2", "Claude 2", "Mistral"]
    prompt_versions = ["osr", "ad", "nd"]
    metrics = ["precision", "recall", "F1"]

    # Creating all possible combinations
    data = []
    for dataset, model, prompt in product(datasets, models, prompt_versions):
        precision = round(0.75 + (hash(dataset + model + prompt) % 10) / 100, 2)
        recall = round(0.70 + (hash(model + prompt) % 10) / 100, 2)
        f1 = round((2 * precision * recall) / (precision + recall), 2)
        data.append([dataset, model, prompt, precision, recall, f1])

    # Creating the DataFrame
    df = pd.DataFrame(data, columns=["dataset", "model", "prompt_version", "precision", "recall", "F1"])
    print(df)

    boxplot = BoxPlot(data=df)
    boxplot.plot()
