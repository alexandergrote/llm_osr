import numpy as np
import pandera as pa
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

from pandera.typing import DataFrame, Series
from pydantic import BaseModel, ConfigDict


class StrategyBoxPlotDatasetSchema(pa.DataFrameModel):
    dataset: Series[str] = pa.Field(coerce=True)
    model: Series[str] = pa.Field(coerce=True)
    prompt_version: Series[str] = pa.Field(coerce=True)
    precision: Series[float]
    recall: Series[float]
    F1: Series[float]


class StrategyBoxPlot(BaseModel):
    """
    Creates boxplots that compare performance across different prompting strategies,
    with one boxplot per metric and strategy combination.
    """
    data: DataFrame[StrategyBoxPlotDatasetSchema]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_datasets(self) -> List[str]:
        return self.data[StrategyBoxPlotDatasetSchema.dataset].unique()

    def get_models(self) -> List[str]:
        return self.data[StrategyBoxPlotDatasetSchema.model].unique()

    def get_prompts(self) -> List[str]:
        return self.data[StrategyBoxPlotDatasetSchema.prompt_version].unique()

    def get_metrics(self) -> List[str]:
        return ["precision", "recall", "F1"]

    def plot(self):
        metrics = self.get_metrics()
        prompt_versions = self.get_prompts()
        
        # Set the seaborn style
        sns.set(style="whitegrid")
        
        # Create a figure with a grid of subplots (metrics × prompt versions)
        fig, axes = plt.subplots(
            len(metrics), 
            len(prompt_versions), 
            figsize=(5 * len(prompt_versions), 4 * len(metrics)),
            sharex='col',  # Share x-axis within columns
            sharey='row'   # Share y-axis within rows
        )
        
        # Iterate through metrics and prompt versions to create the grid of plots
        for i, metric in enumerate(metrics):
            for j, prompt in enumerate(prompt_versions):
                # Filter data for this prompt version
                prompt_data = self.data[self.data['prompt_version'] == prompt]
                
                # Get the current axis
                ax = axes[i, j]
                
                # Create boxplot for this metric and prompt version
                sns.boxplot(
                    x='model',
                    y=metric,
                    hue='model',
                    data=prompt_data,
                    ax=ax,
                    palette='viridis',
                    width=0.6,
                    legend=False
                )
                
                # Add individual data points
                sns.stripplot(
                    x='model',
                    y=metric,
                    data=prompt_data,
                    ax=ax,
                    color='black',
                    alpha=0.5,
                    size=4,
                    jitter=True
                )
                
                # Set titles and labels
                if i == 0:  # Only set column titles for the first row
                    ax.set_title(f"Prompt Strategy: {prompt}", fontsize=14)
                
                if j == 0:  # Only set row labels for the first column
                    ax.set_ylabel(f"{metric.capitalize()}", fontsize=12)
                else:
                    ax.set_ylabel("")
                
                if i == len(metrics) - 1:  # Only set x-labels for the bottom row
                    ax.set_xlabel("Model", fontsize=12)
                else:
                    ax.set_xlabel("")
                
                # Rotate x-axis labels
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
                
                # Add reference lines
                ax.axhline(y=0.7, color='r', linestyle='--', alpha=0.3)
                ax.axhline(y=0.8, color='g', linestyle='--', alpha=0.3)
                
                # Set y-axis limits for consistency
                ax.set_ylim(0.65, 0.85)
        
        # Add a legend for the reference lines in the top-right subplot
        axes[0, -1].legend(
            handles=[
                plt.Line2D([0], [0], color='r', linestyle='--', alpha=0.3, label='0.7 threshold'),
                plt.Line2D([0], [0], color='g', linestyle='--', alpha=0.3, label='0.8 threshold')
            ],
            loc='upper right'
        )
        
        # Add overall title
        plt.suptitle("Performance Metrics by Prompting Strategy", fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("strategy_boxplot.pdf")
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
    
    strategy_boxplot = StrategyBoxPlot(data=df)
    strategy_boxplot.plot()
