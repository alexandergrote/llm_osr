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
        
        # Set the seaborn style
        sns.set(style="whitegrid")
        
        # Create a figure with one subplot per metric
        fig, axes = plt.subplots(
            len(metrics), 
            1, 
            figsize=(10, 4 * len(metrics)),
            sharex=True  # Share x-axis between subplots
        )
        
        # If there's only one metric, axes won't be an array
        if len(metrics) == 1:
            axes = [axes]
        
        # Define a color palette for prompt strategies
        prompt_versions = self.get_prompts()
        palette = sns.color_palette("viridis", len(prompt_versions))
        
        # Iterate through metrics to create the plots
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Create boxplot for this metric, grouped by prompt_version
            sns.boxplot(
                x='prompt_version',
                y=metric,
                data=self.data,
                ax=ax,
                palette=palette,
                width=0.7,
                order=sorted(prompt_versions)
            )
            
            # Add individual data points
            sns.stripplot(
                x='prompt_version',
                y=metric,
                data=self.data,
                ax=ax,
                color='black',
                alpha=0.5,
                size=4,
                jitter=True,
                order=sorted(prompt_versions)
            )
            
            # Set titles and labels
            ax.set_ylabel(f"{metric.capitalize()}", fontsize=12)
            
            if i == len(metrics) - 1:  # Only set x-labels for the bottom plot
                ax.set_xlabel("Prompt Strategy", fontsize=12)
            else:
                ax.set_xlabel("")
            
            # Add reference lines
            ax.axhline(y=0.7, color='r', linestyle='--', alpha=0.3)
            ax.axhline(y=0.8, color='g', linestyle='--', alpha=0.3)
            
            # Set y-axis limits for consistency
            ax.set_ylim(0.65, 0.85)
            
            # Add a legend for the reference lines
            ax.legend(
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
