import numpy as np
import pandera as pa

import matplotlib.pyplot as plt
from typing import List

from pandera.typing import DataFrame, Series
from pydantic import BaseModel, ConfigDict

from src.util.constants import Directory


class SpiderDatasetSchema(pa.DataFrameModel):
    dataset: Series[str] = pa.Field(coerce=True)
    model: Series[str] = pa.Field(coerce=True)
    prompt_version: Series[str] = pa.Field(coerce=True)
    precision: Series[float]
    recall: Series[float]
    F1: Series[float]


class SpiderPlot(BaseModel):

    data: DataFrame[SpiderDatasetSchema]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_datasets(self) -> List[str]:
        return self.data[SpiderDatasetSchema.dataset].unique()

    def get_models(self) -> List[str]:
        return self.data[SpiderDatasetSchema.model].unique()

    def get_prompts(self) -> List[str]:
        return self.data[SpiderDatasetSchema.prompt_version].unique()

    def get_metrics(self) -> List[str]:
        return [SpiderDatasetSchema.precision, SpiderDatasetSchema.recall, SpiderDatasetSchema.F1]

    def plot(self):

        models = self.get_models()
        datasets = self.get_datasets()
        prompts = self.get_prompts()
        metrics = self.get_metrics()

        angles = np.linspace(0, 2 * np.pi, len(models), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, axes = plt.subplots(len(metrics), len(datasets), figsize=(18, 12), subplot_kw={"projection": "polar"})
        if len(datasets) == 1:
            axes = np.expand_dims(axes, axis=1)
        if len(metrics) == 1:
            axes = np.expand_dims(axes, axis=0)
        
        for i, metric in enumerate(metrics):
            for j, dataset in enumerate(datasets):
                ax = axes[i, j]
                for prompt in prompts:
                    values = [self.data[(self.data[SpiderDatasetSchema.dataset] == dataset) & (self.data[SpiderDatasetSchema.model] == model) & (self.data[SpiderDatasetSchema.prompt_version] == prompt)][metric].values[0] for model in models]
                    values += values[:1]
                    ax.plot(angles, values, label=prompt, linewidth=2)
                    ax.fill(angles, values, alpha=0.1)
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(models)
                ax.set_title(f"{metric.capitalize()} - {dataset}")
        
        # Add a single legend at the top
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(prompts))
        plt.subplots_adjust(hspace=1)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(Directory.OUTPUT_DIR / f"spider_plot.pdf")
        #plt.show()

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

    spider_plot = SpiderPlot(data=df)
    spider_plot.plot()