import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import os
from tabulate import tabulate
from collections import defaultdict

from src.util.load_hydra import get_hydra_config
from src.util.dynamic_import import DynamicImport
from src.io.data_import.base import BaseDataset
from src.util.constants import DatasetColumn as dfc
from src.util.logger import console
from src.util.constants import Directory

datasets = [
    "clinc", "hwu", "banking"
]

def calculate_dataset_statistics(data, classes, dataset_name):
    """Calculate common statistics for text datasets."""
    stats = {}
    stats["Dataset"] = dataset_name
    stats["Samples"] = len(data)
    stats["Classes"] = len(classes)
    stats["Avg. samples per class"] = round(data.groupby(dfc.LABEL).size().mean(), 1)
    stats["Min samples per class"] = data.groupby(dfc.LABEL).size().min()
    stats["Max samples per class"] = data.groupby(dfc.LABEL).size().max()
    
    # Text statistics
    data['text_length'] = data['text'].apply(len)
    data['word_count'] = data['text'].apply(lambda x: len(x.split()))
    
    stats["Avg. text length (chars)"] = round(data['text_length'].mean(), 1)
    stats["Avg. word count"] = round(data['word_count'].mean(), 1)
    stats["Min text length"] = data['text_length'].min()
    stats["Max text length"] = data['text_length'].max()
    
    # Class distribution statistics
    class_distribution = data.groupby(dfc.LABEL).size()
    stats["Class imbalance ratio"] = round(class_distribution.max() / class_distribution.min(), 2)
    
    # Calculate vocabulary size (unique words)
    all_words = ' '.join(data['text']).lower().split()
    stats["Vocabulary size"] = len(set(all_words))
    
    return stats

if __name__ == "__main__":
    # Dictionary to store statistics for all datasets
    all_stats = []

    target_dir = Directory.OUTPUT_DIR / "eda"
    target_dir.mkdir(exist_ok=True, parents=True)
    
    for dataset in datasets:

        config = get_hydra_config(overrides=[f"io__import={dataset}"])

        dataloader = DynamicImport.init_class_from_dict(
            dictionary=config['io__import']
        )

        assert isinstance(dataloader, BaseDataset)

        data_dict = dataloader.execute()

        assert "data" in data_dict
        assert "all_classes" in data_dict

        console.rule(f"Dataset: {dataset}")

        data = data_dict['data']
        assert isinstance(data, pd.DataFrame)
        classes = data_dict['all_classes']

        console.print("Extract of data")
        for idx, row in data.sample(5, random_state=42).iterrows():
            console.print("-"*5, "Example:", idx +1 , "-"*5)
            console.print(row["text"])
            console.print("Label:", row["label"])

        
        console.print("Classes:")
        console.print("\n".join(classes))

        console.print("Stats")
        console.print(f"Number of data points: {len(data)}")
        console.print(f"Number of classes: {len(classes)}")

        console.print("Stats per class")
        console.print(f"Observations per class: {data.groupby(dfc.LABEL).size()}")
        console.print(f"Average samples per class: {data.groupby(dfc.LABEL).size().mean()}")

        # visualize the number of samples per class in descending order
        # instead of the label names, it should display an id
        data2plot = data.groupby(dfc.LABEL).size().sort_values(ascending=False)#.plot.bar()
        plt.title(f"{dataset.upper()}")
        sns.barplot(data=data2plot.reset_index().reset_index(), x="index", y=0)
        plt.xlabel("Class")
        plt.ylabel("Number of samples")
        ax = plt.gca()
        temp = ax.xaxis.get_ticklabels()
        temp = list(set(temp) - set(temp[::5]))
        for label in temp:
            label.set_visible(False)
        plt.tight_layout()
        plt.savefig(target_dir / f"{dataset}_class_distribution.png")
        
        # Calculate statistics for this dataset
        dataset_stats = calculate_dataset_statistics(data, classes, dataset)
        all_stats.append(dataset_stats)
    
    # Create a table with all dataset statistics
    console.rule("Dataset Statistics Table")
    
    # Convert to DataFrame for easier manipulation
    stats_df = pd.DataFrame(all_stats)
    
    # Print table in different formats
    console.print("Markdown Table Format:")
    console.print(tabulate(stats_df, headers='keys', tablefmt='pipe', showindex=False))
    
    console.print("\nLaTeX Table Format:")
    latex_table = tabulate(stats_df, headers='keys', tablefmt='latex', showindex=False)
    console.print(latex_table)

    with open(target_dir / "dataset_statistics_latex.tex", "w") as f:
        f.write(tabulate(stats_df, headers='keys', tablefmt='latex', showindex=False))
    
    console.print(f"\nTables saved to {target_dir}/dataset_statistics_latex.tex")

