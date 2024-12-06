import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

from src.util.load_hydra import get_hydra_config
from src.util.dynamic_import import DynamicImport
from src.io.data_import.base import BaseDataset
from src.util.constants import DatasetColumn as dfc
from src.util.logger import console

datasets = [
    "clinc", "hwu", "banking"
]

if __name__ == "__main__":

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
        plt.show()

