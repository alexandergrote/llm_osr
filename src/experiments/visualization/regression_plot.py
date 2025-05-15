import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field, ConfigDict

class RegressionPlot(BaseModel):
    """
    Creates a regression plot that shows the relationship between openness degree (x-axis)
    and F1 scores (y-axis) for different models.
    """

    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    x_column: str = "Openness"
    y_column: str = "mean"
    hue_column: str = "Model"
    title: str = "F1 Score vs. Openness Degree"
    x_label: str = "Openness Degree"
    y_label: str = "F1 Score"
    figsize: tuple = (16, 6)
    output_path: Optional[str] = None
    dataset_titles: Optional[List[str]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def plot(self) -> None:
        """
        Generate and display the regression plot.
        """
        # Handle both single dataframe and dictionary of dataframes
        if isinstance(self.data, pd.DataFrame):
            datasets = {"Dataset": self.data}
            titles = ["Dataset"]
        else:
            datasets = self.data
            titles = self.dataset_titles or list(datasets.keys())
        
        # Create a figure with subplots in one row
        fig, axes = plt.subplots(1, len(datasets), figsize=self.figsize)
        if len(datasets) == 1:
            axes = [axes]  # Make axes iterable if only one subplot
        
        sns.set_style("whitegrid")
        
        # Create a common legend for all subplots
        # First, get all unique model names across all datasets
        all_models = set()
        for dataset_name, df in datasets.items():
            all_models.update(df[self.hue_column].unique())
        
        # Plot each dataset in its own subplot
        for i, (dataset_name, df) in enumerate(datasets.items()):
            # Ensure data types are correct
            data_copy = df.copy()
            data_copy[self.x_column] = pd.to_numeric(data_copy[self.x_column], errors='coerce')
            data_copy[self.y_column] = pd.to_numeric(data_copy[self.y_column], errors='coerce')
            
            # Create scatter plot
            sns.scatterplot(
                data=data_copy,
                x=self.x_column,
                y=self.y_column,
                hue=self.hue_column,
                s=80,
                alpha=0.7,
                ax=axes[i]
            )
            
            # Add regression lines for each group
            for name, group in data_copy.groupby(self.hue_column):
                sns.regplot(
                    x=self.x_column,
                    y=self.y_column,
                    data=group,
                    scatter=False,
                    ax=axes[i],
                    line_kws={"linewidth": 2}
                )
            
            # Customize the subplot
            axes[i].set_title(titles[i], fontsize=14)
            axes[i].set_xlabel(self.x_label, fontsize=12)
            if i == 0:  # Only add y-label to the first subplot
                axes[i].set_ylabel(self.y_label, fontsize=12)
            else:
                axes[i].set_ylabel("")
            
            # Remove all legends from subplots
            if axes[i].get_legend() is not None:
                axes[i].get_legend().remove()
        
        # Add a main title
        fig.suptitle(self.title, fontsize=16, y=1.05)
        
        # Create a common legend above the plots
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), 
                  ncol=min(len(labels), 4), title=self.hue_column, fontsize=10, title_fontsize=12)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the legend at the top
        
        # Save the plot if output path is provided
        if self.output_path:
            plt.savefig(self.output_path, dpi=300, bbox_inches='tight')
            
        plt.show()
