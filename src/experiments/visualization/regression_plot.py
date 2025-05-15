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
        else:
            datasets = self.data
        
        # Create a single figure
        plt.figure(figsize=self.figsize)
        sns.set_style("whitegrid")
        
        # Combine all datasets with a dataset identifier column
        combined_data = pd.DataFrame()
        for dataset_name, df in datasets.items():
            # Ensure data types are correct
            data_copy = df.copy()
            data_copy[self.x_column] = pd.to_numeric(data_copy[self.x_column], errors='coerce')
            data_copy[self.y_column] = pd.to_numeric(data_copy[self.y_column], errors='coerce')
            
            # Add dataset identifier
            data_copy['Dataset'] = dataset_name
            combined_data = pd.concat([combined_data, data_copy])
        
        # Create scatter plot with both hue and style for differentiation
        sns.scatterplot(
            data=combined_data,
            x=self.x_column,
            y=self.y_column,
            hue=self.hue_column,
            style='Dataset',
            s=80,
            alpha=0.7
        )
        
        # Add regression lines for each group
        for (model, dataset), group in combined_data.groupby([self.hue_column, 'Dataset']):
            sns.regplot(
                x=self.x_column,
                y=self.y_column,
                data=group,
                scatter=False,
                label=f"{model} ({dataset})",
                line_kws={"linewidth": 2}
            )
        
        # Customize the plot
        plt.title(self.title, fontsize=16)
        plt.xlabel(self.x_label, fontsize=14)
        plt.ylabel(self.y_label, fontsize=14)
        
        # Add legend with better positioning
        plt.legend(title=f"{self.hue_column} (Dataset)", title_fontsize=12, fontsize=10, 
                  loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot if output path is provided
        if self.output_path:
            plt.savefig(self.output_path, dpi=300, bbox_inches='tight')
            
        plt.show()
