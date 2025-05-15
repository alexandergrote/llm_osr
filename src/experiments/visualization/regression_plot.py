import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

class RegressionPlot(BaseModel):
    """
    Creates a regression plot that shows the relationship between openness degree (x-axis)
    and F1 scores (y-axis) for different models.
    """

    data: pd.DataFrame
    x_column: str = "Openness"
    y_column: str = "mean"
    hue_column: str = "Model"
    title: str = "F1 Score vs. Openness Degree"
    x_label: str = "Openness Degree"
    y_label: str = "F1 Score"
    figsize: tuple = (12, 8)
    output_path: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def plot(self) -> None:
        """
        Generate and display the regression plot.
        """
        plt.figure(figsize=self.figsize)
        
        # Create the regression plot using seaborn
        sns.set_style("whitegrid")
        ax = sns.lmplot(
            data=self.data,
            x=self.x_column,
            y=self.y_column,
            hue=self.hue_column,
            height=8,
            aspect=1.5,
            scatter_kws={"s": 80, "alpha": 0.7},
            line_kws={"linewidth": 2},
            legend=False
        )
        
        # Customize the plot
        plt.title(self.title, fontsize=16)
        plt.xlabel(self.x_label, fontsize=14)
        plt.ylabel(self.y_label, fontsize=14)
        
        # Add legend with better positioning
        plt.legend(title=self.hue_column, title_fontsize=12, fontsize=10, 
                  loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        
        plt.tight_layout()
        
        # Save the plot if output path is provided
        if self.output_path:
            plt.savefig(self.output_path, dpi=300, bbox_inches='tight')
            
        plt.show()
