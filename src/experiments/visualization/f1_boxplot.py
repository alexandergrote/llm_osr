import numpy as np
import pandera as pa
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import List, Dict, NamedTuple, Any

from pandera.typing import DataFrame, Series
from pydantic import BaseModel, ConfigDict

from src.util.constants import Directory
from src.experiments.visualization.strategy_boxplot import StrategyBoxPlot, StatTestResult

class F1BoxPlot(StrategyBoxPlot):
    """
    Creates boxplots that compare F1 score performance across different prompting strategies.
    """
    
    def get_metrics(self) -> List[str]:
        return ["F1"]
    
    def plot(self, dataset=None):
        """
        Create F1 score boxplots for the specified dataset or for all datasets if None.
        
        Args:
            dataset: Optional dataset to filter by. If None, plots all datasets combined.
        """
        # Filter data if a specific dataset is requested
        plot_data = self.data
        if dataset is not None:
            plot_data = self.data[self.data['dataset'] == dataset]
        
        # Perform statistical tests
        stat_results = self.perform_statistical_tests()
        
        # Create the plot
        self._create_plot_for_metric("F1", plot_data, stat_results, dataset)
        
        plt.close()
