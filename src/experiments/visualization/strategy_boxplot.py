import numpy as np
import pandera as pa
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import List, Dict, NamedTuple, Any

from pandera.typing import DataFrame, Series
from pydantic import BaseModel, ConfigDict

from src.util.constants import Directory

class StrategyBoxPlotDatasetSchema(pa.DataFrameModel):
    dataset: Series[str] = pa.Field(coerce=True)
    model: Series[str] = pa.Field(coerce=True)
    prompt_version: Series[str] = pa.Field(coerce=True)
    precision: Series[float]
    recall: Series[float]
    F1: Series[float]


class StatTestResult(NamedTuple):
    """Container for statistical test results"""
    statistic: float
    p_value: float
    effect_size: float
    effect_size_interpretation: str

    def __str__(self) -> str:
        """String representation of test results"""
        return f"p={self.p_value:.3f}, d={self.effect_size:.2f} ({self.effect_size_interpretation})"


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
        
    def perform_statistical_tests(self) -> Dict[str, Dict[str, StatTestResult]]:
        """
        Perform Mann-Whitney U tests between all pairs of prompt strategies for each metric.
        Also calculates Cliff's delta as effect size measure.
        
        Returns:
            Dict with structure: {metric: {strategy1_vs_strategy2: StatTestResult}}
        """
        metrics = self.get_metrics()
        prompt_versions = sorted(self.get_prompts())
        results: Dict[str, Any] = {}
        
        for metric in metrics:
            results[metric] = {}
            
            # Compare each pair of prompt strategies
            for i, prompt1 in enumerate(prompt_versions):
                for prompt2 in prompt_versions[i+1:]:
                    # Get data for each prompt strategy
                    data1 = self.data[self.data['prompt_version'] == prompt1][metric].values
                    data2 = self.data[self.data['prompt_version'] == prompt2][metric].values
                    
                    # Perform Mann-Whitney U test
                    statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                    
                    # Calculate Cliff's delta (effect size)
                    effect_size = self._cliffs_delta(data1, data2)
                    
                    # Interpret effect size
                    if abs(effect_size) < 0.147:
                        interpretation = "negligible"
                    elif abs(effect_size) < 0.33:
                        interpretation = "small"
                    elif abs(effect_size) < 0.474:
                        interpretation = "medium"
                    else:
                        interpretation = "large"
                    
                    # Store results
                    comparison_key = f"{prompt1}_vs_{prompt2}"
                    results[metric][comparison_key] = StatTestResult(
                        statistic=statistic,
                        p_value=p_value,
                        effect_size=effect_size,
                        effect_size_interpretation=interpretation
                    )
        
        return results
    
    def _cliffs_delta(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate Cliff's delta, a non-parametric effect size measure.
        
        Args:
            x: First sample
            y: Second sample
            
        Returns:
            Cliff's delta value between -1 and 1
        """
        # Count the number of times values in x are greater/less than values in y
        greater = 0
        less = 0
        
        for i in x:
            for j in y:
                if i > j:
                    greater += 1
                elif i < j:
                    less += 1
        
        # Calculate Cliff's delta
        return (greater - less) / (len(x) * len(y))

    def _create_plot_for_metric(self, metric, plot_data, stat_results, dataset=None, save=True):
        """
        Create a boxplot for a single metric.
        
        Args:
            metric: The metric to plot
            plot_data: DataFrame containing the data to plot
            stat_results: Statistical test results
            dataset: Optional dataset name for title
            save: Whether to save the plot to a file
            
        Returns:
            The figure and axes objects
        """
        # Set the seaborn style for academic publication
        sns.set_style("white")
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300
        })
        
        # Create a figure for this metric
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        
        # Use a single color for all boxes
        prompt_versions = sorted(self.get_prompts())
        single_color = 'gray'  # Dark gray color
        palette = [single_color] * len(prompt_versions)
        
        # Create boxplot for this metric with same color for all boxes
        sns.boxplot(
            x='prompt_version',
            y=metric,
            hue='prompt_version',  # Keep hue parameter to avoid FutureWarning
            data=plot_data,
            ax=ax,
            palette=palette,
            width=0.6,
            order=prompt_versions,
            linewidth=1.0,
            legend=False,  # Hide legend
            fliersize=3,
            flierprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markeredgewidth': 0.8}
        )
        
        # Add grid lines for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
        # Set titles and labels with academic styling
        ax.set_title(f"{metric.capitalize()}", fontweight='bold')
        ax.set_ylabel(f"{metric.capitalize()}", fontweight='bold')
        ax.set_xlabel("Prompt Strategy", fontweight='bold')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set y-axis limits for consistency
        ax.set_ylim(0, 1)
    
        # Add statistical significance annotations
        # Get x-coordinates for each prompt version
        x_coords = {prompt: idx for idx, prompt in enumerate(prompt_versions)}
        
        # Add significance bars and annotations
        bar_height = 0.02  # Height of significance bars
        text_height = 0.01  # Height of p-value text
        text_buffer = 0.04  # Buffer between bars and text to avoid overlap
        max_bars = len(prompt_versions) * (len(prompt_versions) - 1) // 2
        
        # Calculate y positions for significance bars
        y_max = ax.get_ylim()[1]
        bar_positions = np.linspace(
            y_max, 
            y_max + (bar_height + text_height + text_buffer) * max_bars, 
            max_bars
        )
        
        bar_idx = 0
        for p_i, prompt1 in enumerate(prompt_versions):
            for _, prompt2 in enumerate(prompt_versions[p_i+1:], p_i+1):
                comparison_key = f"{prompt1}_vs_{prompt2}"
                if comparison_key in stat_results[metric]:
                    result = stat_results[metric][comparison_key]
                    p_value = result.p_value
                    effect_size = result.effect_size
                
                    # Determine significance level
                    if p_value < 0.001:
                        sig_symbol = '***'
                    elif p_value < 0.01:
                        sig_symbol = '**'
                    elif p_value < 0.05:
                        sig_symbol = '*'
                    else:
                        sig_symbol = ''
                
                    # Get x positions
                    x1, x2 = x_coords[prompt1], x_coords[prompt2]
                    y = bar_positions[bar_idx]
                
                    # Draw the bar with academic styling
                    ax.plot([x1, x2], [y, y], 'k-', linewidth=0.8)
                    ax.plot([x1, x1], [y-bar_height/2, y], 'k-', linewidth=0.8)
                    ax.plot([x2, x2], [y-bar_height/2, y], 'k-', linewidth=0.8)
                
                    # Add p-value and effect size text with academic styling
                    sig_value = r"$^{" + sig_symbol + r"}$"
                    ax.text(
                        (x1+x2)/2, 
                        y + text_height*2,
                        f"p={p_value:.3f}{sig_value}, d={effect_size:.2f}",
                        ha='center', 
                        va='bottom', 
                        color="black",
                        fontsize=8, 
                        fontstyle='italic'
                    )
                
                    bar_idx += 1
        
        # Adjust y-axis limits to accommodate significance bars
        if bar_idx > 0:
            ax.set_ylim(0.0, bar_positions[bar_idx-1] + text_height * 3)
            
        # Determine title and filename
        if dataset is None:
            title = f"{metric.capitalize()} by Prompting Strategy"
            filename_base = f"strategy_boxplot_{metric.lower()}_combined"
        else:
            title = f"{metric.capitalize()} for Dataset {dataset} by Prompting Strategy"
            filename_base = f"strategy_boxplot_{metric.lower()}_dataset_{dataset}"
        
        # Add title to the figure
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.05)
        
        # Adjust layout and save if requested
        plt.tight_layout()
        
        if save:
            full_dir = Directory.OUTPUT_DIR / 'regression_plots'
            full_dir.mkdir(exist_ok=True)
            fig.savefig(full_dir / f"{filename_base}.pdf", bbox_inches='tight')
            fig.savefig(full_dir / f"{filename_base}.png", bbox_inches='tight', dpi=300)
        
        return fig, ax
        
    def plot(self, dataset=None):
        """
        Create boxplots for the specified dataset or for all datasets if None.
        
        Args:
            dataset: Optional dataset to filter by. If None, plots all datasets combined.
        """
        metrics = self.get_metrics()
        
        # Filter data if a specific dataset is requested
        plot_data = self.data
        if dataset is not None:
            plot_data = self.data[self.data['dataset'] == dataset]
        
        # Perform statistical tests
        stat_results = self.perform_statistical_tests()
        
        # Set the seaborn style for academic publication
        sns.set_style("white")
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300
        })
        
        # Create a figure with one subplot for each metric, arranged horizontally
        # Increase figure height to make more room for title
        fig, axes = plt.subplots(
            1,
            len(metrics),
            figsize=(8 * len(metrics), 6),  # Increased height from 5 to 6
            sharey=False,  # Don't share y-axis to allow each plot to have its own y-axis label
            constrained_layout=True
        )
        
        # If there's only one subplot, make sure axes is an array
        if len(metrics) == 1:
            axes = np.array([axes])
        
        # Use a single color for all boxes
        prompt_versions = sorted(self.get_prompts())
        # Create a list with the same color repeated for each prompt version
        single_color = 'gray'  # Dark gray color
        palette = [single_color] * len(prompt_versions)
        
        # Iterate through metrics to create the plots
        for i, metric in enumerate(metrics):
            # Get the current axis
            ax = axes[i]
            
            # Create boxplot for this metric with same color for all boxes
            sns.boxplot(
                x='prompt_version',
                y=metric,
                hue='prompt_version',  # Keep hue parameter to avoid FutureWarning
                data=plot_data,
                ax=ax,
                palette=palette,
                width=0.6,
                order=prompt_versions,
                linewidth=1.0,
                legend=False,  # Hide legend
                fliersize=3,
                flierprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markeredgewidth': 0.8}
            )
            
            # Add grid lines for better readability
            ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        
            # Set titles and labels with academic styling
            ax.set_title(f"{metric.capitalize()}", fontweight='bold')
            
            # Set y-label for each plot with the actual metric name
            ax.set_ylabel(f"{metric.capitalize()}", fontweight='bold')
                
            ax.set_xlabel("Prompt Strategy", fontweight='bold')
            
            # Remove top and right spines for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Set y-axis limits for consistency
            ax.set_ylim(0, 1)
        
            # Add statistical significance annotations
            # Get x-coordinates for each prompt version
            x_coords = {prompt: idx for idx, prompt in enumerate(prompt_versions)}
            
            # Add significance bars and annotations
            bar_height = 0.02  # Height of significance bars
            text_height = 0.01  # Height of p-value text
            text_buffer = 0.04  # Buffer between bars and text to avoid overlap
            max_bars = len(prompt_versions) * (len(prompt_versions) - 1) // 2
            
            # Calculate y positions for significance bars
            y_max = ax.get_ylim()[1]
            bar_positions = np.linspace(
                y_max, 
                y_max + (bar_height + text_height + text_buffer) * max_bars, 
                max_bars
            )
            
            bar_idx = 0
            for p_i, prompt1 in enumerate(prompt_versions):
                for _, prompt2 in enumerate(prompt_versions[p_i+1:], p_i+1):
                    comparison_key = f"{prompt1}_vs_{prompt2}"
                    if comparison_key in stat_results[metric]:
                        result = stat_results[metric][comparison_key]
                        p_value = result.p_value
                        effect_size = result.effect_size
                    
                    # Determine significance level
                    if p_value < 0.001:
                        sig_symbol = '***'
                    elif p_value < 0.01:
                        sig_symbol = '**'
                    elif p_value < 0.05:
                        sig_symbol = '*'
                    else:
                        sig_symbol = ''
                    
                    # Get x positions
                    x1, x2 = x_coords[prompt1], x_coords[prompt2]
                    y = bar_positions[bar_idx]
                    
                    # Draw the bar with academic styling
                    ax.plot([x1, x2], [y, y], 'k-', linewidth=0.8)
                    ax.plot([x1, x1], [y-bar_height/2, y], 'k-', linewidth=0.8)
                    ax.plot([x2, x2], [y-bar_height/2, y], 'k-', linewidth=0.8)
                    
                    # Add p-value and effect size text with academic styling
                    sig_value = r"$^{" + sig_symbol + r"}$"
                    ax.text(
                        (x1+x2)/2, 
                        y + text_height*2,
                        f"p={p_value:.3f}{sig_value}, d={effect_size:.2f}",
                        ha='center', 
                        va='bottom', 
                        color="black",
                        fontsize=8, 
                        fontstyle='italic'
                    )
                    
                    bar_idx += 1
            
            # Adjust y-axis limits to accommodate significance bars
            if bar_idx > 0:
                ax.set_ylim(0.0, bar_positions[bar_idx-1] + text_height * 3)
        
        # Add overall title with academic styling
        if dataset is None:
            title = ""#"Performance Metrics by Prompting Strategy"
            filename = "strategy_boxplot_combined.pdf"
        else:
            title = f"Performance Metrics for Dataset {dataset} by Prompting Strategy"
            filename = f"strategy_boxplot_dataset_{dataset}.pdf"
        
        # Add title to the figure
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.05)
        
        # Adjust layout and save
        plt.tight_layout()
        full_dir = Directory.OUTPUT_DIR / 'strategy_boxplots'
        full_dir.mkdir(exist_ok=True)
        fig.savefig(full_dir / filename, bbox_inches='tight')

        # save as png as well
        filename_png = filename.split('.')[0] + ".png"
        fig.savefig(full_dir / filename_png, bbox_inches='tight', dpi=300)
        
        # Show the plot
        plt.close()
        
        # Now create individual plots for each metric
        for metric in metrics:
            self.plot_metric(metric, dataset)
        
    def plot_metric(self, metric, dataset=None):
        """
        Create a boxplot for a single metric.
        
        Args:
            metric: The metric to plot (precision, recall, or F1)
            dataset: Optional dataset to filter by. If None, plots all datasets combined.
        """
        # Filter data if a specific dataset is requested
        plot_data = self.data
        if dataset is not None:
            plot_data = self.data[self.data['dataset'] == dataset]
        
        # Perform statistical tests
        stat_results = self.perform_statistical_tests()
        
        # Create the plot
        self._create_plot_for_metric(metric, plot_data, stat_results, dataset)
        
        plt.close()
    
    def plot_all_datasets(self):
        """Plot all datasets combined in one figure"""
        # Plot all datasets in one figure
        self.plot()

        datasets = self.get_datasets()

        for dataset in datasets:
            self.plot(dataset=dataset)
            
            # Also plot individual metrics for each dataset
            for metric in self.get_metrics():
                self.plot_metric(metric, dataset)


if __name__ == '__main__':

    import pandas as pd
    from itertools import product
    import numpy as np
    from src.experiments.visualization.precision_boxplot import PrecisionBoxPlot
    from src.experiments.visualization.recall_boxplot import RecallBoxPlot
    from src.experiments.visualization.f1_boxplot import F1BoxPlot

    datasets = ["A", "B", "C"]
    models = ["GPT-3.5", "GPT-4", "Llama 2", "Claude 2", "Mistral"]
    prompt_versions = ["osr", "ad", "nd", "od"]
    metrics = ["precision", "recall", "F1"]

    # Set random seed for reproducibility
    np.random.seed(42)

    # Creating all possible combinations with more realistic differences between strategies
    data = []
    for dataset, model, prompt in product(datasets, models, prompt_versions):
        # Base values with strategy-specific biases to ensure statistical differences
        if prompt == "osr":
            base_precision = 0.78
            base_recall = 0.72
        elif prompt == "ad":
            base_precision = 0.82
            base_recall = 0.74
        else:  # nd
            base_precision = 0.76
            base_recall = 0.76
            
        # Add random noise
        precision = round(base_precision + np.random.normal(0, 0.02), 2)
        recall = round(base_recall + np.random.normal(0, 0.02), 2)
        
        # Ensure values are in valid range
        precision = max(0.65, min(0.95, precision))
        recall = max(0.65, min(0.95, recall))
        
        # Calculate F1
        f1 = round((2 * precision * recall) / (precision + recall), 2)
        
        data.append([dataset, model, prompt, precision, recall, f1])

    # Creating the DataFrame
    df = pd.DataFrame(data, columns=["dataset", "model", "prompt_version", "precision", "recall", "F1"])
    
    # Create and use the combined plot
    print("Creating combined plots...")
    strategy_boxplot = StrategyBoxPlot(data=df)
    strategy_boxplot.plot_all_datasets()
    
    # Create and use individual metric plots
    print("Creating precision plots...")
    precision_boxplot = PrecisionBoxPlot(data=df)
    precision_boxplot.plot_all_datasets()
    
    print("Creating recall plots...")
    recall_boxplot = RecallBoxPlot(data=df)
    recall_boxplot.plot_all_datasets()
    
    print("Creating F1 plots...")
    f1_boxplot = F1BoxPlot(data=df)
    f1_boxplot.plot_all_datasets()
