import numpy as np
import pandera as pa
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import List, Dict, Tuple, NamedTuple

from pandera.typing import DataFrame, Series
from pydantic import BaseModel, ConfigDict


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
        results = {}
        
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

    def plot(self):
        metrics = self.get_metrics()
        
        # Perform statistical tests
        stat_results = self.perform_statistical_tests()
        
        # Set the seaborn style
        sns.set(style="whitegrid")
        
        # Create a figure with one subplot per metric
        fig, axes = plt.subplots(
            len(metrics), 
            1, 
            figsize=(10, 5 * len(metrics)),  # Increase height to accommodate significance annotations
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
                hue='prompt_version',  # Add hue parameter to fix FutureWarning
                data=self.data,
                ax=ax,
                palette=palette,
                width=0.7,
                order=sorted(prompt_versions),
                legend=False  # Hide the legend since it's redundant with x-axis
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
            
            # Add statistical significance annotations
            prompt_versions = sorted(self.get_prompts())
            
            # Get x-coordinates for each prompt version
            x_coords = {prompt: idx for idx, prompt in enumerate(prompt_versions)}
            
            # Add significance bars and annotations
            bar_height = 0.02  # Height of significance bars
            text_height = 0.01  # Height of p-value text
            max_bars = len(prompt_versions) * (len(prompt_versions) - 1) // 2
            
            # Calculate y positions for significance bars
            y_max = ax.get_ylim()[1]
            bar_positions = np.linspace(y_max, y_max + (bar_height + text_height) * max_bars, max_bars)
            
            bar_idx = 0
            for i, prompt1 in enumerate(prompt_versions):
                for j, prompt2 in enumerate(prompt_versions[i+1:], i+1):
                    comparison_key = f"{prompt1}_vs_{prompt2}"
                    if comparison_key in stat_results[metric]:
                        _, p_value = stat_results[metric][comparison_key]
                        
                        result = stat_results[metric][comparison_key]
                        p_value = result.p_value
                        effect_size = result.effect_size
                        effect_interp = result.effect_size_interpretation
                        
                        # Determine significance level
                        if p_value < 0.001:
                            sig_symbol = '***'
                        elif p_value < 0.01:
                            sig_symbol = '**'
                        elif p_value < 0.05:
                            sig_symbol = '*'
                        else:
                            sig_symbol = 'ns'
                        
                        # Get effect size color based on interpretation
                        if effect_interp == "large":
                            effect_color = "darkred"
                        elif effect_interp == "medium":
                            effect_color = "darkorange"
                        elif effect_interp == "small":
                            effect_color = "darkgreen"
                        else:  # negligible
                            effect_color = "darkgray"
                        
                        # Only draw significance bars for significant results
                        if p_value < 0.05:
                            # Get x positions
                            x1, x2 = x_coords[prompt1], x_coords[prompt2]
                            y = bar_positions[bar_idx]
                            
                            # Draw the bar
                            ax.plot([x1, x2], [y, y], 'k-', linewidth=1.5)
                            ax.plot([x1, x1], [y-bar_height/2, y], 'k-', linewidth=1.5)
                            ax.plot([x2, x2], [y-bar_height/2, y], 'k-', linewidth=1.5)
                            
                            # Add significance symbol
                            ax.text((x1+x2)/2, y + text_height/2, sig_symbol, 
                                   ha='center', va='bottom', color='black', fontsize=12)
                            
                            # Add p-value and effect size text
                            ax.text((x1+x2)/2, y + text_height*2, 
                                   f"p={p_value:.3f}, d={effect_size:.2f}", 
                                   ha='center', va='bottom', color=effect_color, 
                                   fontsize=9, style='italic')
                            
                            bar_idx += 1
            
            # Adjust y-axis limits to accommodate significance bars
            if bar_idx > 0:
                ax.set_ylim(0.65, bar_positions[bar_idx-1] + text_height * 2)
        
        # Add overall title
        plt.suptitle("Performance Metrics by Prompting Strategy", fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("strategy_boxplot.pdf")
        plt.show()


if __name__ == '__main__':
    import pandas as pd
    from itertools import product
    import numpy as np

    datasets = ["A", "B"]
    models = ["GPT-3.5", "GPT-4", "Llama 2", "Claude 2", "Mistral"]
    prompt_versions = ["osr", "ad", "nd"]
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
    
    strategy_boxplot = StrategyBoxPlot(data=df)
    strategy_boxplot.plot()
