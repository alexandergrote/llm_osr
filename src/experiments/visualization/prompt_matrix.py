import matplotlib.pyplot as plt

from pydantic import BaseModel
from typing import List

from src.util.constants import Directory


class PromptMatrix(BaseModel):

    top_left: str  # implicit zeroshot
    top_right: str  # implicit fewshot
    bottom_left: str  # explicit zeroshot
    bottom_right: str  # explicit fewshot

    def to_matrix(self) -> List[List[str]]:
        return [
            [self.top_left, self.top_right],
            [self.bottom_left, self.bottom_right]
        ]

class PromptMatrixPlot(BaseModel):

    prompt_matrix: PromptMatrix
    
    def plot(self) -> None:

        matrix = self.prompt_matrix.to_matrix()
        assert len(matrix) == 2, "Matrix must be 2x2"
        assert len(matrix[0]) == 2, "Matrix must be 2x2"
        assert len(matrix[1]) == 2, "Matrix must be 2x2"

        fig, ax = plt.subplots(figsize=(16, 4))

        # Hide axes
        ax.axis('off')

        # Set limits
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)

        # Add text to each cell with wrapping
        for i in range(2):
            for j in range(2):
                text_box = ax.text(
                    j + 0.5, 1.5 - i, 
                    matrix[i][j].strip(), 
                    ha='center', 
                    va='center', 
                    fontsize=14, 
                    #bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="gray"),
                    wrap=True
                )
                # Set the width for text wrapping (adjust as needed)
                text_box.set_linespacing(1.)  # Increase line spacing for wrapped text
                text_box._get_wrap_line_width = lambda: 460  # Adjust this value to control wrap width

        # Draw grid
        for x in range(3):
            ax.axvline(x, color='black', lw=1)
            ax.axhline(x, color='black', lw=1)

        # Add x-axis labels
        ax.text(0.5, -0.1, "Zero-shot", ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(1.5, -0.1, "Few-shot", ha='center', va='center', fontsize=14, fontweight='bold')

        # Add y-axis labels
        ax.text(-0.025, 1.5, "Implicit", ha='center', va='center', fontsize=14, fontweight='bold', rotation=90)
        ax.text(-0.025, 0.5, "Explicit", ha='center', va='center', fontsize=14, fontweight='bold', rotation=90)

        # Save the plot
        plt.tight_layout()
        plt.savefig(Directory.OUTPUT_DIR / "text_matrix_plot.pdf", bbox_inches='tight', dpi=300)

if __name__ == "__main__":

    top_left = """Given these examples and their classes, which class does "{text}" belong to? If unsure, answer {unknown_label}. \n\n{examples} {chain_of_thought} {classes} {json_instructions}"""
    top_right = """Given these examples and their classes, which class does "{text}" belong to? If unsure, answer {unknown_label}. {examples} {chain_of_thought} {classes} {json_instructions}"""

    bottom_left = """Given these examples, is this data point "{text}" an outlier? {examples} {chain_of_thought} {classes} {json_instructions}"""
    bottom_right = """Given these examples, is this data point "{text}" an outlier? {examples} {chain_of_thought} {classes} {json_instructions}"""

    bottom_right = """Given these examples and their classes, is this data point "What is the capital of France?" an outlier that differs in its intent?

    The president signed a new law. -> inlier
    The final score was 3-1. -> inlier
    The cake is delicious. -> outlier
    It's raining heavily. -> inlier 
    """

    bottom_left = """Given these examples and their classes, is this data point "What is the capital of France?" an outlier that differs in its intent?

    The final score was 3-1. -> inlier
    It's raining heavily. -> inlier
    The president signed a new law. -> inlier 
    """

    top_right = """Given these examples and their classes, which class does "What is the capital of France?" belong to?
    If unsure, answer unknown. 

    The president signed a new law. -> politics
    The final score was 3-1. -> sports
    The cake is delicious. -> unknown
    It's raining heavily. -> weather
    """

    top_left = """Given these examples and their classes, which class does "What is the capital of France?" belong to?
    If unsure, answer unknown. 

    The final score was 3-1. -> sports
    It's raining heavily. -> weather
    The president signed a new law. -> politics
    """
    
    
    prompt_matrix = {
        "top_left": top_left,
        "top_right": top_right,
        "bottom_left": bottom_left,
        "bottom_right": bottom_right
    }

    plot_obj = PromptMatrixPlot(
        prompt_matrix=prompt_matrix
    )

    plot_obj.plot()
