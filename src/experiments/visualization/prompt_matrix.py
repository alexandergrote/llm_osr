import matplotlib.pyplot as plt

# Define the 2x2 text matrix
matrix = [
    ["Top Left", "Top Right"],
    ["Bottom Left", "Bottom Right"]
]

top_left = """Given these examples, is this data point "{text}" an outlier? {examples} {chain_of_thought} {classes} {json_instructions}"""

matrix = [
    [top_left, "Top Right"],
    ["Bottom Left", "Bottom Right"]
]

fig, ax = plt.subplots(figsize=(10, 8))

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
            matrix[i][j], 
            ha='center', 
            va='center', 
            fontsize=12, 
            bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="gray"),
            wrap=True
        )
        # Set the width for text wrapping (adjust as needed)
        text_box.set_linespacing(1.5)  # Increase line spacing for wrapped text
        text_box._get_wrap_line_width = lambda: 200  # Adjust this value to control wrap width

# Draw grid
for x in range(3):
    ax.axvline(x, color='black', lw=1)
    ax.axhline(x, color='black', lw=1)

# Add x-axis labels
ax.text(0.5, -0.1, "Zero-shot", ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(1.5, -0.1, "Few-shot", ha='center', va='center', fontsize=14, fontweight='bold')

# Add y-axis labels
ax.text(-0.1, 1.5, "Implicit", ha='center', va='center', fontsize=14, fontweight='bold', rotation=90)
ax.text(-0.1, 0.5, "Explicit", ha='center', va='center', fontsize=14, fontweight='bold', rotation=90)

# Save the plot
plt.savefig("text_matrix_plot.png", bbox_inches='tight')
plt.tight_layout()
plt.show()
