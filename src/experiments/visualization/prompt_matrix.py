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

fig, ax = plt.subplots()

# Hide axes
ax.axis('off')

# Set limits
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)

# Add text to each cell
for i in range(2):
    for j in range(2):
        ax.text(j + 0.5, 1.5 - i, matrix[i][j], ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="gray"))

# Draw grid
for x in range(3):
    ax.axvline(x, color='black', lw=1)
    ax.axhline(x, color='black', lw=1)

# Save the plot
plt.savefig("text_matrix_plot.png", bbox_inches='tight')
plt.show()
