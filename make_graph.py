import matplotlib.pyplot as plt
import numpy as np

# Data
names = ['Mary', 'Alphonse', 'Ian']
first_attempts = np.array([23, 55, 34])
second_attempts = np.array([34, 55, 78])
improvements = second_attempts - first_attempts

# Create the bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Plot first attempts
bars1 = ax.bar(names, first_attempts, label='First Attempt', color='skyblue')

# Plot improvements on top
bars2 = ax.bar(names, improvements, bottom=first_attempts, 
               label='Improvement', color='lightgreen')

# Customize the chart
ax.set_ylabel('Scores')
ax.set_title('Student Scores: First Attempt vs Improvement')
ax.legend()

# Add value labels on the bars
def add_labels(bars, values):
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_y() + height/2.,
                f'{val}',
                ha='center', va='center')

add_labels(bars1, first_attempts)
add_labels(bars2, improvements)

# Add total values at the top
for i, total in enumerate(second_attempts):
    ax.text(i, total + 1, f'Total: {total}', ha='center')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()