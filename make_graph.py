import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
# Optional: Use seaborn style for better aesthetics
plt.style.use('bmh')
# Data
names = [
    # Closed Source Models
    'O1 Preview',
    'O1 Mini',
    'GPT-4O Mini',
    'ChatGPT-4O',
    'GPT-3.5 Turbo',
    'Claude 3.5 Sonnet',
    'Claude 3.5 Haiku',
    'Gemini Pro 1.5',
    'Gemini Flash 1.5',
    'Grok Beta',
    # Open Source Models
    'Qwen 2.5 Coder',
    'Hermes 3 Llama',
    'Llama 3.1 Instruct',
    'Llama 70B Instruct',
    'Mistral Large',
    'Codestral Mamba',
    'Deepseek Chat'
]
# Round all numbers to 1 decimal place
first_attempts = np.round(np.array([
    66.3, 47.6, 39.8, 57.6, 40.7,  # OpenAI
    65.9, 58.4,                     # Anthropic
    51.8, 30.7,                     # Google
    24.1,                           # X.AI
    33.3, 45.7, 36.1, 20.3,        # Open Source (first part)
    27.7, 18.6, 36.0               # Open Source (second part)
]), 1)
second_attempts = np.round(np.array([
    85.5, 73.5, 57.8, 79.3, 56.8,  # OpenAI
    81.9, 76.3,                     # Anthropic
    72.3, 43.8,                     # Google
    42.2,                           # X.AI
    49.2, 59.3, 62.7, 40.6,        # Open Source (first part)
    42.8, 24.2, 51.8               # Open Source (second part)
]), 1)
# Sort by second attempts
sort_idx = np.argsort(second_attempts)
names = [names[i] for i in sort_idx]
first_attempts = first_attempts[sort_idx]
second_attempts = second_attempts[sort_idx]
improvements = np.round(second_attempts - first_attempts, 1)
# Define colors
first_color = '#4C72B0'    # A pleasing blue
improvement_color = '#55A868'  # A complementary green
# Create the horizontal bar chart
fig, ax = plt.subplots(figsize=(12, 10))
# Plot first attempts
bars1 = ax.barh(names, first_attempts, label='First Attempt', color=first_color)
# Plot improvements
bars2 = ax.barh(names, improvements, left=first_attempts, 
                label='Correction', color=improvement_color)
# Customize the chart
ax.set_xlabel('Scores', fontsize=12)
ax.set_title('Model Scores: First Attempt and Correction', fontsize=14, weight='bold')
ax.legend(fontsize=12)
# Set x-axis limit to 100
ax.set_xlim(0, 100)
# Add grid lines for x-axis
ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
# Adjust y-axis labels
ax.set_yticklabels(names, fontsize=10)
# Add value labels on the bars
def add_labels(bars, values, lefts=None, color='black'):
    for i, (bar, val) in enumerate(zip(bars, values)):
        width = bar.get_width()
        if lefts is not None:
            # Position label inside the improvement bar
            x = lefts[i] + width/2
            ha = 'center'
        else:
            x = bar.get_x() + width / 2
            ha = 'center'
        y = bar.get_y() + bar.get_height() / 2
        ax.text(x, y, f'{val:.1f}', va='center', ha=ha, fontsize=11, color=color)
add_labels(bars1, first_attempts, color='white')
add_labels(bars2, second_attempts, first_attempts, color='black')
# Improve layout to prevent label cutoff
plt.tight_layout()
# Show the plot
plt.show()