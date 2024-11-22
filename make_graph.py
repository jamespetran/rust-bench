import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

plt.style.use('bmh')

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
   'Deepseek Chat',
   'Phi 3.5 Mini',
   'LFM 40B'
]

first_attempts = np.round(np.array([
   66.3, 48.2, 38.6, 58.1, 39.8,  # OpenAI
   65.9, 55.6,                     # Anthropic
   51.8, 27.3,                     # Google
   24.1,                           # X.AI
   35.3, 45.0, 36.1, 20.1,        # Open Source (first part)
   27.7, 18.6, 35.5,              # Open Source (second part)
   0.6, 4.2                        # New models
]), 1)

second_attempts = np.round(np.array([
   85.5, 73.5, 58.3, 79.3, 55.8,  # OpenAI
   81.9, 75.6,                     # Anthropic
   72.3, 46.9,                     # Google
   42.2,                           # X.AI
   51.4, 59.0, 62.7, 42.2,        # Open Source (first part)
   42.8, 24.2, 52.4,              # Open Source (second part)
   1.2, 4.2                        # New models
]), 1)

# Sort by second attempts
sort_idx = np.argsort(second_attempts)
names = [names[i] for i in sort_idx]
first_attempts = first_attempts[sort_idx]
second_attempts = second_attempts[sort_idx]
improvements = np.round(second_attempts - first_attempts, 1)

first_color = '#4C72B0'    
improvement_color = '#55A868'  

fig, ax = plt.subplots(figsize=(12, 10))
bars1 = ax.barh(names, first_attempts, label='First Attempt', color=first_color)
bars2 = ax.barh(names, improvements, left=first_attempts, 
                label='Correction', color=improvement_color)

ax.set_xlabel('Scores', fontsize=12)
ax.set_title('Model Scores: First Attempt and Correction', fontsize=14, weight='bold')
ax.legend(fontsize=12)
ax.set_xlim(0, 100)
ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
ax.set_yticklabels(names, fontsize=10)

def add_labels(bars, values, lefts=None, color='black'):
    for i, (bar, val) in enumerate(zip(bars, values)):
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        
        # Special handling for very small values (last two bars)
        if i < 2:  # For the last two bars
            x = 1.5*width + (1 if lefts is not None else 0)*3 + 5 # Position slightly to the right of the bar
            ha = 'right'
            color = 'black'
        else:
            if lefts is not None:
                x = lefts[i] + width/2
                ha = 'center'
            else:
                x = bar.get_x() + width/2
                ha = 'center'
            
        ax.text(x, y, f'{val:.1f}', va='center', ha=ha, fontsize=11, color=color)

add_labels(bars1, first_attempts, color='white')
add_labels(bars2, second_attempts, first_attempts, color='black')

plt.tight_layout()
plt.show()