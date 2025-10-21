import matplotlib.pyplot as plt
import numpy as np

# Your 3 models data - replace with your actual values
models = ['XLM - Roberta - Large',
'Scibert',
'Agribert']

# Example data - replace with your actual values
f1_scores = [0.81, 0.79, 0.75]  # F1 scores for the 3 models
times = [14783,
7702,
18304]  # Power values for the 3 models (e.g., in kWh)

# Color scheme from the image
# Light blue, Yellow-green, Orange
colors = ['#7BAFCF', '#A8B862', '#E89B5A']

# Create figure with blue background matching the image
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('#2B5F87')
ax.set_facecolor('#2B5F87')

# Create scatter plot with larger markers
for i, (model, f1, time) in enumerate(zip(models, f1_scores, times)):
    ax.scatter(f1, time, s=600, color=colors[i], 
               edgecolor='white', linewidth=3, alpha=0.9, zorder=3)
    
    # Add model name as label
    ax.annotate(model.replace('\n', ' '), 
                xy=(f1, time), 
                xytext=(10, 10), 
                textcoords='offset points',
                fontsize=12, 
                fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.6', 
                         facecolor=colors[i], 
                         edgecolor='white', 
                         linewidth=2,
                         alpha=0.95))

# Customize axes with white text
ax.set_xlabel('Exact F1 Score', fontsize=14, fontweight='bold', color='white', labelpad=10)
ax.set_ylabel('Time (seconds)', fontsize=14, fontweight='bold', color='white', labelpad=10)
ax.set_title('Model Performance vs Time Comparison', fontsize=16, fontweight='bold', 
             color='white', pad=20)

# Add grid with lighter blue color
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1, color='white', zorder=0)
ax.set_axisbelow(True)

# Customize spines with white
for spine in ax.spines.values():
    spine.set_edgecolor('white')
    spine.set_linewidth(1.5)

# Customize tick labels to white
ax.tick_params(colors='white', labelsize=11)

# Set limits with padding
ax.set_xlim(min(f1_scores) - 0.05, max(f1_scores) + 0.05)
ax.set_ylim(min(times) - 2000, max(times) + 2000)
plt.tight_layout()
plt.savefig('encoder_f1_vs_time.png', dpi=300, bbox_inches='tight', facecolor='#2B5F87')
plt.show()

print("✅ F1 vs cost plot saved as 'encoder_f1_vs_time.png'")