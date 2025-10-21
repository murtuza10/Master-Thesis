import matplotlib.pyplot as plt
import numpy as np

# Your model data - replace with actual time values in seconds
models = ['XLM - Roberta - Large',
'Scibert',
'Agribert']

# Example time values in seconds - replace with your actual data
time_values = [14783,
7702,
18304]

# Normalize sizes for visualization with better scaling
max_time = max(time_values)
min_time = min(time_values)

# Calculate circle radii (smaller scale to fit within bounds)
def scale_radius(value):
    if value == max_time:
        return 0.18  # Maximum radius for largest value
    elif value == min_time:
        return 0.03  # Minimum radius for smallest value
    else:
        # Scale between min and max
        ratio = (value - min_time) / (max_time - min_time)
        return 0.03 + (ratio * 0.15)

radii = [scale_radius(t) for t in time_values]

# Create figure with blue background
fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor('#1F5F8B')
ax.set_facecolor('#1F5F8B')

# Create bubble chart with adjusted positions for 3 items
positions = [0.25, 0.5, 0.75]

for i, (pos, radius, time, model) in enumerate(zip(positions, radii, time_values, models)):
    # Draw circle
    circle = plt.Circle((pos, 0.5), radius, color='white', 
                        edgecolor='white', linewidth=3, fill=False)
    ax.add_patch(circle)
    
    # Add time value above circle
    ax.text(pos, 0.5 + radius + 0.05, f'{time:,}', 
            ha='center', va='bottom', fontsize=18, 
            color='white', fontweight='bold')
    
    # Add model name below circle
    ax.text(pos, 0.5 - radius - 0.05, model, 
            ha='center', va='top', fontsize=12, 
            color='white', multialignment='center')

# Set axis limits with padding
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout(pad=1)
plt.savefig('model_time_comparison.png', dpi=300, bbox_inches='tight', 
           facecolor='#1F5F8B', edgecolor='none', pad_inches=0.2)
plt.show()

print("✅ Time comparison plot saved as 'model_time_comparison.png'")