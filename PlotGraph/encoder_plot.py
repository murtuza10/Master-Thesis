import matplotlib.pyplot as plt
import numpy as np

# Data for the chart
groups = ['XLM-Roberta-Large', 'Agribert', 'Scibert']
x = np.linspace(0, 2, len(groups))  # wider spacing between groups
width = 0.13  # clean, slimmer bars

# Data values
english_broad = [0.75, 0.73, 0.74]
english_specific = [0.74, 0.71, 0.72]
german_english_specific = [0.74, 0.66, 0.63]
german_english_broad = [0.77, 0.61, 0.63]
german_english_no_soil = [0.81, 0.75, 0.79]

# Colors
colors = ['#5B9DB8', '#B8A839', '#E8B456', '#E85D3D', '#A0522D']

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Bar plotting setup
offsets = [-2*width, -width, 0, width, 2*width]
datasets = [
    english_broad,
    english_specific,
    german_english_specific,
    german_english_broad,
    german_english_no_soil
]
labels = [
    'English Broad',
    'English Specific',
    'German English Specific',
    'German English Broad',
    'German English No Soil'
]

bars = []
for i in range(len(datasets)):
    bars.append(
        ax.bar(x + offsets[i], datasets[i], width,
               label=labels[i],
               color=colors[i],
               alpha=0.92,
               edgecolor='white',
               linewidth=1.1)
    )

# Value labels
for group in bars:
    for bar in group:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.2f}',
            ha='center', va='bottom',
            fontsize=10, color='white',
            bbox=dict(boxstyle='round,pad=0.25',
                      facecolor='#0E4A80',
                      edgecolor='white',
                      linewidth=1)
        )

# # Title
# ax.set_title(
#     'Encoder Model Comparison (F1 Scores)',
#     fontsize=18,
#     fontweight='bold',
#     color='white',
#     pad=25
# )

# Legend (Option 2: directly under title)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    fontsize=11,
    labelcolor='white'
)

# Axes labels and ticks
ax.set_ylabel('F1 Score', fontsize=13, color='white')
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=12, color='white', rotation=8)

ax.set_ylim(0, 1.0)

# Background & grid
fig.patch.set_facecolor('#0E4A80')
ax.set_facecolor('#0E4A80')
ax.grid(axis='y', linestyle='--', alpha=0.30, linewidth=0.8, color='white')

# Spines
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color('white')
    ax.spines[spine].set_linewidth(1.4)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
plt.savefig('encoder_model_comparison.png', dpi=300, bbox_inches='tight', facecolor='#0E4A80')