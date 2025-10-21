import matplotlib.pyplot as plt
import numpy as np

# Your data
models = ['XLM - Roberta - Large',
'Scibert',
'Agribert']
f1_scores = [0.81,
0.79,
0.75]

# Single teal/blue color matching the image
bar_color = '#1F5F8B'  # Deep teal blue from the image

# Create figure with clean white background
plt.style.use('default')
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Create horizontal bars with the teal color
bars = ax.barh(models, f1_scores, color=bar_color, edgecolor='white', 
               linewidth=1.5, alpha=0.9, height=0.7)

# Customize labels
ax.set_xlabel('Exact F1 Score', fontsize=14, fontweight='normal', color='#333333', labelpad=10)
ax.set_ylabel('Model Name', fontsize=14, fontweight='normal', color='#333333', labelpad=10)
ax.set_title('Encoder Model Performance Comparison - Exact F1 Scores', 
             fontsize=16, fontweight='bold', color='#333333', pad=20)

# Adjust x-axis
ax.set_xlim(0, 0.90)
plt.yticks(fontsize=11, color='#333333')
plt.xticks(fontsize=11, color='#333333')

# Add subtle grid
ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.8, color='#CCCCCC')
ax.set_axisbelow(True)

# Add value labels at the end of bars
for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
            f'{score:.4f}',
            ha='left', va='center', fontsize=10, 
            color='#333333', fontweight='normal')

# Customize spines - minimal style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)

# Reverse the y-axis to match the image (highest at top)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('encoder_model_f1_comparison.png', dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
plt.show()

print("✅ Plot saved as 'model_f1_comparison.png'")
