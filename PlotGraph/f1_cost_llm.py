import matplotlib.pyplot as plt
import numpy as np

# Your 8 models data
models = ['gpt-5',
'Qwen2.5-32B-Instruct',
'DeepSeekV3',
'Qwen2.5-14B-Instruct',
'Qwen2.5-7B-Instruct',
'Qwen2.5-72B-Instruct',
'Llama-3.3-70B-Instruct',
'Llama-3.1-8B-Instruct']
f1_scores = [0.5983615844,
0.5511211639,
0.525220418,
0.4968471643,
0.4799931612,
0.473688528,
0.4254672521,
0.3842925905]

costs = [0.14,
0.0013,
0.041,
0.00062,
0.0003,
0.00232,
0.00262,
0.00056]  # Cost values (in Euros)

# Color scheme
colors = ['#7BAFCF', '#A8B862', '#E89B5A','#D96B3C','#9B6BA8','#E87B9B','#5BAEA8','#C4D65A']

# Create figure with blue background
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('#2B5F87')
ax.set_facecolor('#2B5F87')

# Create scatter plot with larger markers and legend
scatter_points = []
for i, (model, f1, cost) in enumerate(zip(models, f1_scores, costs)):
    scatter = ax.scatter(f1, cost, s=600, color=colors[i], 
                        edgecolor='white', linewidth=3, alpha=0.9, 
                        zorder=3, label=model)
    scatter_points.append(scatter)

# Customize axes with white text
ax.set_xlabel('Exact F1 Score', fontsize=14, fontweight='bold', color='white', labelpad=10)
ax.set_ylabel('Cost (Euros)', fontsize=14, fontweight='bold', color='white', labelpad=10)
ax.set_title('Large Language Models Performance vs Cost Comparison', fontsize=16, fontweight='bold', 
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

# Add legend with custom styling
legend = ax.legend(loc='upper left', fontsize=11, framealpha=0.95, 
                  facecolor='#2B5F87', edgecolor='white', labelcolor='white',
                  frameon=True, shadow=False, ncol=1, borderpad=1)
legend.get_frame().set_linewidth(2)

# Set limits with padding
ax.set_xlim(min(f1_scores) - 0.05, max(f1_scores) + 0.05)
ax.set_ylim(min(costs) - 0.01, max(costs) + 0.01)

plt.tight_layout()
plt.savefig('llm_f1_vs_cost.png', dpi=300, bbox_inches='tight', facecolor='#2B5F87')
plt.show()

print("✅ F1 vs cost plot saved as 'llm_f1_vs_cost.png'")
