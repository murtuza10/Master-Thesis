import matplotlib.pyplot as plt
import numpy as np

# Set the color scheme
bg_color = '#1F5F8B'
text_color = 'white'
grid_color = '#2D7BA8'

# Model colors matching your reference
colors = {
    'Qwen2.5-7B-Instruct': '#A9B82E',
    'Qwen2.5-14B-Instruct': '#E67E22',
    'Qwen2.5-32B-Instruct': '#5DADE2',
    'Qwen2.5-72B-Instruct': '#E74C3C',
    'Llama-3.1-8B-Instruct': '#95A5A6',
    'Llama-3.3-70B-Instruct': '#E91E63',
    'DeepSeekV3': '#9B59B6',
    'GPT-5': '#F39C12'
}

# Data from Five Shot Prompting Embeddings Specific table
models = ['DeepSeekV3', 'GPT-5', 'Llama-3.1-8B-Instruct', 'Llama-3.3-70B-Instruct',
          'Qwen2.5-14B-Instruct', 'Qwen2.5-32B-Instruct', 'Qwen2.5-72B-Instruct', 'Qwen2.5-7B-Instruct']

f1_scores = [0.536022, 0.507546, 0.332936, 0.377012, 0.457524, 0.527633, 0.490441, 0.440615]
precision = [0.702855, 0.627604, 0.586264, 0.676326, 0.732071, 0.70773, 0.635107, 0.609432]
recall = [0.450658, 0.4375, 0.241611, 0.273026, 0.348684, 0.434211, 0.417763, 0.352273]
time_seconds = [1050, 179, 876, 5527, 1013, 2312, 4788, 645]
costs = [0.0250346, 0.1445, 0.0006, 0.0034, 0.00064, 0.00136, 0.00242, 0.00036]
power = [None, None, 0.03, 0.17, 0.032, 0.068, 0.121, 0.018]

model_colors = [colors[m] for m in models]

# Chart 1: F1 Scores Comparison (sorted)
fig1, ax1 = plt.subplots(figsize=(12, 8), facecolor=bg_color)
ax1.set_facecolor(bg_color)

sorted_indices = np.argsort(f1_scores)
sorted_models = [models[i] for i in sorted_indices]
sorted_f1 = [f1_scores[i] for i in sorted_indices]
sorted_colors = [model_colors[i] for i in sorted_indices]

bars1 = ax1.barh(sorted_models, sorted_f1, color=sorted_colors, edgecolor=text_color, linewidth=1.5)
ax1.set_xlabel('F1 Score', color=text_color, fontsize=14, fontweight='bold')
ax1.set_title('F1 Score Comparison - Five Shot Prompting Embeddings Specific', 
              color=text_color, fontsize=16, fontweight='bold', pad=20)
ax1.tick_params(colors=text_color, labelsize=11)
ax1.grid(axis='x', color=grid_color, alpha=0.3, linestyle='--')
ax1.set_xlim(0.3, 0.56)

for i, (bar, val) in enumerate(zip(bars1, sorted_f1)):
    ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
             va='center', color=text_color, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('f1_scores_comparison.png', dpi=300, facecolor=bg_color)
print("Chart 1 saved: f1_scores_comparison.png")

# Chart 2: Cost Comparison (logarithmic scale)
fig2, ax2 = plt.subplots(figsize=(12, 8), facecolor=bg_color)
ax2.set_facecolor(bg_color)

bars2 = ax2.bar(models, costs, color=model_colors, edgecolor=text_color, linewidth=1.5)
ax2.set_yscale('log')
ax2.set_ylabel('Cost (EUR, log scale)', color=text_color, fontsize=14, fontweight='bold')
ax2.set_title('Inference Cost Comparison - Five Shot Prompting Embeddings Specific', 
              color=text_color, fontsize=16, fontweight='bold', pad=20)
ax2.tick_params(colors=text_color, labelsize=10)
plt.xticks(rotation=45, ha='right')
ax2.grid(axis='y', color=grid_color, alpha=0.3, linestyle='--')

for bar, val in zip(bars2, costs):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height * 1.3, f'€{val:.4f}', 
             ha='center', va='bottom', color=text_color, fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('cost_comparison.png', dpi=300, facecolor=bg_color)
print("Chart 2 saved: cost_comparison.png")

# Chart 3: Performance vs Execution Time (scatter plot)
fig3, ax3 = plt.subplots(figsize=(12, 8), facecolor=bg_color)
ax3.set_facecolor(bg_color)

for i, model in enumerate(models):
    ax3.scatter(f1_scores[i], time_seconds[i], s=250, c=model_colors[i], 
               edgecolors=text_color, linewidth=2, label=model, alpha=0.9)

ax3.set_xlabel('F1 Score', color=text_color, fontsize=14, fontweight='bold')
ax3.set_ylabel('Execution Time (seconds, log scale)', color=text_color, fontsize=14, fontweight='bold')
ax3.set_yscale('log')
ax3.set_title('Performance vs Execution Time Trade-off - Five Shot Prompting Embeddings Specific', 
              color=text_color, fontsize=16, fontweight='bold', pad=20)
ax3.tick_params(colors=text_color, labelsize=11)
ax3.grid(color=grid_color, alpha=0.3, linestyle='--')
ax3.legend(loc='best', facecolor=bg_color, edgecolor=text_color, 
          fontsize=9, framealpha=0.9, labelcolor=text_color)

plt.tight_layout()
plt.savefig('performance_vs_time.png', dpi=300, facecolor=bg_color)
print("Chart 3 saved: performance_vs_time.png")

# Chart 4: Power Consumption (local models only)
local_models = []
local_power = []
local_colors = []

for i, p in enumerate(power):
    if p is not None:
        local_models.append(models[i])
        local_power.append(p)
        local_colors.append(model_colors[i])

fig4, ax4 = plt.subplots(figsize=(12, 8), facecolor=bg_color)
ax4.set_facecolor(bg_color)

bars4 = ax4.bar(local_models, local_power, color=local_colors, edgecolor=text_color, linewidth=1.5)
ax4.set_ylabel('Power Consumption (kWh)', color=text_color, fontsize=14, fontweight='bold')
ax4.set_title('Power Consumption - Five Shot Prompting Embeddings Specific (Local Models)', 
              color=text_color, fontsize=16, fontweight='bold', pad=20)
ax4.tick_params(colors=text_color, labelsize=10)
plt.xticks(rotation=45, ha='right')
ax4.grid(axis='y', color=grid_color, alpha=0.3, linestyle='--')

for bar, val in zip(bars4, local_power):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.3f}', 
             ha='center', va='bottom', color=text_color, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('power_consumption.png', dpi=300, facecolor=bg_color)
print("Chart 4 saved: power_consumption.png")

print("\nAll charts generated successfully with Llama-3.3-70B-Instruct correction!")
