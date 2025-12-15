import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


# Color palette from your reference chart
colors = ['#5CA6B7', '#A8B450', '#F2B35D', '#E86E37', '#D94F2A']


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "charts"



# Helper function for smaller fonts with staggered labels
def plot_chart_small_labels(df, title, ylabel, save_path=None, show=False, value_format=None):
    plt.figure(figsize=(9, 6))
    bars = plt.bar(df['Methodology'], df['Value'], color=colors)
    
    # Titles and axes
    plt.title(title, fontsize=13, color='white', weight='bold')
    plt.ylabel(ylabel, color='white', fontsize=10)
    plt.xticks([], [])
    plt.yticks(color='white', fontsize=9)
    plt.ylim(0, df['Value'].max() * 1.25)
    plt.gca().set_facecolor('#0E4A80')
    plt.gcf().set_facecolor('#0E4A80')
    
    # Value labels
    for i, v in enumerate(df['Value']):
        if value_format:
            label = value_format(v)
        else:
            label = f"{v:.2f}" if v < 1000 else f"{int(v)}"

        plt.text(i, v + (df['Value'].max() * 0.02),
                 label,
                 ha='center', fontsize=8, color='white', weight='bold')
    
    # Main labels + sub-labels with staggered positioning (multiple rows)
    for i, (main_label, sub_label) in enumerate(zip(df['Methodology'], df['Model'])):
        # Alternate vertical positions to create two rows
        if i % 2 == 0:  # Even indices - first row (higher)
            main_y_offset = -df['Value'].max() * 0.08
            sub_y_offset = -df['Value'].max() * 0.14
        else:  # Odd indices - second row (lower)
            main_y_offset = -df['Value'].max() * 0.20
            sub_y_offset = -df['Value'].max() * 0.26
        
        plt.text(i, main_y_offset, main_label,
                 ha='center', color='white', fontsize=7, weight='bold')
        plt.text(i, sub_y_offset, f"({sub_label})",
                 ha='center', color='white', fontsize=6, style='italic')
    
    plt.grid(axis='y', linestyle='--', alpha=0.5, color='white')
    
    # Adjust bottom margin to accommodate staggered labels
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()

    saved_path = None
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved chart to {save_path}")
        saved_path = save_path

    if show:
        plt.show()

    plt.close()
    return saved_path



# 1️⃣ F1-Scores
f1_df = pd.DataFrame({
    'Methodology': [
        'Rule-based baseline',
        'Zero Shot Prompting',
        'Few Shot Prompting',
        'Retrieval Augmented Prompting',
        'Fine-Tuned Encoder Models'
    ],
    'Model': [
        '—',
        'DeepSeekV3 – Specific Schema',
        'DeepSeekV3 – Specific Schema',
        'DeepSeekV3 – Specific Schema',
        'XLM-Roberta-Large – Broad Schema'
    ],
    'Value': [56.85, 53.76, 58.91, 59.24, 78.01]
})



# 2️⃣ Power Consumption
power_df = pd.DataFrame({
    'Methodology': [
        'Rule-based System',
        'Fine-Tuned Encoder Models',
        'Zero Shot Prompting',
        'Few Shot Prompting',
        'Retrieval Augmented Prompting'
    ],
    'Model': [
        '—',
        'XLM-Roberta-Large – Broad Schema',
        'Llama-3.3-70B-Instruct – Specific Schema',
        'Qwen-2.5-32B-Instruct – Specific Schema',
        'Qwen-2.5-32B-Instruct – Broad Schema'
    ],
    'Value': [0.0005724, 0.566, 0.103, 0.040, 0.068]
})



# 3️⃣ Cost
cost_df = pd.DataFrame({
    'Methodology': [
        'Rule-based System',
        'Fine-Tuned Encoder Models',
        'Zero Shot Prompting',
        'Few Shot Prompting',
        'Retrieval Augmented Prompting'
    ],
    'Model': [
        '—',
        'XLM-Roberta-Large – Broad Schema',
        'DeepSeekV3 – Specific Schema',
        'DeepSeekV3 – Specific Schema',
        'DeepSeekV3 – Broad Schema'
    ],
    'Value': [0.04, 0.02, 55.17, 122.20, 126.06]
})



# 4️⃣ Processing Time
time_df = pd.DataFrame({
    'Methodology': [
        'Rule-based System',
        'Retrieval Augmented Prompting',
        'Zero Shot Prompting',
        'Few Shot Prompting',
        'Fine-Tuned Encoder Models'
    ],
    'Model': [
        '—',
        'DeepSeekV3 – Broad Schema',
        'DeepSeekV3 – Specific Schema',
        'DeepSeekV3 – Specific Schema',
        'XLM-Roberta-Large – Broad Schema'
    ],
    'Value': [57, 297, 365, 665, 15741]
})



encoder_model_f1_df = pd.DataFrame({
    'Model Name': ['Agribert', 'Scibert', 'XLM-Roberta-Large'],
    'Exact F1 Score': [0.75, 0.79, 0.81]
})



llm_f1_df = pd.DataFrame({
    'Model Name': [
        'GPT-5',
        'Qwen2.5-32B-Instruct',
        'DeepSeekV3',
        'Qwen2.5-14B-Instruct',
        'Qwen2.5-7B-Instruct',
        'Qwen2.5-72B-Instruct',
        'Llama-3.3-70B-Instruct',
        'Llama-3.1-8B-Instruct'
    ],
    'Exact F1 Score': [
        0.470003,
        0.515105,
        0.532209,
        0.519347,
        0.402957,
        0.479482,
        0.294811,
        0.432523
    ]
})



def plot_horizontal_chart(df, category_col, value_col, title, xlabel, save_path=None, show=False, value_format=None):
    plt.figure(figsize=(12, 6))
    y_positions = range(len(df))
    bar_colors = [colors[i % len(colors)] for i in range(len(df))]
    bars = plt.barh(y_positions, df[value_col], color=bar_colors)

    plt.title(title, fontsize=14, color='white', weight='bold')
    plt.xlabel(xlabel, color='white', fontsize=11)
    plt.ylabel('Model Name', color='white', fontsize=11)
    plt.yticks(y_positions, df[category_col], fontsize=10, color='white')
    plt.xticks(color='white', fontsize=10)
    plt.gca().invert_yaxis()
    plt.gca().set_facecolor('#0E4A80')
    plt.gcf().set_facecolor('#0E4A80')
    plt.grid(axis='x', linestyle='--', alpha=0.4, color='white')

    if value_format is None:
        value_format = lambda v: f"{v:.4f}"

    for bar, value in zip(bars, df[value_col]):
        plt.text(value + (max(df[value_col]) * 0.01),
                 bar.get_y() + bar.get_height() / 2,
                 value_format(value),
                 va='center', ha='left', color='white', fontsize=10, weight='bold')

    plt.tight_layout()

    saved_path = None
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved chart to {save_path}")
        saved_path = save_path

    if show:
        plt.show()

    plt.close()
    return saved_path



def generate_all_charts(output_dir=DEFAULT_OUTPUT_DIR, show=False):
    output_dir = Path(output_dir)
    chart_configs = [
        (f1_df, 'Comparison of Partial Match F1-Scores Across All Methodologies', 'F1 Score (%)', output_dir / 'f1_partial_scores.png', None),
        (power_df, 'Sustainability Metrics Comparison: Power Consumption', 'Power (KWh)', output_dir / 'power_consumption.png', lambda v: f"{v:.4f}"),
        (cost_df, 'Sustainability Metrics Comparison: Cost', 'Cost (€)', output_dir / 'cost.png', None),
        (time_df, 'Sustainability Metrics Comparison: Processing Time', 'Time (seconds)', output_dir / 'processing_time.png', None),
    ]

    saved_files = []
    for df, title, ylabel, path, value_format in chart_configs:
        saved_path = plot_chart_small_labels(df, title, ylabel, save_path=path, show=show, value_format=value_format)
        saved_files.append(saved_path)

    horizontal_chart_configs = [
        (encoder_model_f1_df.sort_values('Exact F1 Score', ascending=True),
         'Model Name',
         'Exact F1 Score',
         'Encoder Model Performance Comparison - Exact F1 Scores',
         'Exact F1 Score',
         output_dir / 'encoder_model_f1_scores.png'),
        (llm_f1_df.sort_values('Exact F1 Score', ascending=True),
         'Model Name',
         'Exact F1 Score',
         'Large Language Models Performance Comparison - Exact F1 Scores',
         'Exact F1 Score',
         output_dir / 'llm_f1_scores.png')
    ]

    for df, category_col, value_col, title, xlabel, path in horizontal_chart_configs:
        saved_path = plot_horizontal_chart(df, category_col, value_col, title, xlabel, save_path=path, show=show)
        saved_files.append(saved_path)

    return saved_files



if __name__ == "__main__":
    generate_all_charts(show=True)
