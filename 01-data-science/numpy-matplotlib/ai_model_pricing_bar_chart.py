import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Tokens per $1 dataset array
tokens = np.array([3571428, 1666666, 400000, 100000, 100000, 66666, 13333])
models = ['DeepSeek-V3', 'GPT-4o-mini', 'Gemini 2.5 Flash', 'Gemini 2.5 Pro', 'GPT-4o', 'Claude 3.5 Sonnet', 'Claude 3.5 Opus']

# Statistics
mean_val = np.mean(tokens)
median_val = np.median(tokens)
mode_val = stats.mode(tokens, keepdims=True).mode[0]
std_val = np.std(tokens)
min_val = np.min(tokens)
max_val = np.max(tokens)

# Matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B0', '#CCB974', '#64B5CD', '#8C564B']
bars = ax.barh(models, tokens, color=colors, edgecolor='black', alpha=0.85)


ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:,.0f}')
ax.axvline(median_val, color='green', linestyle=':', linewidth=2, label=f'Median: {median_val:,.0f}')

ax.set_xscale('log')
ax.set_xlabel('Output Tokens per $1 (Log Scale)', fontsize=12, fontweight='bold')
ax.set_title('Top AI API Output Tokens per $1 with Statistical Markers', fontsize=14, fontweight='bold', pad=15)

for bar in bars:
    width = bar.get_width()
    ax.text(width * 1.15, bar.get_y() + bar.get_height()/2, f'{width:,}', va='center', fontsize=10, fontweight='bold')

plt.xlim(right=max(tokens) * 5)
plt.gca().invert_yaxis()
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()

