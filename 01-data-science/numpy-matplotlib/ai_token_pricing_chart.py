import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

dtype = [('company', 'U15'), ('model', 'U20'), ('input_per_1m', 'f4'), ('output_per_1m', 'f4')]

pricing_data = np.array([
    ('OpenAI',    'GPT-4o',           2.50, 10.00),
    ('OpenAI',    'GPT-4o-mini',      0.15,  0.60),
    ('Anthropic', 'Claude-3.5-Sonnet',3.00, 15.00),
    ('Google',    'Gemini-1.5-Pro',   1.25,  5.00),
    ('DeepSeek',  'DeepSeek-V3',      0.14,  0.28),
    ('Meta/Groq', 'Llama-3.3-70B',   0.18,  0.40),
    ('Mistral',   'Mistral-Large',    2.00,  6.00),
    ('xAI',       'Grok-2',           2.00, 10.00),
    ('Cohere',    'Command-R-Plus',   2.50, 10.00),
], dtype=dtype)

# Fixed 500k (0.5M) input tokens; vary output from 100k to 1M
input_m = 0.5
output_m_range = np.linspace(0.1, 1.0, 10)

fig, ax = plt.subplots(figsize=(11, 6))

for row in pricing_data:
    costs = (input_m * row['input_per_1m']) + (output_m_range * row['output_per_1m'])
    ax.plot(
        output_m_range * 1_000_000,
        costs,
        label=f"{row['company']} ({row['model']})",
        marker='o',
    )

ax.set_title('AI Model API Cost Scaling\n(Fixed 500k Input Tokens, Varying Output)', fontsize=13)
ax.set_xlabel('Output Tokens')
ax.set_ylabel('Total Cost (USD $)')
ax.grid(True, linestyle='--', alpha=0.6)

# Format x-axis ticks as human-readable (100K, 500K, 1M) instead of scientific notation
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda val, _: f'{int(val):,}')
)
plt.xticks(rotation=20)

ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('ai_pricing_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
