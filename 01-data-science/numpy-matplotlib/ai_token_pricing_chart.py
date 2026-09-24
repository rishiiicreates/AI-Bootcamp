import numpy as np
import matplotlib.pyplot as plt

dtype = [('company', 'U15'), ('model', 'U20'), ('input_per_1m', 'f4'), ('output_per_1m', 'f4')]

pricing_data = np.array([
    ('OpenAI', 'GPT-4o', 2.50, 10.00),
    ('OpenAI', 'GPT-4o-mini', 0.15, 0.60),
    ('Anthropic', 'Claude-3.5-Sonnet', 3.00, 15.00),
    ('Google', 'Gemini-1.5-Pro', 1.25, 5.00),
    ('DeepSeek', 'DeepSeek-V3', 0.14, 0.28),
    ('Meta/Groq', 'Llama-3.3-70B', 0.18, 0.40),
    ('Mistral', 'Mistral-Large', 2.00, 6.00),
    ('xAI', 'Grok-2', 2.00, 10.00),
    ('Cohere', 'Command-R-Plus', 2.50, 10.00)
], dtype=dtype)

input_m = 0.5
output_m_range = np.linspace(0.1, 1.0, 10)

plt.figure(figsize=(10, 6))

for row in pricing_data:
    costs = (input_m * row['input_per_1m']) + (output_m_range * row['output_per_1m'])
    plt.plot(output_m_range * 1_000_000, costs, label=f"{row['company']} ({row['model']})", marker='o')

plt.title('AI Model API Cost Scaling (Fixed 500k Input Tokens)')
plt.xlabel('Output Tokens')
plt.ylabel('Total Cost ($ USD)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig('ai_pricing_comparison.png', dpi=300)
plt.show()

