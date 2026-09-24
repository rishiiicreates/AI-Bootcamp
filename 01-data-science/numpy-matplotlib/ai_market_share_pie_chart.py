import matplotlib.pyplot as plt

# Data for AI API market share distribution by token volume
labels = ['OpenAI', 'Anthropic', 'Google Gemini', 'DeepSeek', 'Meta (Llama)', 'Others']
shares = [38, 22, 15, 12, 8, 5]
colors = ['#10a37f', '#d97706', '#4285f4', '#0066ff', '#06b6d4', '#8b5cf6']

plt.figure(figsize=(7, 7))
plt.pie(shares, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, explode=(0.05, 0, 0, 0, 0, 0))
plt.title('Estimated AI API Market Share (By Token Volume)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

