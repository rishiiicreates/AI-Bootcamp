import pandas as pd
import requests
import json

df = pd.read_csv('/Users/rishii/AI-Bootcamp/models_benchmark_and_training.csv')
csv_text = df[['model_name', 'total_parameters_b', 'chatbot_arena_elo', 'best_use_case_notes']].to_string()

user_query = input("Enter your workflow requirement: ")
prompt = f"Dataset:\n{csv_text}\n\nBased strictly on the dataset above, recommend the best model for: {user_query}"

payload = {
    "model": "qwen3.5:4b",
    "prompt": prompt,
    "stream": True
}

response = requests.post('http://localhost:11434/api/generate', json=payload, stream=True)

for line in response.iter_lines():
    if line:
        data = json.loads(line.decode('utf-8'))
        print(data.get('response', ''), end='', flush=True)
print()

