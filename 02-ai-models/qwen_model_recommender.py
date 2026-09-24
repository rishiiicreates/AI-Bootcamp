import json
import os
import pandas as pd
import requests
from pathlib import Path

# Resolve CSV path relative to this script — works from any working directory
CSV_PATH = Path(__file__).parent / "models_benchmark_and_training.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
csv_text = df[
    ['model_name', 'total_parameters_b', 'chatbot_arena_elo', 'best_use_case_notes']
].to_string()

user_query = input("Enter your workflow requirement: ").strip()
if not user_query:
    print("No query entered. Exiting.")
    raise SystemExit

prompt = (
    f"Dataset:\n{csv_text}\n\n"
    f"Based strictly on the dataset above, recommend the best model for: {user_query}"
)

payload = {
    "model": "qwen3.5:4b",
    "prompt": prompt,
    "stream": True,
}

try:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        stream=True,
        timeout=60,
    )
    response.raise_for_status()
except requests.exceptions.ConnectionError:
    print("Error: Ollama is not running. Start it with: ollama serve")
    raise SystemExit
except requests.exceptions.HTTPError as e:
    print(f"HTTP error from Ollama: {e}")
    raise SystemExit

for line in response.iter_lines():
    if line:
        data = json.loads(line.decode("utf-8"))
        print(data.get("response", ""), end="", flush=True)
print()
