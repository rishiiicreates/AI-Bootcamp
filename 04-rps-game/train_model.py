import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, "dataset", "data.csv")
model_path = os.path.join(script_dir, "rps_model.pkl")

if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    print("Please run collect_hand_gesture_data.py first to capture samples for rock, paper, and scissors.")
    exit(1)

# Each hand has 21 3D landmarks (x, y, z) = 63 features + 1 label
df = pd.read_csv(csv_path, header=None)

if df.empty or len(df.columns) < 64:
    print("Error: dataset appears empty or corrupted.")
    exit(1)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(f"Loaded {len(df)} samples across classes: {sorted(set(y))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%\n")
print(classification_report(y_test, clf.predict(X_test)))

joblib.dump(clf, model_path)
print(f"Model saved successfully to: {model_path}")
