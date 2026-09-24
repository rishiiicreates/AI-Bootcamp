# AI Bootcamp — Complete Study Notes & Practicals

> **FutureSkills PRIME / CDAC — Bootcamp on Artificial Intelligence (AI)**
>
> This repository contains all notes, practicals, code, MCQs, revisions, and mock tests for the AI Bootcamp.
> The official bootcamp is **40 hours**: 20 hours of theory + 20 hours of practical/e-Lab.

---

## 📁 Repository Structure

```text
AI-Bootcamp/
│
├── 00-python-basics/
│   ├── calculator.py                   # eval-based expression calculator
│   ├── filter_even_numbers.py          # filter even numbers from input list
│   └── reverse_list.py                 # in-place list reversal algorithm
│
├── 01-data-science/
│   ├── pandas/
│   │   ├── pandas_dataframe_basics.py  # DataFrame creation, head(), column access, iloc
│   │   └── pandas_dataframe_intro.py   # DataFrame creation and column access intro
│   │
│   └── numpy-matplotlib/
│       ├── matplotlib_bar_chart.py         # Bar chart: students vs marks
│       ├── matplotlib_sine_wave.py         # Static sine wave plot
│       ├── matplotlib_sine_animation.py    # Animated sine wave with FuncAnimation
│       ├── ai_model_pricing_bar_chart.py   # AI model tokens/$1 comparison (NumPy + SciPy)
│       ├── ai_market_share_pie_chart.py    # AI API market share pie chart
│       └── ai_token_pricing_chart.py       # AI model input/output token pricing chart
│
├── 02-ai-models/
│   ├── qwen_model_recommender.py       # LLM-based model recommender using Qwen3.5 via Ollama
│   └── models_benchmark_and_training.csv  # Benchmark dataset (ELO scores, parameters, use cases)
│
├── 03-computer-vision/
│   ├── image-basics/
│   │   ├── opencv_image_basics.py      # Load, display, and resize an image with OpenCV
│   │   └── opencv_load_and_display.py  # Load and visualise image (OpenCV + Matplotlib)
│   ├── air_canvas.py                   # Virtual air-drawing with HSV colour tracking
│   └── yolo_object_detection.py        # Live object detection + counting with YOLOv8n
│
├── 04-rps-game/
│   ├── collect_hand_gesture_data.py    # Step 1 — Collect MediaPipe hand landmark data
│   ├── train_model.py                  # Step 2 — Train Random Forest on gesture dataset
│   └── play_rock_paper_scissors.py     # Step 3 — Play RPS in real-time via webcam
│
├── 05-nlp/
│   ├── nlp.py                          # Real-time sentiment analyzer using TextBlob
│   ├── resumescan.py                   # Resume entity extraction with spaCy NER
│   ├── transformers.py                 # Attention mechanism visualization demo
│   └── wordcld.py                      # Personal word cloud generator with NLTK
│
├── notebooks/
│   ├── numpy_pandas_sklearn_basics.ipynb  # NumPy, Pandas, scikit-learn intro notebook
│   └── numpy_pandas_intro.ipynb           # Basic numpy/pandas DataFrame notebook
│
├── assets/
│   ├── cat.jpeg                        # Sample image used in CV demos
│   └── plot_output.png                 # Sample chart output
│
├── data/
│   └── hazard/
│       ├── train/   {earthquake, fire, flood, normal}   # Training images (MobileNetV2)
│       ├── val/     {earthquake, fire, flood, normal}   # Validation images
│       └── test/    {earthquake, fire, flood, normal}   # 30 test images per class (120 total)
│
├── calamity_model.py                   # 🔴 Disaster scan system (MobileNetV2 + YOLOv8)
├── test_calamity_model.py              # Automated test harness — 120 real images, full report
└── README.md
```

---

## ⚙️ Setup (One-time, start here)

### 1. Clone the repo

```bash
git clone https://github.com/rishiiicreates/AI-Bootcamp.git
cd AI-Bootcamp
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install all dependencies

```bash
pip install --upgrade pip
pip install numpy pandas matplotlib scipy scikit-learn \
            opencv-python mediapipe ultralytics \
            textblob spacy wordcloud nltk \
            torch torchvision pillow joblib
```

### 4. Download required NLP models/data

```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
python -c "import textblob; textblob.download_corpora()" 2>/dev/null || true
```

> **Note:** `yolov8n.pt` (~6 MB) is **not included** in the repo (binary). It auto-downloads on first run of `yolo_object_detection.py` or `calamity_model.py`.

---

## 🚀 Running Each Module

### 00 — Python Basics

```bash
# Calculator (type an expression, press Enter)
python 00-python-basics/calculator.py

# Filter even numbers (enter space-separated integers)
python 00-python-basics/filter_even_numbers.py

# Reverse a list (enter space-separated integers)
python 00-python-basics/reverse_list.py
```

---

### 01 — Data Science (Pandas + NumPy + Matplotlib)

```bash
# Pandas basics
python 01-data-science/pandas/pandas_dataframe_basics.py
python 01-data-science/pandas/pandas_dataframe_intro.py

# Charts (opens a matplotlib window)
python 01-data-science/numpy-matplotlib/matplotlib_bar_chart.py
python 01-data-science/numpy-matplotlib/matplotlib_sine_wave.py
python 01-data-science/numpy-matplotlib/matplotlib_sine_animation.py
python 01-data-science/numpy-matplotlib/ai_model_pricing_bar_chart.py
python 01-data-science/numpy-matplotlib/ai_market_share_pie_chart.py
python 01-data-science/numpy-matplotlib/ai_token_pricing_chart.py
```

---

### 02 — AI Models

```bash
# Requires Ollama running locally with Qwen3.5 pulled:
# ollama pull qwen2.5:3b
python 02-ai-models/qwen_model_recommender.py
```

---

### 03 — Computer Vision

> Requires a **webcam** for air canvas and YOLO. Image scripts work headlessly.

```bash
# Load + display cat.jpeg (needs a GUI / display)
python 03-computer-vision/image-basics/opencv_image_basics.py
python 03-computer-vision/image-basics/opencv_load_and_display.py

# Virtual air canvas — wave a BLUE object in front of webcam
python 03-computer-vision/air_canvas.py

# Live object detection with YOLOv8n (downloads model on first run ~6 MB)
python 03-computer-vision/yolo_object_detection.py
```

**Controls:**
| Script | Key | Action |
|--------|-----|--------|
| `opencv_image_basics.py` | any key | advance window |
| `air_canvas.py` | `c` | clear canvas |
| `air_canvas.py` / `yolo_object_detection.py` | `q` | quit |

---

### 04 — Rock Paper Scissors Game

> Run these **in order**. Requires a webcam.

```bash
# Step 1 — Collect gesture samples (R=Rock, P=Paper, S=Scissors, Q=Quit)
#           Aim for 100+ samples per class
python 04-rps-game/collect_hand_gesture_data.py

# Step 2 — Train the Random Forest classifier
python 04-rps-game/train_model.py

# Step 3 — Play the game!
python 04-rps-game/play_rock_paper_scissors.py
```

---

### 05 — NLP

```bash
# Real-time sentiment analyzer (type sentences interactively, 'exit' to quit)
python 05-nlp/nlp.py

# Resume entity extraction with spaCy
python 05-nlp/resumescan.py

# Attention mechanism demo (no dependencies beyond Python)
python 05-nlp/transformers.py

# Word cloud generator (opens matplotlib window)
python 05-nlp/wordcld.py
```

---

### 🔴 Calamity / Disaster Detection Model

> **Most advanced module** — MobileNetV2 hazard classifier + YOLOv8 people detector + rule engine.

#### Dataset Layout
The repo ships with **120 labelled test images** (`data/hazard/test/`) across 4 classes:
- `earthquake` — 30 images
- `fire` — 30 images
- `flood` — 30 images
- `normal` — 30 images

To **train** you need the full dataset in `data/hazard/train/` and `data/hazard/val/`.

#### Commands

```bash
# 1. Train the hazard classifier (MobileNetV2, 8 epochs, ~5 min on CPU)
python calamity_model.py train-hazard

# 2. (Optional) Fine-tune YOLOv8 for people detection
#    Requires YOLO-format dataset at data/people/people.yaml
python calamity_model.py train-people

# 3. Scan a single image and get a full risk report
python calamity_model.py scan path/to/image.jpg

# 4. Run the full automated test suite (120 images, confusion matrix, report)
python test_calamity_model.py
```

#### Example scan output

```
hazard         : flood
confidence     : 0.94
people_detected: 7
risk           : HIGH
actions        : ['Evacuate to high ground', 'Cut power to flooded zones', ...]
reinforcements : boats + rescue swimmers -> bottom-centre of the frame (people cluster, 7 detected)
Note: model output, verify with a human responder before acting.
```

---

### 📓 Notebooks

```bash
# Launch Jupyter
pip install jupyter
jupyter notebook

# Then open:
# notebooks/numpy_pandas_sklearn_basics.ipynb
# notebooks/numpy_pandas_intro.ipynb
```

---

## 🧠 Study Roadmap

```text
AI BOOTCAMP
│
├── 00. Foundations
│   └── Python basics required for AI
│
├── 01. Introduction to AI
│   ├── Introduction to AI
│   ├── Current Trends in AI
│   └── Applications of AI
│
├── 02. Python Basics & Data Preprocessing
│   ├── Python Basics (w.r.t AI)
│   ├── Data Preprocessing
│   └── Git Basics
│
├── 03. Data Visualisation & Machine Learning
│   ├── Data Visualisation
│   ├── Introduction to Machine Learning
│   ├── Types of Machine Learning
│   └── Evaluation Metrics
│
├── 04. Deep Learning
│   ├── Explainable AI
│   ├── Introduction to Deep Learning
│   ├── Types of Deep Learning
│   └── Hyperparameters
│
├── 05. Computer Vision
│   ├── Image Processing with OpenCV
│   └── Video Processing with OpenCV
│
├── 06. Natural Language Processing
│   ├── Introduction to NLP
│   ├── Text Preprocessing
│   ├── Generative AI
│   ├── Foundation Models
│   ├── LLM
│   └── Transformers
│
├── 07. Applied NLP Practicals
│   ├── Sentiment Analysis with TextBlob
│   ├── Named Entity Recognition with spaCy
│   ├── Attention Mechanism Visualization
│   └── Word Cloud Generation with NLTK
│
└── 08. Ethics in AI
    ├── Ethical AI
    ├── Explainable AI
    └── Responsible AI
```

---

## 🔬 Testing

```bash
# Run automated test suite for calamity_model.py
# (train the model first: python calamity_model.py train-hazard)
python test_calamity_model.py
```

Outputs: per-class accuracy, confusion table, misclassified images, high-confidence errors, and inference speed (ms/image).

---

## 📦 Dependencies Summary

| Package | Used In |
|---------|---------|
| `numpy` | All data science + CV modules |
| `pandas` | 01-data-science, 04-rps-game |
| `matplotlib` | 01-data-science, 05-nlp |
| `scipy` | 01-data-science charts |
| `scikit-learn` | 04-rps-game (Random Forest) |
| `opencv-python` | 03-computer-vision, 04-rps-game |
| `mediapipe` | 04-rps-game (hand landmarks) |
| `ultralytics` | 03-computer-vision (YOLOv8), calamity_model |
| `torch` + `torchvision` | calamity_model (MobileNetV2) |
| `pillow` | calamity_model |
| `joblib` | 04-rps-game (model save/load) |
| `textblob` | 05-nlp/nlp.py |
| `spacy` | 05-nlp/resumescan.py |
| `wordcloud` | 05-nlp/wordcld.py |
| `nltk` | 05-nlp/wordcld.py |

---

## ⚠️ Known Limitations

- `yolo_object_detection.py`, `air_canvas.py`, and the RPS game **require a physical webcam** — they will raise a `RuntimeError` otherwise.
- `calamity_model.py train-people` requires a separate YOLO-format people dataset (`data/people/people.yaml`) not included in the repo.
- `02-ai-models/qwen_model_recommender.py` requires **Ollama** running locally with `qwen2.5:3b` pulled.
- Model weights (`*.pt`, `*.pkl`) are excluded from the repo via `.gitignore`. Always train before scanning.
