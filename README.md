# AI Bootcamp — Complete Study Notes

> **FutureSkills PRIME / CDAC — Bootcamp on Artificial Intelligence (AI)**
>
> This repository contains all notes, practicals, code, MCQs, revisions, and mock tests for the AI Bootcamp.

The official bootcamp is **40 hours**: 20 hours of theory + 20 hours of practical/e-Lab. The theory assessment is **40 marks across 30 questions**.

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
│   └── yolo_object_detection.py        # Live object detection + counting with YOLOv8
│
├── 04-rps-game/
│   ├── collect_hand_gesture_data.py    # Collect MediaPipe hand landmark data for RPS training
│   ├── train_model.py                  # Train ML classifier on gesture dataset (Random Forest)
│   └── play_rock_paper_scissors.py     # Play RPS in real-time against CPU using webcam
│
├── notebooks/
│   ├── numpy_pandas_sklearn_basics.ipynb  # NumPy, Pandas, scikit-learn intro notebook
│   └── numpy_pandas_intro.ipynb           # Basic numpy/pandas DataFrame notebook
│
├── assets/
│   ├── cat.jpeg                        # Sample image used in CV demos
│   └── plot_output.png                 # Sample chart output
│
└── README.md
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
└── 07. Ethics in AI
    ├── Ethical AI
    ├── Explainable AI
    └── Responsible AI
```

---

## ⚙️ Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install numpy pandas matplotlib scipy scikit-learn opencv-python mediapipe ultralytics
```

> **Note:** `yolov8n.pt` is excluded from the repo (large binary). It will be auto-downloaded by `ultralytics` on first run.
