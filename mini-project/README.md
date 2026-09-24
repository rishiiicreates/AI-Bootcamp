# Mini-Project — AI Disaster Scan System

> **AI Bootcamp (FutureSkills PRIME / CDAC) — Mini Project**
>
> An end-to-end disaster response system combining transfer learning (MobileNetV2)
> for hazard classification with YOLOv8 for people detection, connected to a
> rule engine that generates risk levels and dispatches actionable responses.

---

## 📋 Project Overview

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Hazard Classifier | MobileNetV2 (transfer learning) | Classify scene: earthquake / fire / flood / normal |
| People Detector | YOLOv8n (fine-tuned) | Count and locate people in the frame |
| Rule Engine | Python logic | Combine hazard + people → risk + dispatch plan |

### Risk Levels

| Scenario | Risk |
|----------|------|
| Normal scene | LOW |
| Hazard detected, no people | MEDIUM |
| Hazard + people present | HIGH |
| Hazard + 10+ people | CRITICAL |

---

## 📁 Folder Structure

```text
mini-project/
│
├── calamity_model.py          # Main model: train + scan commands
├── test_calamity_model.py     # Automated test harness (120 images)
│
└── data/
    └── hazard/
        ├── train/             # Training images (place your dataset here)
        │   ├── earthquake/
        │   ├── fire/
        │   ├── flood/
        │   └── normal/
        ├── val/               # Validation images (same structure)
        └── test/              # 30 test images per class (pre-loaded)
            ├── earthquake/    # 30 images
            ├── fire/          # 30 images
            ├── flood/         # 30 images
            └── normal/        # 30 images
```

---

## ⚙️ Setup

From the **repo root**, follow the main [README setup](../README.md#️-setup-one-time-start-here) first.

Additional dependencies for this mini-project:
```bash
pip install torch torchvision pillow ultralytics
```

---

## 🚀 How to Run

> **All commands must be run from inside the `mini-project/` folder:**

```bash
cd mini-project
```

### 1. Train the Hazard Classifier

Requires your full dataset in `data/hazard/train/` and `data/hazard/val/`
(subdirectory per class: `earthquake/`, `fire/`, `flood/`, `normal/`).

```bash
python calamity_model.py train-hazard
```

- Uses MobileNetV2 pretrained on ImageNet
- Freezes backbone, trains only the classification head
- Runs for 8 epochs (~5 minutes on CPU, ~1 minute on GPU)
- Saves weights to `hazard_mobilenet.pt`

Training output:
```
epoch 1/8  val acc: 0.773
epoch 2/8  val acc: 0.806
epoch 3/8  val acc: 0.832
...
epoch 8/8  val acc: 0.872
```

---

### 2. (Optional) Fine-tune YOLOv8 for People Detection

Requires a YOLO-format dataset at `data/people/people.yaml`.

```bash
python calamity_model.py train-people
```

---

### 3. Scan a Single Image

```bash
python calamity_model.py scan data/hazard/test/fire/image_1218.png
python calamity_model.py scan data/hazard/test/flood/image_239.png
python calamity_model.py scan /path/to/any/image.jpg
```

Example output:
```
hazard         : flood
confidence     : 0.94
people_detected: 7
risk           : HIGH
actions        : ['Evacuate to high ground', 'Cut power to flooded zones', 'Use boats/helicopters for stranded people']
reinforcements : boats + rescue swimmers -> bottom-centre of the frame (people cluster, 7 detected)
Note: model output, verify with a human responder before acting.
```

---

### 4. Run the Full Automated Test Suite

Tests all 120 pre-loaded images (30 per class) and prints a full report:

```bash
python test_calamity_model.py
```

Output includes:
- ✅ Per-image prediction + confidence
- 📊 Per-class accuracy table
- 🔢 Confusion matrix
- ⚠️ Misclassified images list
- ⚠️ High-confidence errors (≥80% wrong)
- ⚡ Inference speed (ms/image, FPS)

**Benchmark results (CPU, MPS disabled):**

| Class | Accuracy | Correct/Total |
|-------|----------|---------------|
| Earthquake | **93.3%** | 28/30 |
| Fire | 73.3% | 22/30 |
| Flood | 73.3% | 22/30 |
| Normal | 76.7% | 23/30 |
| **Overall** | **79.2%** | **95/120** |

Inference speed: ~20ms/image (~50 FPS on CPU)

---

## ⚠️ Important Notes

- `hazard_mobilenet.pt` (model weights) is **not tracked in git** — train first.
- `yolov8n.pt` auto-downloads (~6 MB) on first scan run.
- `train-people` requires a separate YOLO-format people dataset not included here.
- Always verify model output with a human responder before real-world action.
