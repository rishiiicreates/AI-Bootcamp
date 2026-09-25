"""
calamity_model_V1.py - train and run a small disaster-scan system.

  1) Hazard classifier  (CLIP zero-shot, no training needed) -> predicts from image directly
  2) People detector    (YOLOv8n fine-tune)                  -> VisDrone / HERIDAL in YOLO format
  3) Rule engine        (hazard + people -> risk, actions, where to send help)

Setup:
    pip install torch torchvision ultralytics pillow
    pip install git+https://github.com/openai/CLIP.git

Data layout:
    data/people/people.yaml  (YOLO dataset yaml with class 0 = person)
    NOTE: No hazard training data needed - CLIP classifies images directly.

Commands:
    python calamity_model_V1.py train-people
    python calamity_model_V1.py scan photo.jpg
"""
import sys

import torch
import clip
from PIL import Image

PEOPLE_YAML = "data/people/people.yaml"
PEOPLE_WEIGHTS = "runs/detect/people/weights/best.pt"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# NOTE: CLIP handles its own image preprocessing internally.


# ---------- 1) hazard classifier (CLIP zero-shot) ----------

# Cache so we don't reload the model on every scan() call.
_CLIP_CACHE = None

# Descriptive text labels — more specific = better CLIP accuracy.
HAZARD_LABELS = [
    "flooded streets with brown water and submerged buildings",
    "collapsed buildings and rubble after an earthquake",
    "large wildfire with flames and smoke",
    "normal outdoor scene with no disaster",
]

LABEL_MAP = {
    HAZARD_LABELS[0]: "flood",
    HAZARD_LABELS[1]: "earthquake",
    HAZARD_LABELS[2]: "fire",
    HAZARD_LABELS[3]: "normal",
}


def predict_hazard(img):
    """Classify disaster type directly from image using CLIP (no training needed)."""
    global _CLIP_CACHE
    if _CLIP_CACHE is None:
        _CLIP_CACHE = clip.load("ViT-B/32", device=DEVICE)
    model, preprocess = _CLIP_CACHE
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    tokens = clip.tokenize(HAZARD_LABELS).to(DEVICE)
    with torch.no_grad():
        logits, _ = model(img_tensor, tokens)
        probs = logits.softmax(dim=-1)[0]
    i = int(probs.argmax())
    return LABEL_MAP[HAZARD_LABELS[i]], float(probs[i])



#2) people detector
def train_people(epochs=30):
    from ultralytics import YOLO
    YOLO("yolov8n.pt").train(data=PEOPLE_YAML, epochs=epochs, imgsz=960, batch=8, name="people")


def detect_people(path):
    from ultralytics import YOLO
    import os
    if not os.path.exists(PEOPLE_WEIGHTS):
        # YOLO fallback
        return YOLO("yolov8n.pt")(path, imgsz=960, conf=0.25, verbose=False, classes=[0])[0].boxes.xyxy.cpu().tolist()
    r = YOLO(PEOPLE_WEIGHTS)(path, imgsz=960, conf=0.25, verbose=False)[0]
    return r.boxes.xyxy.cpu().tolist()  # [[x1,y1,x2,y2], ...]


#3) rule engine
def region_of(boxes, w, h):
    """Where the people cluster, as a 3x3 grid cell name."""
    cx = sum((b[0] + b[2]) / 2 for b in boxes) / len(boxes)
    cy = sum((b[1] + b[3]) / 2 for b in boxes) / len(boxes)
    col = ["left", "centre", "right"][min(int(cx / w * 3), 2)]
    row = ["top", "middle", "bottom"][min(int(cy / h * 3), 2)]
    return f"{row}-{col} of the frame"


PLAYBOOK = {
    "flood": ("boats + rescue swimmers", ["Evacuate to high ground", "Cut power to flooded zones",
                                         "Use boats/helicopters for stranded people"]),
    "fire": ("fire engines + medical", ["Create firebreaks / evacuate downwind", "Establish water supply",
                                       "Treat smoke inhalation casualties"]),
    "earthquake": ("heavy machinery + search & rescue", ["Shore up unstable structures",
                                                        "Use search dogs / thermal cameras",
                                                        "Shut off gas and power"]),
    "collapsed_building": ("heavy machinery + search & rescue", ["Shore up unstable structures",
                                                                "Use search dogs / thermal cameras",
                                                                "Shut off gas and power"]),
    "traffic_accident": ("ambulance + police", ["Secure the scene", "Triage casualties", "Divert traffic"]),
    "normal": (None, ["No hazard detected - keep monitoring"]),
}


def assess(hazard, conf, boxes, size):
    n = len(boxes)
    risk = "LOW"
    if hazard != "normal":
        risk = "MEDIUM"
        if n > 0:
            risk = "HIGH"
        if n >= 10:
            risk = "CRITICAL"
    resource, actions = PLAYBOOK.get(hazard.lower().replace(" ", "_"), PLAYBOOK["normal"])
    send = None
    if resource and n:
        send = f"{resource} -> {region_of(boxes, *size)} (people cluster, {n} detected)"
    return {"hazard": hazard, "confidence": round(conf, 2), "people_detected": n,
            "risk": risk, "actions": actions, "reinforcements": send}


def scan(path):
    img = Image.open(path).convert("RGB")
    hazard, conf = predict_hazard(img)
    try:
        boxes = detect_people(path)
    except Exception as e:
        print(f"Warning: People detection unavailable ({e}), continuing with hazard-only mode")
        boxes = []
    report = assess(hazard, conf, boxes, img.size)
    for k, v in report.items():
        print(f"{k:15}: {v}")
    print("Note: model output, verify with a human responder before acting.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "train-people":
        train_people()
    elif cmd == "scan" and len(sys.argv) > 2:
        scan(sys.argv[2])
    else:
        print(__doc__)

