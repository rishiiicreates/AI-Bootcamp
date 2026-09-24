"""
disaster_model.py - train and run a small disaster-scan system.

  1) Hazard classifier  (MobileNetV2, transfer learning)   -> AIDER-style folders
  2) People detector    (YOLOv8n fine-tune)                -> VisDrone / HERIDAL in YOLO format
  3) Rule engine        (hazard + people -> risk, actions, where to send help)

Setup:
    pip install torch torchvision ultralytics pillow

Data layout:
    data/hazard/train/<class_name>/*.jpg   and   data/hazard/val/<class_name>/*.jpg
    data/people/people.yaml                (YOLO dataset yaml with class 0 = person)

Commands:
    python disaster_model.py train-hazard
    python disaster_model.py train-people
    python disaster_model.py scan photo.jpg
"""
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms

HAZARD_DIR = "data/hazard"
HAZARD_WEIGHTS = "hazard_mobilenet.pt"
PEOPLE_YAML = "data/people/people.yaml"
PEOPLE_WEIGHTS = "runs/detect/people/weights/best.pt"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------- 1) hazard classifier ----------
def train_hazard(epochs=8):
    train = datasets.ImageFolder(f"{HAZARD_DIR}/train", TF)
    val = datasets.ImageFolder(f"{HAZARD_DIR}/val", TF)
    tl = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    vl = torch.utils.data.DataLoader(val, batch_size=32)

    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for p in m.features.parameters():
        p.requires_grad = False  # freeze backbone, train head only
    m.classifier[1] = nn.Linear(m.last_channel, len(train.classes))
    m.to(DEVICE)

    opt = torch.optim.Adam(m.classifier.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss_fn(m(x), y).backward()
            opt.step()
        m.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                pred = m(x.to(DEVICE)).argmax(1).cpu()
                correct += (pred == y).sum().item()
                total += len(y)
        print(f"epoch {ep + 1}/{epochs}  val acc: {correct / total:.3f}")
    torch.save({"state": m.state_dict(), "classes": train.classes}, HAZARD_WEIGHTS)


def predict_hazard(img):
    ck = torch.load(HAZARD_WEIGHTS, map_location=DEVICE)
    m = models.mobilenet_v2()
    m.classifier[1] = nn.Linear(m.last_channel, len(ck["classes"]))
    m.load_state_dict(ck["state"])
    m.to(DEVICE).eval()
    with torch.no_grad():
        probs = m(TF(img).unsqueeze(0).to(DEVICE)).softmax(1)[0]
    i = int(probs.argmax())
    return ck["classes"][i], float(probs[i])


# ---------- 2) people detector ----------
def train_people(epochs=30):
    from ultralytics import YOLO
    YOLO("yolov8n.pt").train(data=PEOPLE_YAML, epochs=epochs, imgsz=960, batch=8, name="people")


def detect_people(path):
    from ultralytics import YOLO
    import os
    if not os.path.exists(PEOPLE_WEIGHTS):
        # Fallback to base YOLOv8n if fine-tuned weights don't exist
        return YOLO("yolov8n.pt")(path, imgsz=960, conf=0.25, verbose=False, classes=[0])[0].boxes.xyxy.cpu().tolist()
    r = YOLO(PEOPLE_WEIGHTS)(path, imgsz=960, conf=0.25, verbose=False)[0]
    return r.boxes.xyxy.cpu().tolist()  # [[x1,y1,x2,y2], ...]


# ---------- 3) rule engine ----------
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
    if cmd == "train-hazard":
        train_hazard()
    elif cmd == "train-people":
        train_people()
    elif cmd == "scan" and len(sys.argv) > 2:
        scan(sys.argv[2])
    else:
        print(__doc__)
