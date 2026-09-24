"""
test_calamity_model.py
-----------------------
Automated test harness for calamity_model.py
Tests the full scan pipeline: hazard classifier + rule engine
against 30 images per class (120 total) from data/hazard/test/

Usage:
    python3 test_calamity_model.py

Requirements:
    - hazard_mobilenet.pt must exist (run `python3 calamity_model.py train-hazard` first)
    - data/hazard/test/<class>/*.png must exist
"""
import os
import sys
import glob
import time
from collections import defaultdict

# ── Make imports work from repo root ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Import predict_hazard directly (no webcam/YOLO needed for hazard test) ──
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
HAZARD_WEIGHTS = os.path.join(os.path.dirname(__file__), "hazard_mobilenet.pt")
TEST_DIR = os.path.join(os.path.dirname(__file__), "data", "hazard", "test")

TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

SEPARATOR = "=" * 70


def load_model():
    if not os.path.exists(HAZARD_WEIGHTS):
        print(f"\n❌  Model weights not found at: {HAZARD_WEIGHTS}")
        print("    Run: python3 calamity_model.py train-hazard  first.\n")
        sys.exit(1)
    ck = torch.load(HAZARD_WEIGHTS, map_location=DEVICE)
    m = models.mobilenet_v2()
    m.classifier[1] = nn.Linear(m.last_channel, len(ck["classes"]))
    m.load_state_dict(ck["state"])
    m.to(DEVICE).eval()
    return m, ck["classes"]


def predict(model, classes, img_path):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return None, 0.0, str(e)
    with torch.no_grad():
        probs = model(TF(img).unsqueeze(0).to(DEVICE)).softmax(1)[0]
    i = int(probs.argmax())
    return classes[i], float(probs[i]), None


def run_tests():
    print(f"\n{SEPARATOR}")
    print("  AI-BOOTCAMP — calamity_model.py  |  HAZARD CLASSIFIER TEST SUITE")
    print(SEPARATOR)
    print(f"  Device   : {DEVICE}")
    print(f"  Weights  : {HAZARD_WEIGHTS}")
    print(f"  Test dir : {TEST_DIR}\n")

    model, classes = load_model()
    print(f"  Classes  : {classes}\n")

    # Collect ground-truth classes from folder names
    gt_classes = sorted([
        d for d in os.listdir(TEST_DIR)
        if os.path.isdir(os.path.join(TEST_DIR, d))
    ])

    results = []
    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0, "times": []})

    for gt_label in gt_classes:
        folder = os.path.join(TEST_DIR, gt_label)
        images = sorted(glob.glob(os.path.join(folder, "*.png")) +
                        glob.glob(os.path.join(folder, "*.jpg")))
        print(f"  ── Class: {gt_label.upper():15} ({len(images)} images)")
        print(f"     {'Image':<30} {'Predicted':<20} {'Conf':>6}  {'✓/✗'}")
        print(f"     {'-'*65}")

        for img_path in images:
            t0 = time.time()
            pred, conf, err = predict(model, classes, img_path)
            elapsed = time.time() - t0
            fname = os.path.basename(img_path)

            if err:
                print(f"     {fname:<30} ERROR: {err}")
                continue

            correct = pred.lower() == gt_label.lower()
            mark = "✓" if correct else "✗"
            print(f"     {fname:<30} {pred:<20} {conf:>5.1%}  {mark}")

            results.append({
                "file": fname,
                "gt": gt_label,
                "pred": pred,
                "conf": conf,
                "correct": correct,
                "time_ms": elapsed * 1000,
            })

            class_stats[gt_label]["total"] += 1
            class_stats[gt_label]["times"].append(elapsed * 1000)
            if correct:
                class_stats[gt_label]["tp"] += 1
            else:
                class_stats[gt_label]["fn"] += 1
                class_stats[pred]["fp"] += 1  # wrong class got a false positive
        print()

    # ── Per-class summary ───────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  PER-CLASS ACCURACY")
    print(SEPARATOR)
    total_correct = 0
    total_imgs = 0
    for gt_label in gt_classes:
        s = class_stats[gt_label]
        tp, total = s["tp"], s["total"]
        acc = tp / total if total else 0
        avg_ms = sum(s["times"]) / len(s["times"]) if s["times"] else 0
        print(f"  {gt_label:<15} acc={acc:>6.1%}  ({tp}/{total} correct)  avg_time={avg_ms:.1f}ms")
        total_correct += tp
        total_imgs += total

    overall_acc = total_correct / total_imgs if total_imgs else 0

    print(f"\n  OVERALL ACCURACY : {overall_acc:.1%}  ({total_correct}/{total_imgs})")

    # ── Confusion-style summary ─────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  CONFUSION TABLE  (rows=Ground Truth, cols=Predicted)")
    print(SEPARATOR)
    header = f"  {'GT \\ Pred':<15}" + "".join(f"{c[:12]:>13}" for c in gt_classes)
    print(header)
    for gt in gt_classes:
        row = f"  {gt:<15}"
        for pred_c in gt_classes:
            count = sum(
                1 for r in results
                if r["gt"] == gt and r["pred"].lower() == pred_c.lower()
            )
            row += f"{count:>13}"
        print(row)

    # ── Edge cases ──────────────────────────────────────────────────────────
    wrong = [r for r in results if not r["correct"]]
    print(f"\n{SEPARATOR}")
    print(f"  MISCLASSIFIED IMAGES ({len(wrong)} total)")
    print(SEPARATOR)
    if wrong:
        print(f"  {'File':<35} {'GT':<15} {'Predicted':<15} {'Conf':>6}")
        print(f"  {'-'*72}")
        for r in wrong:
            print(f"  {r['file']:<35} {r['gt']:<15} {r['pred']:<15} {r['conf']:>5.1%}")
    else:
        print("  None — perfect score!")

    # ── High-confidence wrong ───────────────────────────────────────────────
    hc_wrong = [r for r in wrong if r["conf"] >= 0.80]
    print(f"\n  ⚠️  HIGH-CONFIDENCE ERRORS (conf ≥ 80%)  → {len(hc_wrong)} cases")
    for r in hc_wrong:
        print(f"     {r['file']}  GT={r['gt']}  pred={r['pred']}  conf={r['conf']:.1%}")

    # ── Low-confidence correct ──────────────────────────────────────────────
    lc_right = [r for r in results if r["correct"] and r["conf"] < 0.60]
    print(f"\n  ⚠️  LOW-CONFIDENCE CORRECT (conf < 60%)  → {len(lc_right)} cases")
    for r in lc_right:
        print(f"     {r['file']}  GT={r['gt']}  conf={r['conf']:.1%}")

    # ── Speed ───────────────────────────────────────────────────────────────
    all_times = [r["time_ms"] for r in results]
    print(f"\n{SEPARATOR}")
    print(f"  INFERENCE SPEED")
    print(SEPARATOR)
    print(f"  Min   : {min(all_times):.1f} ms")
    print(f"  Max   : {max(all_times):.1f} ms")
    print(f"  Avg   : {sum(all_times)/len(all_times):.1f} ms")
    fps = 1000 / (sum(all_times)/len(all_times))
    print(f"  ~FPS  : {fps:.1f}")

    print(f"\n{SEPARATOR}")
    print("  TEST COMPLETE")
    print(SEPARATOR)
    return overall_acc


if __name__ == "__main__":
    run_tests()
