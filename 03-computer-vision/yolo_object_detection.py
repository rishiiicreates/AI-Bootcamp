import cv2
from ultralytics import settings, YOLO

# Disable telemetry / crash-report syncing before loading model
settings.update({"sync": False})

# 1. Load the lightweight YOLOv8 Nano model (~6 MB, runs on CPU)
model = YOLO("yolov8n.pt")

# 2. Open the default webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Object Counter running — press 'q' in the video window to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab camera frame.")
        break

    # 3. Run object detection on the current frame
    results = model(frame, verbose=False)

    # 4. Count detected objects and render bounding boxes
    detected_count  = len(results[0].boxes)
    annotated_frame = results[0].plot()

    # 5. Overlay object count
    cv2.putText(
        annotated_frame,
        f"Objects detected: {detected_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # 6. Display
    cv2.imshow("Live Object Detection (YOLOv8)", annotated_frame)

    # 7. Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
