import csv
import os
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Keep dataset folder relative to this script directory
dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
os.makedirs(dataset_dir, exist_ok=True)
csv_path = os.path.join(dataset_dir, "data.csv")

file = open(csv_path, "a", newline="")
writer = csv.writer(file)

cap = cv2.VideoCapture(0)

print("Collecting gesture data. Controls: R=Rock | P=Paper | S=Scissors | Q=Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    features = None

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        features = []
        for landmark in hand.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.putText(
        frame,
        "R: Rock | P: Paper | S: Scissors | Q: Quit",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Collect Dataset", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        if features:
            writer.writerow(features + ["rock"])
            print("Saved: rock")
        else:
            print("Hand not detected, move hand into frame")
    elif key == ord("p"):
        if features:
            writer.writerow(features + ["paper"])
            print("Saved: paper")
        else:
            print("Hand not detected, move hand into frame")
    elif key == ord("s"):
        if features:
            writer.writerow(features + ["scissors"])
            print("Saved: scissors")
        else:
            print("Hand not detected, move hand into frame")
    elif key == ord("q"):
        break

file.close()
hands.close()
cap.release()
cv2.destroyAllWindows()