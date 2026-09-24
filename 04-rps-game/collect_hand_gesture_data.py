import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

os.makedirs("dataset", exist_ok=True)

file = open("dataset/data.csv", "a", newline="")
writer = csv.writer(file)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        features = []

        for landmark in hand.landmark:
            features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        cv2.putText(
            frame,
            "R = Rock | P = Paper | S = Scissors | Q = Quit",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            writer.writerow(features + ["rock"])
            print("Rock saved")

        elif key == ord("p"):
            writer.writerow(features + ["paper"])
            print("Paper saved")

        elif key == ord("s"):
            writer.writerow(features + ["scissors"])
            print("Scissors saved")

        elif key == ord("q"):
            break

    cv2.imshow("Collect Dataset", frame)

file.close()
cap.release()
cv2.destroyAllWindows()