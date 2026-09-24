import cv2
import mediapipe as mp
import numpy as np
import random
import joblib

model = joblib.load("rps_model.pkl")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

choices = ["rock", "paper", "scissors"]

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    user_choice = None

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        features = []

        for landmark in hand.landmark:
            features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        prediction = model.predict([features])[0]
        user_choice = prediction

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

    computer_choice = random.choice(choices)

    if user_choice:
        if user_choice == computer_choice:
            game_result = "DRAW"
        elif (
            (user_choice == "rock" and computer_choice == "scissors") or
            (user_choice == "paper" and computer_choice == "rock") or
            (user_choice == "scissors" and computer_choice == "paper")
        ):
            game_result = "YOU WIN"
        else:
            game_result = "YOU LOSE"

        cv2.putText(
            frame,
            f"You: {user_choice}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Computer: {computer_choice}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            game_result,
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )

    cv2.imshow("Rock Paper Scissors", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
