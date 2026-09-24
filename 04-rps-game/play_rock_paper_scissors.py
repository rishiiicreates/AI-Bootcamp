import cv2
import mediapipe as mp
import random
import joblib
import os

# Load trained gesture classifier
model_path = os.path.join(os.path.dirname(__file__), "rps_model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(
        "rps_model.pkl not found. Run collect_hand_gesture_data.py "
        "first, then train and save a model as rps_model.pkl."
    )
model = joblib.load(model_path)

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
choices = ["rock", "paper", "scissors"]

# State persisted across frames so the result doesn't flicker
user_choice     = None
computer_choice = None
game_result     = None

print("Rock Paper Scissors — show your hand gesture. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        features = []
        for landmark in hand.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])

        new_user_choice = model.predict([features])[0]

        # Only update computer move and result when gesture changes
        if new_user_choice != user_choice:
            user_choice     = new_user_choice
            computer_choice = random.choice(choices)   # pick once per gesture

            if user_choice == computer_choice:
                game_result = "DRAW"
            elif (
                (user_choice == "rock"     and computer_choice == "scissors") or
                (user_choice == "paper"    and computer_choice == "rock")     or
                (user_choice == "scissors" and computer_choice == "paper")
            ):
                game_result = "YOU WIN!"
            else:
                game_result = "YOU LOSE"

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
    else:
        # No hand visible — reset state
        user_choice = computer_choice = game_result = None

    # Overlay result on frame
    if user_choice:
        cv2.putText(frame, f"You: {user_choice}",         (20,  40), cv2.FONT_HERSHEY_SIMPLEX, 1,   (255, 255, 255), 2)
        cv2.putText(frame, f"CPU: {computer_choice}",     (20,  80), cv2.FONT_HERSHEY_SIMPLEX, 1,   (255, 255, 255), 2)
        color = (0, 255, 0) if "WIN" in game_result else (0, 0, 255) if "LOSE" in game_result else (200, 200, 0)
        cv2.putText(frame, game_result,                   (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)

    cv2.imshow("Rock Paper Scissors", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
