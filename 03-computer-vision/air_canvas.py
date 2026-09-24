import sys
import cv2
import numpy as np

# cv2.CAP_DSHOW prevents camera-light freeze on Windows; not needed on macOS/Linux
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: could not open webcam.")
    sys.exit(1)

# HSV colour range for a BLUE object — adjust if using red/green
LOWER_COLOR = np.array([100, 150,   0])
UPPER_COLOR = np.array([140, 255, 255])

canvas = None
prev_point = None

print("Air Canvas started! Wave a BLUE marker to draw.")
print("Press 'c' to clear | Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    # Initialise black canvas on first frame
    if canvas is None:
        canvas = np.zeros_like(frame)

    # A. Convert BGR → HSV (easier to isolate specific colours)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # B. Binary mask: white = colour detected, black = background
    mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
    mask = cv2.erode(mask,  None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

    # C. Find contours of detected shape
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    curr_point = None

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            M = cv2.moments(largest)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                curr_point = (cX, cY)
                cv2.circle(frame, curr_point, 8, (0, 255, 255), -1)

    # D. Draw line on canvas when marker moves
    if curr_point and prev_point:
        cv2.line(canvas, prev_point, curr_point, (0, 0, 255), 5)

    prev_point = curr_point

    combined = cv2.add(frame, canvas)
    cv2.putText(
        combined,
        "Press 'c' to Clear | Press 'q' to Quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow('Virtual Air Canvas', combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
