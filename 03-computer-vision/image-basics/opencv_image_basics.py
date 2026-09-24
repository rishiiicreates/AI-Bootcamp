import os
import cv2

# Resolve path reliably regardless of working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.normpath(os.path.join(script_dir, "..", "..", "assets", "cat.jpeg"))

img = cv2.imread(img_path)

if img is None:
    print(f"Error: could not load image from '{img_path}'")
    print("Make sure assets/cat.jpeg exists in the repo root.")
else:
    print("Image type :", type(img))
    print("Shape (H, W, C):", img.shape)

    cv2.imshow("Original", img)
    cv2.waitKey(0)

    img_resized = cv2.resize(img, (256, 256))
    cv2.imshow("Resized (256×256)", img_resized)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
