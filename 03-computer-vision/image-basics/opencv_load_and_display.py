import os
import cv2
import matplotlib.pyplot as plt

# Resolve path reliably regardless of current working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.normpath(os.path.join(script_dir, "..", "..", "assets", "cat.jpeg"))

if not os.path.exists(img_path):
    print(f"File not found at: {img_path}")
    print("Files currently in directory:", os.listdir("."))
else:
    img = cv2.imread(img_path)

    if img is None:
        print("OpenCV could not decode the image. Check if the file is corrupted.")
    else:
        print("Image loaded successfully!")
        print("Shape (Height, Width, Channels):", img.shape)

        # Convert BGR -> RGB so matplotlib displays correct colours
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display original
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title("Original")
        plt.axis("off")

        # Resize and display
        img_resized = cv2.resize(img, (400, 256))
        img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        plt.subplot(1, 2, 2)
        plt.imshow(img_resized_rgb)
        plt.title("Resized (400x256)")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
