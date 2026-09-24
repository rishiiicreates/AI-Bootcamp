import os
import cv2
import matplotlib.pyplot as plt

# Update this path or extension (.jpg / .png) to match your exact file name
img_path = "cat.jpeg"

if not os.path.exists(img_path):
    print(f"File not found at: {img_path}")
    print("Files currently in directory:", os.listdir("/Users/rishii/AI-Bootcamp"))
else:
    img = cv2.imread(img_path)
    
    if img is None:
        print("OpenCV could not decode the image file. Check if the file is corrupted or an unsupported format.")
    else:
        print("Image loaded successfully!")
        print("Shape (Height, Width, Channels):", img.shape)
    
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.show()



img_resize = cv2_resize(img_path,(400,256))
cv2.imshow("img",

