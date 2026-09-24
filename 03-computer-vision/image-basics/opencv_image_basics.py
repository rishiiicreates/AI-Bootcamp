import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("cat.jpeg")

print(type(img))
print(img.shape)

cv2.imshow("original", img)
cv2.waitKey(0)

img_resize = cv2.resize(img, (256, 256))

cv2.imshow("resized", img_resize)
cv2.waitKey(0)

cv2.destroyAllWindows()
