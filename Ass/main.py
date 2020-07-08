frame='img-389.png'
import cv2

# Apply Gaussian blur
blur = cv2.GaussianBlur(frame)
cv2.imshow(blur)