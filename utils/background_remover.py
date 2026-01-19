import cv2
import numpy as np

# Load image
img = cv2.imread("demo/pcb/pcb_0002_NG_ZW_C1_20231028093914.jpg")  # BGR

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to separate object from white background
_, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

# Find contours of the object
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# If at least one contour found
if contours:
    # Get bounding box of the largest contour
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Crop the image
    object_only = img[y:y+h, x:x+w]

    # Save
    cv2.imwrite("demo/pcb/pcb_0044_NG_ZW_C1_20231028093914.jpg", object_only)
else:
    print("No object detected!")
