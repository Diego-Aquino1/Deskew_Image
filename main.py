import cv2
import numpy as np
from matplotlib import pyplot as plt

def deskew_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded correctly
    if image is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    # Calculate the skew angle
    angles = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = (theta - np.pi/2) * (180 / np.pi)
            # Consider only near-horizontal lines
            if -45 < angle < 45:
                angles.append(angle)
    if len(angles) == 0:
        return image  # No lines detected, return the original image

    median_angle = np.median(angles)
    
    # Rotate the image to correct the skew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    return rotated

# Path to the input image
input_image_path = 'test_images/2.jpg'

# Deskew the image
deskewed_image = deskew_image(input_image_path)

# Save or display the result
output_image_path = '/output/deskewed_image_1.jpg'
cv2.imwrite(output_image_path, deskewed_image)

# Display the images (optional)
plt.subplot(121), plt.imshow(cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB))
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(deskewed_image, cv2.COLOR_BGR2RGB))
plt.title('Deskewed Image'), plt.xticks([]), plt.yticks([])
plt.show()

output_image_path
