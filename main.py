import cv2
import numpy as np
from matplotlib import pyplot as plt

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to binarize the image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
    
    # Apply morphological transformations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    return binary

def deskew_image(image_path, hough_threshold=300):

    image = cv2.imread(image_path)
    
    # Check if the image was loaded correctly
    if image is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}")
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Edge detection
    edges = cv2.Canny(preprocessed_image, 50, 150, apertureSize=3)
    
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_threshold)
    
    # Visualize the lines for debugging
    line_image = np.copy(image)
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Display the original image with detected lines
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Lines'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    # Calculate the skew angle
    angles = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = (theta - np.pi/2) * (180 / np.pi)
            # Filter out near-vertical angles
            if -80 < angle < 80:  # Adjust this range
                angles.append(angle)
    if len(angles) == 0:
        return image  # No lines detected, return the original image

    # Use the median angle to deskew the image
    median_angle = np.median(angles)
    
    # Normalize the angle to be within -45 to 45 degrees
    if median_angle < -45:
        median_angle += 90
    elif median_angle > 45:
        median_angle -= 90
    
    # Rotate the image to correct the skew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    return rotated

# Paths to the input and output directories
input_directory = 'test_images/'
output_directory = 'output/'

try:
    for i in range(1, 6):
        input_image_path = f'{input_directory}{i}.jpg'
        output_image_path = f'{output_directory}{i}_deskewed.jpg'

        # Deskew the image
        deskewed_image = deskew_image(input_image_path, hough_threshold=150)  # Adjust this value as needed

        # Save the result
        cv2.imwrite(output_image_path, deskewed_image)

        # Display the images (optional)
        plt.subplot(121), plt.imshow(cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB))
        plt.title(f'Input Image {i}'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(cv2.cvtColor(deskewed_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Deskewed Image {i}'), plt.xticks([]), plt.yticks([])
        plt.show()

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")
