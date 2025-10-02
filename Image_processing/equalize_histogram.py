import numpy as np
import cv2


def equalize_histogram(img):
    """
    Implements the full histogram equalization process to enhance image contrast
    
    Parameters:
    img: input grayscale image (numpy array)
    
    Returns:
    new_img: histogram equalized image
    """
    if img is None:
        raise ValueError("Input image is None")
    
    # Convert to grayscale if color image
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Calculate histogram for 256 bins (0-255 intensity levels)
    counts, _ = np.histogram(img.flatten(), bins=256, range=[0, 255])
    
    # Step 2: Calculate normalized probability distribution
    total_pixels = img.size
    dist = counts.astype(np.float32) / total_pixels
    
    # Step 3: Calculate cumulative distribution function (CDF)
    cdf = np.cumsum(dist)
    
    # Step 4: Create transformation function
    # Map CDF values to new intensity range [0, 255]
    transform_map = np.round(cdf * 255).astype(np.uint8)
    
    # Step 5: Apply transformation to image using numpy indexing
    # Each pixel value is replaced by its corresponding transformed value
    new_img = transform_map[img]
    
    return new_img


