import numpy as np
import cv2


def median_filter(img, size=3):
    """
    Apply a median filter of the given size to the input image
    
    Basic process: For each window:
    1. Extract the pixel neighborhood
    2. Sort the values 
    3. Replace the center pixel with the median value
    
    Parameters:
    img: input image (numpy array)
    size: size of the median filter window (default=3, should be odd)
    
    Returns:
    new_img: median filtered image
    """
    if img is None:
        raise ValueError("Input image is None")
    
    # Convert to grayscale if color image using cv2
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ensure filter size is odd
    if size % 2 == 0:
        raise ValueError("Filter size must be odd")
    
    # Calculate padding size
    pad = size // 2
    
    # Apply border padding using cv2 (reflect padding for better edge handling)
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    
    # Initialize output image
    new_img = np.zeros_like(img)
    
    # Apply median filter to each pixel
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Extract pixel neighborhood (window)
            neighborhood = padded[i:i+size, j:j+size]
            
            # Sort values and find median using numpy
            # This replaces the center pixel with the median value
            new_img[i, j] = np.median(neighborhood)
    
    # Convert back to uint8
    new_img = new_img.astype(np.uint8)
    
    return new_img