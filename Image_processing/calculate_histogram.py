import numpy as np
import cv2


def calculate_histogram(img, bins):
    """
    Calculate the histogram (counts) and normalized histogram (probability distribution) 
    of a grayscale image for the given number of bins across the entire intensity range [0, 255]
    
    Parameters:
    img: input grayscale image (numpy array)
    bins: number of bins for histogram calculation
    
    Returns:
    counts: histogram counts for each bin
    dist: normalized histogram (probability distribution)
    """
    if img is None:
        raise ValueError("Input image is None")
    
    # Convert to grayscale if color image using cv2
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram using numpy across full intensity range [0, 255]
    counts, _ = np.histogram(img.flatten(), bins=bins, range=[0, 255])
    
    # Calculate normalized histogram (probability distribution)
    # Each bin value divided by total number of pixels
    total_pixels = img.size
    dist = counts.astype(np.float32) / total_pixels
    
    return counts, dist