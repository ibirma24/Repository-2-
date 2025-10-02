import numpy as np
import cv2


def contrast_stretch(img, r_min, r_max):
    """
    Linearly maps the intensity range [r_min, r_max] to the full output range [0, 255]
    
    Parameters:
    img: input grayscale image (numpy array)
    r_min: minimum intensity value to stretch from
    r_max: maximum intensity value to stretch from
    
    Returns:
    new_img: contrast stretched image
    """
    if img is None:
        raise ValueError("Input image is None")
    
    # Convert to grayscale if color image using cv2
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Validate range parameters
    if r_min >= r_max:
        raise ValueError("r_min must be less than r_max")
    
    # Convert to float for calculations to avoid overflow
    img_float = img.astype(np.float32)
    
    # Apply contrast stretching formula using numpy operations
    # Linear mapping: new_value = ((old_value - r_min) / (r_max - r_min)) * 255
    stretched = ((img_float - r_min) / (r_max - r_min)) * 255
    
    # Clip values to [0, 255] range using numpy
    stretched = np.clip(stretched, 0, 255)
    
    # Convert back to uint8
    new_img = stretched.astype(np.uint8)
    
    return new_img