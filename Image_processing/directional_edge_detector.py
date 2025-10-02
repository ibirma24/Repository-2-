import numpy as np
import cv2
from calculate_gradient import calculate_gradient_extended


def directional_edge_detector(img, direction_range):
    """
    A function that uses calculate_gradient() to obtain the gradient direction map and
    then applies thresholding based on the values of direction_range to return a binary directional map.
    
    Parameters:
    img: input image (numpy array)
    direction_range: tuple (min_angle, max_angle) in degrees, e.g., (40, 50)
    
    Returns:
    edge_directional_map: binary directional map (0 or 255 values)
    """
    if img is None:
        raise ValueError("Input image is None")
    
    if len(direction_range) != 2:
        raise ValueError("direction_range must be a tuple of (min_angle, max_angle)")
    
    # Convert to grayscale if color image using cv2
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use calculate_gradient() to obtain the gradient direction map
    _, grad_angle = calculate_gradient_extended(img)
    
    # Convert angles from radians to degrees and normalize to [0, 180] range
    # Using 0-180 range because edge direction is independent of gradient sign
    grad_angle_deg = np.abs(np.degrees(grad_angle)) % 180
    
    # Create binary directional map
    edge_directional_map = np.zeros_like(img, dtype=np.uint8)
    
    # Apply thresholding based on direction_range values
    min_angle, max_angle = direction_range
    
    # Handle normal ranges (e.g., (40, 50)) and wrapped ranges (e.g., (170, 10))
    if min_angle <= max_angle:
        # Normal range: edges between min_angle and max_angle
        directional_mask = (grad_angle_deg >= min_angle) & (grad_angle_deg <= max_angle)
    else:
        # Wrapped range: edges from min_angle to 180 and from 0 to max_angle
        directional_mask = (grad_angle_deg >= min_angle) | (grad_angle_deg <= max_angle)
    
    # Create binary directional map: set pixels in direction range to 255
    edge_directional_map[directional_mask] = 255
    
    return edge_directional_map