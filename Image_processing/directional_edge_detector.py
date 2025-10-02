import numpy as np
from calculate_gradient import calculate_gradient_extended

def directional_edge_detector(img, direction_range):
    """
    Detect edges in specific direction range
    """
    _, grad_angle = calculate_gradient_extended(img)
    
    # Convert angles to degrees (0-180 range)
    grad_angle_deg = np.degrees(grad_angle) % 180
    
    directional_map = np.zeros_like(img)
    
    # Apply directional thresholding
    mask = (grad_angle_deg >= direction_range[0]) & (grad_angle_deg <= direction_range[1])
    directional_map[mask] = 255
    
    return directional_map