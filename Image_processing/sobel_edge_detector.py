import numpy as np
import cv2


def apply_convolution(image, kernel):
    """
    Apply convolution operation (from Project 1)
    """
    kernel = np.flipud(np.fliplr(kernel))
    output = np.zeros_like(image, dtype=np.float32)
    pad = kernel.shape[0] // 2
    
    padded = np.pad(image, pad, mode='constant')
    
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            output[y, x] = (kernel * padded[y:y+kernel.shape[0], x:x+kernel.shape[1]]).sum()
    
    return output


def sobel_edge_detector(img, threshold):
    """
    Apply Sobel edge detection with binary thresholding to produce a clean binary edge map.
    
    Process:
    1. Apply Sobel filters to the input image
    2. Calculate gradient magnitude
    3. Apply binary threshold (pixels above threshold → 255, below → 0)
    
    Parameters:
    img: input image (numpy array)
    threshold: threshold value for binary edge map creation
    
    Returns:
    edge_map: clean binary edge map (0 or 255 values)
    """
    if img is None:
        raise ValueError("Input image is None")
    
    # Convert to grayscale if color image using cv2
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Sobel operators for edge detection
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sx filter
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Sy filter
    
    # Apply Sobel filters to input image
    grad_x = apply_convolution(img.astype(np.float32), sobel_x)
    grad_y = apply_convolution(img.astype(np.float32), sobel_y)
    
    # Calculate gradient magnitude using numpy
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Apply binary threshold to produce clean binary edge map
    # Set pixels above threshold to 255, below to 0
    edge_map = np.zeros_like(grad_magnitude, dtype=np.uint8)
    edge_map[grad_magnitude > threshold] = 255
    
    return edge_map


def sobel_edge_detector_with_normalization(img, threshold):
    """
    Alternative Sobel edge detector that normalizes gradient magnitude before thresholding.
    Useful when you want threshold values in range [0, 255].
    
    Parameters:
    img: input image (numpy array)
    threshold: threshold value (0-255) for binary edge map creation
    
    Returns:
    edge_map: clean binary edge map (0 or 255 values)
    """
    if img is None:
        raise ValueError("Input image is None")
    
    # Convert to grayscale if color image using cv2
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Sobel operators for edge detection
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Apply Sobel filters
    grad_x = apply_convolution(img.astype(np.float32), sobel_x)
    grad_y = apply_convolution(img.astype(np.float32), sobel_y)
    
    # Calculate gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient magnitude to [0, 255] range
    if grad_magnitude.max() > 0:
        grad_magnitude_normalized = (grad_magnitude / grad_magnitude.max() * 255)
    else:
        grad_magnitude_normalized = grad_magnitude
    
    # Apply binary threshold
    edge_map = np.zeros_like(grad_magnitude_normalized, dtype=np.uint8)
    edge_map[grad_magnitude_normalized > threshold] = 255
    
    return edge_map


def demonstrate_sobel_edge_detection():
    """
    Demonstrate Sobel edge detection with different threshold values
    """
    print("=== Sobel Edge Detection Demonstration ===\n")
    
    # Create a test image with various features
    test_img = np.zeros((100, 100), dtype=np.uint8)
    
    # Add geometric shapes with different intensities
    test_img[20:80, 20:80] = 100    # Gray square
    test_img[30:70, 30:70] = 180    # Lighter square
    test_img[40:60, 40:60] = 255    # White square
    
    # Add a diagonal line
    for i in range(100):
        if 0 <= i < 100 and 0 <= i < 100:
            test_img[i, i] = 50
    
    print("1. Created test image with geometric shapes and lines")
    
    # Test different threshold values
    thresholds = [20, 50, 100, 150]
    
    print("2. Testing different threshold values:")
    for thresh in thresholds:
        edge_map = sobel_edge_detector(test_img, thresh)
        edge_pixels = np.sum(edge_map == 255)
        print(f"   Threshold {thresh:3d}: {edge_pixels:4d} edge pixels detected")
    
    # Compare with normalized version
    print("\n3. Comparing with normalized threshold approach:")
    for thresh in [50, 100, 150, 200]:
        edge_map_norm = sobel_edge_detector_with_normalization(test_img, thresh)
        edge_pixels_norm = np.sum(edge_map_norm == 255)
        print(f"   Normalized threshold {thresh:3d}: {edge_pixels_norm:4d} edge pixels detected")
    
    # Save example results
    try:
        cv2.imwrite('test_image.png', test_img)
        edge_map_example = sobel_edge_detector(test_img, 50)
        cv2.imwrite('sobel_edges_threshold_50.png', edge_map_example)
        edge_map_norm_example = sobel_edge_detector_with_normalization(test_img, 100)
        cv2.imwrite('sobel_edges_normalized_100.png', edge_map_norm_example)
        print("\n✓ Example images saved successfully")
    except Exception as e:
        print(f"\nNote: Could not save images: {e}")
    
    print("\n=== Sobel Edge Detection Complete ===")
    print("✓ Binary threshold applied: pixels above threshold → 255, below → 0")
    print("✓ Clean binary edge map produced")
    print("✓ Multiple threshold options available")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_sobel_edge_detection()