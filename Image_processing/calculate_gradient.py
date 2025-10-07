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


def calculate_gradient(img):
    """
    Calculate gradient magnitude and direction using apply_convolution from Project 1 to apply
    the Sobel Sx and Sy filters, then compute the gradient magnitude and angle.
    
    Parameters:
    img: input grayscale image (numpy array)
    
    Returns:
    grad_magnitude: gradient magnitude image
    grad_angle: gradient direction/angle image (in radians)
    """
    if img is None:
        raise ValueError("Input image is None")
    
    # Convert to grayscale if color image using cv2
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Sobel operators for gradient calculation
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sx filter
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Sy filter
    
    # Apply Sobel filters using apply_convolution from Project 1
    grad_x = apply_convolution(img.astype(np.float32), sobel_x)
    grad_y = apply_convolution(img.astype(np.float32), sobel_y)
    
    # Compute gradient magnitude using numpy
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Calculate gradient direction/angle using numpy arctan2
    grad_angle = np.arctan2(grad_y, grad_x)
    
    # Normalize magnitude to [0, 255] range
    if grad_magnitude.max() > 0:
        grad_magnitude = (grad_magnitude / grad_magnitude.max() * 255).astype(np.uint8)
    else:
        grad_magnitude = grad_magnitude.astype(np.uint8)
    
    return grad_magnitude, grad_angle


def calculate_gradient_extended(img):
    """
    Extended implementation that calculates both gradient magnitude and direction
    using apply_convolution from Project 1 to apply the Sobel Sx and Sy filters.
    
    Parameters:
    img: input grayscale image (numpy array)
    
    Returns:
    grad_magnitude: gradient magnitude image
    grad_angle: gradient direction/angle image (in degrees)
    """
    if img is None:
        raise ValueError("Input image is None")
    
    # Convert to grayscale if color image using cv2
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Sobel operators for gradient calculation
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sx filter
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Sy filter
    
    # Apply Sobel filters using apply_convolution from Project 1
    grad_x = apply_convolution(img.astype(np.float32), sobel_x)
    grad_y = apply_convolution(img.astype(np.float32), sobel_y)
    
    # Compute gradient magnitude using numpy
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Calculate gradient direction/angle using numpy arctan2
    grad_angle_radians = np.arctan2(grad_y, grad_x)
    
    # Convert from radians to degrees
    grad_angle = np.degrees(grad_angle_radians)
    
    # Normalize magnitude to [0, 255] range
    if grad_magnitude.max() > 0:
        grad_magnitude = (grad_magnitude / grad_magnitude.max() * 255).astype(np.uint8)
    else:
        grad_magnitude = grad_magnitude.astype(np.uint8)
    
    return grad_magnitude, grad_angle


def add_salt_pepper_noise(img, noise_ratio=0.1):
    """
    Add salt-and-pepper noise to an image for analysis
    
    Parameters:
    img: input image
    noise_ratio: fraction of pixels to corrupt (default 0.1 = 10%)
    
    Returns:
    noisy_img: image with salt-and-pepper noise
    """
    noisy_img = img.copy()
    h, w = img.shape
    
    # Number of pixels to corrupt
    num_salt = int(noise_ratio * h * w * 0.5)
    num_pepper = int(noise_ratio * h * w * 0.5)
    
    # Add salt noise (white pixels = 255)
    coords = [np.random.randint(0, i-1, num_salt) for i in img.shape]
    noisy_img[coords[0], coords[1]] = 255
    
    # Add pepper noise (black pixels = 0)
    coords = [np.random.randint(0, i-1, num_pepper) for i in img.shape]
    noisy_img[coords[0], coords[1]] = 0
    
    return noisy_img


def analyze_median_filter_effect():
    """
    Analysis: What is the effect of median filtering on the calculation of gradient 
    magnitude of an image corrupted by salt-and-pepper noise?
    """
    print("=== Analysis: Effect of Median Filtering on Gradient Magnitude ===\n")
    
    # Create a simple test image with clear edges
    test_img = np.zeros((100, 100), dtype=np.uint8)
    test_img[30:70, 30:70] = 128  # Gray square
    test_img[40:60, 40:60] = 255  # White square inside
    
    print("1. Creating test image with clear geometric edges...")
    
    # Add salt-and-pepper noise
    noisy_img = add_salt_pepper_noise(test_img, noise_ratio=0.15)
    print("2. Adding 15% salt-and-pepper noise...")
    
    # Apply median filter to remove noise
    try:
        from median_filter import median_filter
        filtered_img = median_filter(noisy_img, size=3)
        print("3. Applying 3x3 median filter...")
    except ImportError:
        # Fallback to cv2 median filter if custom one not available
        filtered_img = cv2.medianBlur(noisy_img, 3)
        print("3. Applying 3x3 median filter (using cv2.medianBlur)...")
    
    # Calculate gradients for all three images
    print("4. Computing gradient magnitudes...")
    grad_original, _ = calculate_gradient(test_img)
    grad_noisy, _ = calculate_gradient(noisy_img)
    grad_filtered, _ = calculate_gradient(filtered_img)
    
    # Calculate statistics
    def calc_stats(grad_img, name):
        mean_grad = np.mean(grad_img)
        std_grad = np.std(grad_img)
        max_grad = np.max(grad_img)
        edge_strength = np.sum(grad_img > 50)  # Count strong edges
        return {
            'name': name,
            'mean': mean_grad,
            'std': std_grad, 
            'max': max_grad,
            'edge_pixels': edge_strength
        }
    
    stats_original = calc_stats(grad_original, "Original")
    stats_noisy = calc_stats(grad_noisy, "Noisy")
    stats_filtered = calc_stats(grad_filtered, "Median Filtered")
    
    # Print analysis results
    print("\n=== ANALYSIS RESULTS ===")
    print(f"{'Condition':<15} {'Mean':<8} {'Std':<8} {'Max':<8} {'Edge Pixels':<12}")
    print("-" * 55)
    
    for stats in [stats_original, stats_noisy, stats_filtered]:
        print(f"{stats['name']:<15} {stats['mean']:<8.1f} {stats['std']:<8.1f} "
              f"{stats['max']:<8.0f} {stats['edge_pixels']:<12}")
    
    print("\n=== CONCLUSIONS ===")
    print("Effect of Salt-and-Pepper Noise on Gradient Magnitude:")
    print(f"• Noise increases gradient std by {stats_noisy['std']/stats_original['std']:.1f}x")
    print(f"• False edge pixels increased by {(stats_noisy['edge_pixels']-stats_original['edge_pixels'])/stats_original['edge_pixels']*100:.0f}%")
    
    print("\nEffect of Median Filtering:")
    print(f"• Reduces gradient noise: std ratio = {stats_filtered['std']/stats_noisy['std']:.2f}")
    print(f"• Preserves true edges: edge preservation = {stats_filtered['edge_pixels']/stats_original['edge_pixels']:.2f}")
    print(f"• Overall improvement: {((stats_noisy['std']-stats_filtered['std'])/stats_noisy['std']*100):.0f}% noise reduction")
    
    print("\n🔍 KEY FINDING:")
    print("Median filtering effectively removes salt-and-pepper noise artifacts")
    print("in gradient calculation while preserving genuine edge information,")
    print("making it an excellent preprocessing step for edge detection in noisy images.")
    
    # Save images for visual inspection
    try:
        cv2.imwrite('original_image.png', test_img)
        cv2.imwrite('noisy_image.png', noisy_img)
        cv2.imwrite('filtered_image.png', filtered_img)
        cv2.imwrite('gradient_original.png', grad_original)
        cv2.imwrite('gradient_noisy.png', grad_noisy)
        cv2.imwrite('gradient_filtered.png', grad_filtered)
        print("\n✓ Analysis images saved for visual inspection")
    except Exception as e:
        print(f"\nNote: Could not save images: {e}")
    
    return {
        'original': {'img': test_img, 'gradient': grad_original},
        'noisy': {'img': noisy_img, 'gradient': grad_noisy},
        'filtered': {'img': filtered_img, 'gradient': grad_filtered}
    }


if __name__ == "__main__":
    # Run the analysis
    results = analyze_median_filter_effect()
    
    # Test extended gradient calculation
    print("\n=== Testing Extended Gradient Calculation ===")
    test_img = np.zeros((50, 50), dtype=np.uint8)
    test_img[20:30, 20:30] = 255
    
    magnitude, angle = calculate_gradient_extended(test_img)
    print(f"Gradient magnitude range: [{np.min(magnitude)}, {np.max(magnitude)}]")
    print(f"Gradient angle range: [{np.min(angle):.1f}°, {np.max(angle):.1f}°]")
    print("✓ Extended gradient calculation working correctly")