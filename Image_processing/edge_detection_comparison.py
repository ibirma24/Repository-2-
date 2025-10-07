"""
Edge Detection Comparison Script
Applies three different edge detection methods to XboxSeriesX.jpg:
1. Sobel edge detection (gradient magnitude-based)
2. Directional edge detection (45-degree edges)
3. Canny edge detection (OpenCV implementation)

Each method is displayed as an individual image for comparison.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sobel_edge_detector import sobel_edge_detector
from directional_edge_detector import directional_edge_detector

# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')

def load_and_prepare_image(image_path):
    """Load image and convert to grayscale for edge detection."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB for matplotlib display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to grayscale for edge detection
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img_rgb, img_gray

def apply_sobel_edge_detection(img_gray):
    """Apply Sobel edge detection based on gradient magnitude."""
    print("Applying Sobel edge detection...")
    # Use threshold of 50 for edge detection
    edge_map = sobel_edge_detector(img_gray, threshold=50)
    return edge_map

def apply_directional_edge_detection(img_gray):
    """Apply directional edge detection for ~45 degree edges."""
    print("Applying directional edge detection (45 degrees)...")
    # Use direction range for 45-degree edges (40-50 degrees)
    edge_map = directional_edge_detector(img_gray, direction_range=(40, 50))
    return edge_map

def apply_canny_edge_detection(img_gray):
    """Apply OpenCV's Canny edge detection algorithm."""
    print("Applying Canny edge detection...")
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.4)
    
    # Apply Canny edge detection with automatic threshold selection
    # Using ratio of 1:2 or 1:3 for low:high thresholds
    low_threshold = 50
    high_threshold = 150
    
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    return edges

def display_individual_results(original, sobel_edges, directional_edges, canny_edges):
    """Display each edge detection result as a separate figure."""
    
    # Create individual plots for each method
    methods = [
        ("Original Image", original, 'viridis'),
        ("Sobel Edge Detection (Gradient Magnitude)", sobel_edges, 'gray'),
        ("Directional Edge Detection (45° Edges)", directional_edges, 'gray'), 
        ("Canny Edge Detection (OpenCV)", canny_edges, 'gray')
    ]
    
    for i, (title, image, cmap) in enumerate(methods):
        plt.figure(figsize=(10, 8))
        
        if i == 0:  # Original image in color
            plt.imshow(image)
        else:  # Edge maps in grayscale
            plt.imshow(image, cmap=cmap)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save individual images
        filename = f"xbox_edge_detection_{i+1}_{title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('°', 'deg')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

def analyze_edge_detection_results(sobel_edges, directional_edges, canny_edges):
    """Provide quantitative analysis of the edge detection results."""
    
    print("\n" + "="*60)
    print("EDGE DETECTION ANALYSIS RESULTS")
    print("="*60)
    
    # Calculate edge pixel statistics
    sobel_edge_pixels = np.sum(sobel_edges > 0)
    directional_edge_pixels = np.sum(directional_edges > 0)
    canny_edge_pixels = np.sum(canny_edges > 0)
    
    total_pixels = sobel_edges.size
    
    print(f"Image dimensions: {sobel_edges.shape}")
    print(f"Total pixels: {total_pixels:,}")
    print()
    
    print("Edge Pixel Counts:")
    print(f"  Sobel:       {sobel_edge_pixels:,} pixels ({sobel_edge_pixels/total_pixels*100:.2f}%)")
    print(f"  Directional: {directional_edge_pixels:,} pixels ({directional_edge_pixels/total_pixels*100:.2f}%)")
    print(f"  Canny:       {canny_edge_pixels:,} pixels ({canny_edge_pixels/total_pixels*100:.2f}%)")
    print()
    
    # Calculate edge strength statistics for continuous outputs
    if sobel_edges.dtype != bool and np.max(sobel_edges) > 1:
        sobel_mean = np.mean(sobel_edges[sobel_edges > 0]) if sobel_edge_pixels > 0 else 0
        print(f"Average edge strength (Sobel): {sobel_mean:.2f}")
    
    print("\nMethod Characteristics:")
    print("  Sobel:       • Detects edges in all directions")
    print("               • Based on gradient magnitude threshold")
    print("               • Good for general edge detection")
    print()
    print("  Directional: • Specifically detects ~45° edges")
    print("               • Filters by gradient angle (40-50 degrees)")
    print("               • Highlights diagonal features")
    print()
    print("  Canny:       • Two-threshold approach with hysteresis")
    print("               • Noise reduction via Gaussian blur")
    print("               • Produces thin, connected edges")
    print()

def main():
    """Main function to run edge detection comparison."""
    
    print("EDGE DETECTION COMPARISON: Xbox Series X")
    print("="*50)
    
    # Load and prepare the image
    image_path = "Images/XboxSeriesX.jpg"
    
    try:
        img_rgb, img_gray = load_and_prepare_image(image_path)
        print(f"Successfully loaded: {image_path}")
        print(f"Image shape: {img_gray.shape}")
        print()
        
        # Apply all three edge detection methods
        sobel_edges = apply_sobel_edge_detection(img_gray)
        directional_edges = apply_directional_edge_detection(img_gray)
        canny_edges = apply_canny_edge_detection(img_gray)
        
        print("All edge detection methods applied successfully!")
        print()
        
        # Display results as individual images
        print("Creating individual visualizations...")
        display_individual_results(img_rgb, sobel_edges, directional_edges, canny_edges)
        
        # Analyze and compare results
        analyze_edge_detection_results(sobel_edges, directional_edges, canny_edges)
        
        print("="*60)
        print("EDGE DETECTION COMPARISON COMPLETE!")
        print("Check the generated PNG files for individual results.")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the XboxSeriesX.jpg file exists in the Images/ directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()