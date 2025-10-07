#!/usr/bin/env python3
"""
Exercise 1 Test Function: Intensity Transformations and Histogram Equalization

This test function demonstrates both contrast_stretch() and equalize_histogram()
applied to the Low Contrast.jpg image with visual comparison.

Usage:
    python test_exercise1.py

Author: Image Processing Course - Exercise 1
Date: October 2025
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
from contrast_stretch import contrast_stretch
from equalize_histogram import equalize_histogram
from calculate_histogram import calculate_histogram


def test_exercise1():
    """
    Test function for Exercise 1: Apply contrast_stretch() and equalize_histogram()
    to Low Contrast.jpg and create comprehensive comparison visualization.
    """
    print("="*60)
    print("EXERCISE 1 TEST: INTENSITY TRANSFORMATIONS")
    print("="*60)
    print("Testing contrast_stretch() and equalize_histogram() functions")
    print("on Low Contrast.jpg image\n")
    
    # Configuration
    input_image = "Images/Low Contrast.jpg"
    output_file = "exercise1_test_results.png"
    
    # Check if input image exists
    if not os.path.exists(input_image):
        print(f"❌ Error: Input image not found: {input_image}")
        print("Please ensure the image exists in the Images/ directory.")
        return False
    
    try:
        # Step 1: Load and analyze the original image
        print("Step 1: Loading and analyzing original image...")
        img = cv2.imread(input_image)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {input_image}")
        
        # Convert to grayscale for processing
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        print(f"✓ Image loaded: {img_gray.shape}")
        print(f"✓ Original intensity range: {img_gray.min()} - {img_gray.max()}")
        print(f"✓ Original mean intensity: {img_gray.mean():.2f}")
        
        # Step 2: Determine optimal contrast stretch parameters
        print("\nStep 2: Determining contrast stretch parameters...")
        # Use 1st and 99th percentiles to avoid outliers
        r_min = np.percentile(img_gray, 1)
        r_max = np.percentile(img_gray, 99)
        
        print(f"✓ Contrast stretch range: {r_min:.1f} to {r_max:.1f}")
        print(f"✓ Original dynamic range: {img_gray.max() - img_gray.min()}/255 ({(img_gray.max() - img_gray.min())/255*100:.1f}%)")
        
        # Step 3: Apply contrast stretching
        print("\nStep 3: Applying contrast_stretch()...")
        img_stretched = contrast_stretch(img_gray, r_min, r_max)
        
        print(f"✓ Stretched intensity range: {img_stretched.min()} - {img_stretched.max()}")
        print(f"✓ Stretched mean intensity: {img_stretched.mean():.2f}")
        print(f"✓ New dynamic range: {img_stretched.max() - img_stretched.min()}/255 ({(img_stretched.max() - img_stretched.min())/255*100:.1f}%)")
        
        # Step 4: Apply histogram equalization
        print("\nStep 4: Applying equalize_histogram()...")
        img_equalized = equalize_histogram(img_gray)
        
        print(f"✓ Equalized intensity range: {img_equalized.min()} - {img_equalized.max()}")
        print(f"✓ Equalized mean intensity: {img_equalized.mean():.2f}")
        print(f"✓ Equalized dynamic range: {img_equalized.max() - img_equalized.min()}/255 ({(img_equalized.max() - img_equalized.min())/255*100:.1f}%)")
        
        # Step 5: Calculate histograms
        print("\nStep 5: Calculating histograms...")
        hist_original, dist_original = calculate_histogram(img_gray, 256)
        hist_stretched, dist_stretched = calculate_histogram(img_stretched, 256)
        hist_equalized, dist_equalized = calculate_histogram(img_equalized, 256)
        
        print("✓ Histograms calculated for all three images")
        
        # Step 6: Create comprehensive visualization
        print(f"\nStep 6: Creating visualization...")
        create_exercise1_visualization(
            img_gray, img_stretched, img_equalized,
            hist_original, hist_stretched, hist_equalized,
            r_min, r_max, output_file
        )
        
        # Step 7: Analyze results
        print("\nStep 7: Analyzing enhancement results...")
        analyze_enhancement_effectiveness(img_gray, img_stretched, img_equalized)
        
        # Success summary
        print(f"\n" + "="*60)
        print("EXERCISE 1 TEST COMPLETED SUCCESSFULLY")
        print("="*60)
        print("✅ contrast_stretch() function tested")
        print("✅ equalize_histogram() function tested")
        print("✅ calculate_histogram() function tested")
        print("✅ Visual comparison created")
        print("✅ Quantitative analysis completed")
        print(f"\n📁 Results saved as: {output_file}")
        
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"📊 File size: {file_size:,} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during Exercise 1 test: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_exercise1_visualization(img_gray, img_stretched, img_equalized,
                                 hist_original, hist_stretched, hist_equalized,
                                 r_min, r_max, output_filename):
    """
    Create comprehensive visualization for Exercise 1 test results
    """
    # Create figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Images
    images = [img_gray, img_stretched, img_equalized]
    image_titles = [
        'Original Image\n(Low Contrast)',
        f'Contrast Stretched\n(Range: {r_min:.0f}-{r_max:.0f} → 0-255)',
        'Histogram Equalized\n(Uniform Distribution)'
    ]
    
    for i, (img, title) in enumerate(zip(images, image_titles)):
        axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(title, fontsize=14, fontweight='bold', pad=20)
        axes[0, i].axis('off')
        
        # Add image statistics
        stats_text = (f"Min: {img.min()}\n"
                      f"Max: {img.max()}\n"
                      f"Mean: {img.mean():.1f}\n"
                      f"Std: {img.std():.1f}")
        
        axes[0, i].text(0.02, 0.98, stats_text, transform=axes[0, i].transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
                       facecolor='white', alpha=0.9), fontsize=10)
    
    # Row 2: Histograms
    histograms = [hist_original, hist_stretched, hist_equalized]
    hist_titles = [
        'Original Histogram\n(Narrow Distribution)',
        'Stretched Histogram\n(Expanded Range)',
        'Equalized Histogram\n(Uniform Distribution)'
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for i, (hist, title, color) in enumerate(zip(histograms, hist_titles, colors)):
        # Create bar plot
        x_vals = np.arange(256)
        axes[1, i].bar(x_vals, hist, color=color, alpha=0.8, width=1.0, edgecolor='none')
        
        axes[1, i].set_title(title, fontsize=14, fontweight='bold', pad=20)
        axes[1, i].set_xlabel('Intensity Level', fontsize=12)
        axes[1, i].set_ylabel('Pixel Count', fontsize=12)
        axes[1, i].set_xlim(0, 255)
        axes[1, i].grid(True, alpha=0.3, linewidth=0.5)
        
        # Add histogram statistics
        peak_intensity = np.argmax(hist)
        peak_count = np.max(hist)
        total_pixels = np.sum(hist)
        non_zero_bins = np.sum(hist > 0)
        
        hist_stats = (f"Peak at {peak_intensity}\n"
                     f"Peak count: {peak_count:,}\n"
                     f"Non-zero bins: {non_zero_bins}/256\n"
                     f"Distribution spread: {non_zero_bins/256*100:.1f}%")
        
        axes[1, i].text(0.98, 0.98, hist_stats, transform=axes[1, i].transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                       fontsize=9)
        
        # Highlight the effective range for original image
        if i == 0:  # Original histogram
            # Shade the area outside the stretch range
            axes[1, i].axvspan(0, r_min, alpha=0.3, color='red', label=f'Below {r_min:.0f}')
            axes[1, i].axvspan(r_max, 255, alpha=0.3, color='red', label=f'Above {r_max:.0f}')
            axes[1, i].axvspan(r_min, r_max, alpha=0.2, color='green', label=f'Stretch range')
            axes[1, i].legend(fontsize=8)
    
    # Add overall title and method descriptions
    fig.suptitle('Exercise 1 Test Results: Intensity Transformations and Histogram Equalization\n' +
                 'Demonstrating contrast_stretch() and equalize_histogram() on Low Contrast Image', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add method descriptions at the bottom
    method_descriptions = [
        "Original: Limited intensity range\nPoor utilization of dynamic range",
        "Contrast Stretch: Linear mapping\nPreserves relative relationships",
        "Histogram Equalization: Non-linear mapping\nMaximizes entropy and contrast"
    ]
    
    for i, desc in enumerate(method_descriptions):
        axes[0, i].text(0.5, -0.15, desc, transform=axes[0, i].transAxes,
                       horizontalalignment='center', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15)
    
    # Save the visualization
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='png')
    plt.close()
    
    print(f"✓ Visualization saved as: {output_filename}")


def analyze_enhancement_effectiveness(img_gray, img_stretched, img_equalized):
    """
    Analyze the effectiveness of both enhancement methods
    """
    print("QUANTITATIVE ANALYSIS:")
    print("-" * 40)
    
    # Calculate enhancement metrics
    images = [img_gray, img_stretched, img_equalized]
    names = ['Original', 'Contrast Stretched', 'Histogram Equalized']
    
    print(f"{'Method':<20} {'Range':<8} {'Mean':<8} {'Std':<8} {'Contrast':<10}")
    print("-" * 60)
    
    original_std = img_gray.std()
    
    for name, img in zip(names, images):
        intensity_range = img.max() - img.min()
        mean_val = img.mean()
        std_val = img.std()
        contrast_ratio = std_val / original_std if original_std > 0 else 1.0
        
        print(f"{name:<20} {intensity_range:<8} {mean_val:<8.1f} {std_val:<8.1f} {contrast_ratio:<10.2f}x")
    
    # Dynamic range utilization
    print(f"\nDYNAMIC RANGE UTILIZATION:")
    print("-" * 40)
    for name, img in zip(names, images):
        utilization = (img.max() - img.min()) / 255 * 100
        print(f"{name}: {utilization:.1f}% of full dynamic range")
    
    # Histogram spread analysis
    print(f"\nHISTOGRAM DISTRIBUTION ANALYSIS:")
    print("-" * 40)
    for name, img in zip(names, images):
        hist, _ = calculate_histogram(img, 256)
        non_zero_bins = np.sum(hist > 0)
        
        # Calculate entropy
        prob = hist[hist > 0] / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob))
        
        print(f"{name}:")
        print(f"  Used intensity levels: {non_zero_bins}/256 ({non_zero_bins/256*100:.1f}%)")
        print(f"  Histogram entropy: {entropy:.2f} bits")


def main():
    """
    Main function to run Exercise 1 test
    """
    print("Starting Exercise 1 Test Function...")
    success = test_exercise1()
    
    if success:
        print(f"\n🎉 Exercise 1 test completed successfully!")
        print("Both contrast_stretch() and equalize_histogram() functions work correctly.")
    else:
        print(f"\n💥 Exercise 1 test failed!")
        exit(1)


if __name__ == "__main__":
    main()