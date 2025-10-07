import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
from contrast_stretch import contrast_stretch
from equalize_histogram import equalize_histogram
from calculate_histogram import calculate_histogram


def analyze_image_contrast(img):
    """
    Analyze the contrast characteristics of an image
    
    Parameters:
    img: grayscale image (numpy array)
    
    Returns:
    dict with contrast metrics
    """
    return {
        'min': img.min(),
        'max': img.max(),
        'mean': img.mean(),
        'std': img.std(),
        'range': img.max() - img.min(),
        'range_percent': (img.max() - img.min()) / 255 * 100
    }


def determine_stretch_parameters(img, percentile_low=1, percentile_high=99):
    """
    Automatically determine optimal contrast stretch parameters
    
    Parameters:
    img: grayscale image
    percentile_low: lower percentile for stretch range (default 1%)
    percentile_high: upper percentile for stretch range (default 99%)
    
    Returns:
    r_min, r_max: stretch range parameters
    """
    r_min = np.percentile(img, percentile_low)
    r_max = np.percentile(img, percentile_high)
    return r_min, r_max


def apply_contrast_enhancements(image_path):
    """
    Apply both contrast enhancement methods to an image
    
    Parameters:
    image_path: path to input image
    
    Returns:
    tuple of (original, stretched, equalized) images and their histograms
    """
    print(f"Loading image: {image_path}")
    
    # Load and validate image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Analyze original image
    original_stats = analyze_image_contrast(img_gray)
    print(f"Original image stats:")
    print(f"  Shape: {img_gray.shape}")
    print(f"  Intensity range: {original_stats['min']}-{original_stats['max']}")
    print(f"  Mean: {original_stats['mean']:.1f}, Std: {original_stats['std']:.1f}")
    print(f"  Dynamic range: {original_stats['range']}/255 ({original_stats['range_percent']:.1f}%)")
    
    # Determine optimal stretch parameters
    r_min, r_max = determine_stretch_parameters(img_gray)
    print(f"Contrast stretch parameters: {r_min:.1f} to {r_max:.1f}")
    
    # Apply contrast stretching
    print("Applying contrast stretching...")
    img_stretched = contrast_stretch(img_gray, r_min, r_max)
    
    # Apply histogram equalization
    print("Applying histogram equalization...")
    img_equalized = equalize_histogram(img_gray)
    
    # Calculate histograms
    print("Calculating histograms...")
    hist_original, _ = calculate_histogram(img_gray, 256)
    hist_stretched, _ = calculate_histogram(img_stretched, 256)
    hist_equalized, _ = calculate_histogram(img_equalized, 256)
    
    return (img_gray, img_stretched, img_equalized,
            hist_original, hist_stretched, hist_equalized)


def create_comparison_visualization(img_gray, img_stretched, img_equalized,
                                  hist_original, hist_stretched, hist_equalized,
                                  output_filename="contrast_enhancement_comparison.png"):
    """
    Create comprehensive comparison visualization
    
    Parameters:
    img_gray, img_stretched, img_equalized: processed images
    hist_original, hist_stretched, hist_equalized: corresponding histograms
    output_filename: name for output file
    
    Returns:
    output_filename: path to saved comparison image
    """
    print("Creating comparison visualization...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Image data and titles
    images = [img_gray, img_stretched, img_equalized]
    image_titles = ['Original (Low Contrast)', 'Contrast Stretched', 'Histogram Equalized']
    
    # Plot images (top row)
    for i, (img, title) in enumerate(zip(images, image_titles)):
        axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(title, fontsize=16, fontweight='bold', pad=20)
        axes[0, i].axis('off')
        
        # Add image statistics
        stats = analyze_image_contrast(img)
        stats_text = (f"Range: {stats['min']}-{stats['max']}\n"
                      f"Mean: {stats['mean']:.1f}\n"
                      f"Std: {stats['std']:.1f}")
        
        axes[0, i].text(0.02, 0.98, stats_text, transform=axes[0, i].transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
                       facecolor='white', alpha=0.9), fontsize=12)
    
    # Plot histograms (bottom row)
    histograms = [hist_original, hist_stretched, hist_equalized]
    hist_titles = ['Original Histogram', 'Stretched Histogram', 'Equalized Histogram']
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    
    for i, (hist, title, color) in enumerate(zip(histograms, hist_titles, colors)):
        # Create bar plot
        x_vals = np.arange(256)
        axes[1, i].bar(x_vals, hist, color=color, alpha=0.8, width=1.0, edgecolor='none')
        
        axes[1, i].set_title(title, fontsize=16, fontweight='bold', pad=20)
        axes[1, i].set_xlabel('Intensity Level', fontsize=12)
        axes[1, i].set_ylabel('Pixel Count', fontsize=12)
        axes[1, i].set_xlim(0, 255)
        axes[1, i].grid(True, alpha=0.3, linewidth=0.5)
        
        # Add histogram statistics
        peak_intensity = np.argmax(hist)
        peak_count = np.max(hist)
        total_pixels = np.sum(hist)
        
        hist_stats = (f"Peak at {peak_intensity} (count: {peak_count:,})\n"
                     f"Total pixels: {total_pixels:,}\n"
                     f"Non-zero bins: {np.sum(hist > 0)}/256")
        
        axes[1, i].text(0.98, 0.98, hist_stats, transform=axes[1, i].transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                       fontsize=10)
    
    # Add overall title and description
    fig.suptitle('Contrast Enhancement Comparison\nOriginal vs Contrast Stretching vs Histogram Equalization', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Add method descriptions
    descriptions = [
        "Limited intensity range\nPoor contrast",
        "Linear mapping to full [0-255]\nPreserves relationships",
        "Uniform intensity distribution\nMaximizes contrast"
    ]
    
    for i, desc in enumerate(descriptions):
        axes[0, i].text(0.5, -0.1, desc, transform=axes[0, i].transAxes,
                       horizontalalignment='center', fontsize=11,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.15)
    
    # Save the comparison
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='png')
    plt.close()
    
    print(f"✓ Comparison saved as: {output_filename}")
    return output_filename


def print_quantitative_analysis(img_gray, img_stretched, img_equalized):
    """
    Print detailed quantitative analysis of enhancement results
    """
    print("\n" + "="*70)
    print("QUANTITATIVE ANALYSIS")
    print("="*70)
    
    # Analyze all images
    images = [img_gray, img_stretched, img_equalized]
    names = ['Original', 'Contrast Stretched', 'Histogram Equalized']
    
    # Basic statistics table
    print(f"{'Method':<20} {'Min':<5} {'Max':<5} {'Mean':<8} {'Std':<8} {'Range':<6} {'Coverage':<10}")
    print("-" * 70)
    
    stats_list = []
    for name, img in zip(names, images):
        stats = analyze_image_contrast(img)
        stats_list.append(stats)
        
        print(f"{name:<20} {stats['min']:<5.0f} {stats['max']:<5.0f} "
              f"{stats['mean']:<8.1f} {stats['std']:<8.1f} {stats['range']:<6.0f} "
              f"{stats['range_percent']:<10.1f}%")
    
    # Improvement analysis
    print(f"\nCONTRAST IMPROVEMENT METRICS:")
    original_std = stats_list[0]['std']
    
    for i, (name, stats) in enumerate(zip(names, stats_list)):
        if i == 0:
            print(f"  {name}: {stats['std']:.2f} (baseline)")
        else:
            improvement = stats['std'] / original_std
            print(f"  {name}: {stats['std']:.2f} ({improvement:.2f}× improvement)")
    
    # Histogram entropy analysis
    print(f"\nHISTOGRAM DISTRIBUTION ANALYSIS:")
    for name, img in zip(names, images):
        hist, _ = calculate_histogram(img, 256)
        
        # Calculate entropy
        prob = hist[hist > 0] / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob))
        
        # Calculate uniformity (inverse of entropy normalized)
        max_entropy = np.log2(256)  # Maximum possible entropy
        uniformity = entropy / max_entropy
        
        non_zero_bins = np.sum(hist > 0)
        
        print(f"  {name}:")
        print(f"    Used intensity levels: {non_zero_bins}/256 ({non_zero_bins/256*100:.1f}%)")
        print(f"    Histogram entropy: {entropy:.2f} bits (max: {max_entropy:.2f})")
        print(f"    Distribution uniformity: {uniformity:.2f} (1.0 = perfectly uniform)")


def main():
    """
    Main function to run the contrast enhancement comparison
    """
    print("="*70)
    print("CONTRAST ENHANCEMENT COMPARISON TOOL")
    print("="*70)
    print("This tool applies contrast stretching and histogram equalization")
    print("to enhance low-contrast images and provides visual comparison.\n")
    
    # Configuration
    input_image = "Images/Low Contrast.jpg"
    output_image = "contrast_enhancement_comparison.png"
    
    # Validate input
    if not os.path.exists(input_image):
        print(f"❌ Error: Input image not found: {input_image}")
        print("Please ensure the image exists in the Images/ directory.")
        return False
    
    try:
        # Process the image
        results = apply_contrast_enhancements(input_image)
        img_gray, img_stretched, img_equalized, hist_original, hist_stretched, hist_equalized = results
        
        # Create visualization
        output_file = create_comparison_visualization(
            img_gray, img_stretched, img_equalized,
            hist_original, hist_stretched, hist_equalized,
            output_image
        )
        
        # Print analysis
        print_quantitative_analysis(img_gray, img_stretched, img_equalized)
        
        # Summary
        print(f"\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("✅ Successfully applied both enhancement methods")
        print("✅ Created comprehensive visual comparison")
        print("✅ Generated quantitative analysis")
        print(f"\n📁 Output file: {output_file}")
        print(f"📊 File size: {os.path.getsize(output_file):,} bytes")
        
        # Key insights
        stretched_improvement = img_stretched.std() / img_gray.std()
        equalized_improvement = img_equalized.std() / img_gray.std()
        
        print(f"\n🔍 KEY INSIGHTS:")
        print(f"   • Contrast stretching: {stretched_improvement:.1f}× contrast improvement")
        print(f"   • Histogram equalization: {equalized_improvement:.1f}× contrast improvement")
        print(f"   • {'Histogram equalization' if equalized_improvement > stretched_improvement else 'Contrast stretching'} provided better enhancement")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 Contrast enhancement comparison completed successfully!")
    else:
        print(f"\n💥 Contrast enhancement comparison failed!")
        exit(1)