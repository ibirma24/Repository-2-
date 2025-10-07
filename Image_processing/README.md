Exercise 1:
contrast_stretch.py
Enhances image contrast by linearly mapping a specified intensity range to the full dynamic range [0, 255].
Implementation
Takes an input image and target range [r_min, r_max] Applies linear transformation: `new_pixel = 255 * (pixel - r_min) / (r_max - r_min)`  Clips values to ensure they remain within [0, 255] Useful for images where most pixel values are concentrated in a narrow range

calculate_histogram.py
Computes both frequency counts and probability distribution of pixel intensities. ivides the intensity range [0, 255] into specified number of bins counts occurrences of pixels in each bin (histogram) normalizes counts to create probability distribution Essential for understanding image contrast and brightness characteristics

equalize_histogram.py
Purpose is to redistributes pixel intensities to create a more uniform histogram, enhancing overall contrast.

Implementation:
Calculates cumulative distribution function (CDF) of the image histogram maps original intensities to new values using the CDF
particularly effective for images with poor contrast distribution

Analysis: Compare how contrast stretching and histogram equalization differently affect low-contrast images by visualizing their histograms side-by-side.

Exercise 2

median_filter.py
Purpose it to reduces salt-and-pepper noise while preserving edges better than linear filters.

Implementation:
Slides a window of specified size (default 3×3) over the image For each window: extracts neighborhood pixels, sorts values, replaces center pixel with median non-linear operation that effectively removes impulse noise without blurring edges significantly

calculate_gradient.py
Purpose is to computes gradient magnitude using Sobel operators to detect intensity changes.

Implementation
Applies horizontal (S_x) and vertical (S_y) Sobel filters using convolution computes gradient magnitude: `√(S_x² + S_y²)dentifies regions of rapid intensity change, which typically correspond to edges

Analysis: Investigates how median filtering affects gradient calculations on noisy images, demonstrating the importance of preprocessing for reliable edge detection.

Exercise 3: Sobel-based Edge Detection

sobel_edge_detector.py
Purpose is to creates clean binary edge maps using gradient magnitude thresholding.

Implementation**:
Extends `calculate_gradient()` to also compute gradient direction: `θ = arctan(S_y / S_x)` Applies binary threshold to gradient magnitude (pixels > threshold = 255, else = 0) Produces a clear edge map highlighting significant intensity transitions

directional_edge_detector.py
Purpose is to detects edges oriented in specific directions.

Implementation**:
Uses gradient direction information from calculate_gradient()applies directional thresholding based on specified angle range (e.g., 40°-50°) isolates edges with particular orientations, useful for detecting specific features

Analysis: Compares simple Sobel-based edge detection with directional filtering and OpenCV's Canny edge detector, evaluating their strengths and limitations in different scenarios.

Usage

All functions are designed to work with grayscale images. Load your images using OpenCV or PIL and convert to grayscale if necessary before processing.

```python
# Example usage
import cv2
from sobel_edge_detector import sobel_edge_detector

img = cv2.imread('images/input.jpg', 0)  # Read as grayscale
edges = sobel_edge_detector(img, threshold=50)

