# Poisson Noise and Anscombe Transform Denoising

A comprehensive image denoising pipeline that simulates Poisson noise (photon counting noise) and removes it using the Anscombe Transform method. This notebook systematically evaluates different denoising filters across multiple noise levels to determine optimal denoising strategies.

## Table of Contents
- [Overview](#overview)
- [What Is Poisson Noise?](#what-is-poisson-noise)
- [The Anscombe Transform Method](#the-anscombe-transform-method)
- [How It Works](#how-it-works)
- [Understanding the Metrics](#understanding-the-metrics)
- [Denoising Filters and Kernel Sizes](#denoising-filters-and-kernel-sizes)
- [Output Structure](#output-structure)
- [How to Use](#how-to-use)
- [Interpreting Results](#interpreting-results)
- [Technical Details](#technical-details)

---

## Overview

This project provides a complete framework for:
1. **Simulating Poisson noise** at 5 different intensity levels
2. **Applying Anscombe Transform denoising** using 6 different filter configurations
3. **Calculating 8 quality metrics** to measure denoising effectiveness
4. **Generating visualizations** including histograms and side-by-side comparisons
5. **Exporting detailed CSV reports** for further analysis

### Applications
- **Medical Imaging**: X-ray, CT scans, PET scans
- **Astronomy**: Telescope and satellite imagery
- **Low-Light Photography**: Night vision, surveillance
- **Scientific Imaging**: Microscopy, spectroscopy
- **Any photon-limited imaging** where counting statistics matter

---

## What Is Poisson Noise?

**Poisson noise** (also called shot noise or photon noise) is the dominant noise source in photon-counting applications. It arises from the quantum nature of light - photons arrive randomly, following a Poisson distribution.

### Key Characteristics:
- **Signal-dependent**: More signal = more noise (variance equals the mean)
- **Unavoidable**: Fundamental physical limit in photon detection
- **Scale parameter**: Controls noise intensity
  - **Lower scale** (e.g., 0.1) = fewer photons = more noise
  - **Higher scale** (e.g., 5.0) = more photons = less noise

### Noise Levels Tested:
| Scale | Description | Photon Count | Typical Application |
|-------|-------------|--------------|---------------------|
| 0.1 | Very high noise | Very low | Extremely low-light conditions |
| 0.5 | High noise | Low | Low-light imaging |
| 1.0 | Standard Poisson | Normal | Typical photon-limited imaging |
| 2.0 | Moderate noise | Good | Well-illuminated scenes |
| 5.0 | Low noise | High | Bright imaging conditions |

---

## The Anscombe Transform Method

The Anscombe Transform is a variance-stabilizing transformation that converts Poisson-distributed noise into approximately Gaussian noise with constant variance. This makes standard denoising filters much more effective.

### Why Use Anscombe Transform?

**Problem**: Poisson noise has signal-dependent variance, making it difficult to denoise with standard filters designed for Gaussian noise.

**Solution**: Transform the data so the noise becomes approximately Gaussian with stable variance, apply standard denoising, then transform back.

### The Three-Step Process:

1. **Forward Transform**: `y = 2√(x + 3/8)`
   - Converts Poisson noise → approximately Gaussian noise
   - Stabilizes variance across the image

2. **Denoising in Transformed Domain**
   - Apply standard denoising filter (Gaussian, Median, or Bilateral)
   - Filters work optimally on Gaussian noise

3. **Inverse Transform**: `x = (y/2)² - 3/8`
   - Converts back to original domain
   - Recovers the denoised image

This approach is mathematically rigorous and widely used in scientific imaging applications.

---

## How It Works

### Processing Pipeline:

```
Original Image
    ↓
Add Poisson Noise (5 different scales)
    ↓
Noisy Image ──────────────→ Save to disk + Calculate metrics vs Original
    ↓
Apply Anscombe Transform
    ↓
Transformed Image (Gaussian-like noise)
    ↓
Apply Denoising Filter (6 configurations)
    ↓
Filtered Image
    ↓
Apply Inverse Transform
    ↓
Denoised Image ───────────→ Save metrics + Calculate improvement
```

### What Gets Tested:
- **5 noise levels** × **6 filter configurations** = **30 total combinations** per image
- Each combination is evaluated with 8 quality metrics
- Results are organized by noise level and filter type

---

## Understanding the Metrics

All metrics compare the **original clean image** to either the **noisy** or **denoised** version. This allows objective evaluation of how well the denoising recovered the original image.

### Primary Quality Metrics:

#### 1. **PSNR (Peak Signal-to-Noise Ratio)**
- **Range**: 0 to ∞ (typically 20-50 dB)
- **Higher is better**: More signal, less noise
- **Interpretation**:
  - < 20 dB: Poor quality
  - 20-30 dB: Acceptable quality
  - 30-40 dB: Good quality
  - \> 40 dB: Excellent quality
- **Formula**: `10 × log₁₀(255² / MSE)`
- **What it measures**: Pixel-level accuracy compared to original

#### 2. **SSIM (Structural Similarity Index)**
- **Range**: -1 to 1 (typically 0 to 1)
- **Higher is better**: Closer to 1 = more similar to original
- **Interpretation**:
  - < 0.5: Poor structural similarity
  - 0.5-0.8: Moderate similarity
  - 0.8-0.95: Good similarity
  - \> 0.95: Excellent similarity
- **What it measures**: Perceived quality (considers luminance, contrast, structure)
- **Advantage**: Better correlates with human perception than PSNR

#### 3. **MSE (Mean Squared Error)**
- **Range**: 0 to ∞
- **Lower is better**: Less error between denoised and original
- **Formula**: `mean((original - denoised)²)`
- **What it measures**: Average squared pixel difference
- **Note**: PSNR is derived from MSE

#### 4. **Entropy Difference**
- **Range**: 0 to ∞
- **Lower is better**: Similar information content to original
- **What it measures**: How much information was lost or added
- **Interpretation**: Lower values mean better preservation of image information

### Additional Metrics:

#### 5. **Noise Variance**
- Variance of the difference between original and processed image
- Lower values indicate less remaining noise

#### 6. **Sharpness (Laplacian Variance)**
- Measures edge clarity using Laplacian operator
- Higher values indicate sharper edges
- Useful for detecting over-smoothing

#### 7. **Spatial Frequency**
- Measures image detail and texture
- Higher values indicate more fine detail preserved

#### 8. **Dynamic Range**
- Difference between maximum and minimum intensity
- Indicates contrast preservation

### Improvement Metrics:

The notebook also calculates how much the denoising helped:

- **PSNR_Improvement**: `Denoised_PSNR - Noisy_PSNR`
  - Positive = denoising improved quality
  - Negative = denoising made it worse

- **SSIM_Improvement**: `Denoised_SSIM - Noisy_SSIM`
  - Shows structural quality improvement

- **MSE_Reduction**: `Noisy_MSE - Denoised_MSE`
  - Positive = less error after denoising
  - Negative = more error (denoising failed)

### What Comparisons Are Being Made:

**For Noisy Metrics**: Original vs Noisy
- Shows how much the noise degraded the image

**For Denoised Metrics**: Original vs Denoised
- Shows how close the denoising got back to the original
- **This is what the histograms visualize**

**For Improvement Metrics**: Denoised vs Noisy
- Shows the effectiveness of the denoising process

---

## Denoising Filters and Kernel Sizes

The notebook tests three types of filters, each with different characteristics and kernel sizes.

### Filter Types:

#### 1. **Gaussian Blur**
- **How it works**: Weighted average of pixels in a neighborhood
- **Weights**: Follow a Gaussian (bell curve) distribution
- **Characteristics**:
  - Smooth, natural-looking results
  - Good for general noise reduction
  - Blurs edges uniformly
- **Best for**: General-purpose denoising, smooth regions

#### 2. **Median Filter**
- **How it works**: Replaces each pixel with the median value in the neighborhood
- **Characteristics**:
  - Excellent at removing salt-and-pepper noise
  - Preserves edges better than Gaussian
  - Non-linear filter
- **Best for**: Impulsive noise, edge preservation

#### 3. **Bilateral Filter**
- **How it works**: Weighted average considering both spatial distance AND intensity similarity
- **Characteristics**:
  - Edge-aware: doesn't blur across edges
  - Smooths flat regions while preserving edges
  - More computationally expensive
- **Best for**: High-quality denoising with edge preservation

### Understanding Kernel Size:

The **kernel size** defines the neighborhood of pixels considered by the filter. It controls the trade-off between noise removal and detail preservation.

#### Small Kernel (3×3):
- **Pixels considered**: 9 neighbors
- **Noise removal**: Moderate
- **Detail preservation**: Excellent
- **Edge sharpness**: Sharp
- **Processing speed**: Fast
- **Best for**: Low noise levels, preserving fine details

#### Medium Kernel (5×5):
- **Pixels considered**: 25 neighbors
- **Noise removal**: Good
- **Detail preservation**: Good
- **Edge sharpness**: Moderate
- **Processing speed**: Moderate
- **Best for**: Balanced noise reduction

#### Large Kernel (7×7):
- **Pixels considered**: 49 neighbors
- **Noise removal**: Strong
- **Detail preservation**: Poor
- **Edge sharpness**: Blurry
- **Processing speed**: Slow
- **Best for**: High noise levels, smooth regions

### Filter-Specific Behaviors:

**Gaussian Blur**:
- Larger kernel = more averaging = stronger blur
- Weights decrease smoothly with distance
- All pixels in kernel contribute (weighted)

**Median Filter**:
- Larger kernel = considers more pixels for median
- Particularly effective at removing outliers
- Only the median value matters (not all pixels)

**Bilateral Filter**:
- Kernel size sets spatial neighborhood
- Also uses intensity similarity threshold
- Only averages pixels that are both spatially close AND similar in intensity
- Result: edges stay sharp, flat regions get smoothed

### Tested Configurations:

| Filter Type | Kernel Sizes | Total Configs |
|-------------|--------------|---------------|
| Gaussian | 3, 5, 7 | 3 |
| Median | 3, 5 | 2 |
| Bilateral | 5 | 1 |
| **Total** | | **6** |

### General Guidelines:

- **High noise** (scale 0.1, 0.5): Larger kernels may be necessary
- **Low noise** (scale 2.0, 5.0): Smaller kernels preserve more detail
- **Edge-heavy images**: Median or Bilateral filters work better
- **Smooth regions**: Gaussian blur is efficient and effective
- **Quality priority**: Bilateral filter (but slower)
- **Speed priority**: Gaussian blur with small kernel

---

## Output Structure

All outputs are organized in a clean, hierarchical directory structure:

```
data_results/data/
│
├── Poisson_Anscombe_Metric/          # CSV files with all metrics
│   ├── noisy_scale_0.1.csv           # Noisy image metrics (color)
│   ├── noisy_scale_0.1_bw.csv        # Noisy image metrics (B&W)
│   ├── denoised_scale_0.1_gaussian_k3.csv
│   ├── denoised_scale_0.1_gaussian_k3_bw.csv
│   └── ... (70 CSV files total)
│
├── Poisson_Noised_Images/            # Images with noise added
│   ├── scale_0.1/
│   │   ├── noisy_image1.jpg
│   │   └── ...
│   ├── scale_0.5/
│   ├── scale_1.0/
│   ├── scale_2.0/
│   └── scale_5.0/
│
├── Poisson_Anscombe_Sample/          # Visual comparisons
│   ├── comparison_image1_scale_0.1.png
│   ├── comparison_image1_scale_0.5.png
│   └── ... (5 samples × 5 noise levels = 25 comparison images)
│
└── graph/poisson/                    # Histogram visualizations
    ├── histogram_denoised_psnr_detailed.png
    ├── histogram_denoised_ssim_detailed.png
    ├── histogram_denoised_mse_detailed.png
    └── histogram_denoised_entropy_diff_detailed.png
```

### File Descriptions:

#### CSV Files:
- **Separate files** for color and B&W images
- **Noisy metrics**: 10 files (5 noise levels × 2 image types)
- **Denoised metrics**: 60 files (5 noise levels × 6 filters × 2 image types)
- **Columns include**: Image name, noise level, scale, filter type, all 8 metrics, improvement metrics

#### Noisy Images:
- Organized by noise scale
- Named with `noisy_` prefix
- Can be used for visual inspection or further processing

#### Sample Comparisons:
- Grid layout showing: Original, Noisy, and 6 Denoised versions
- 2 rows × 4 columns format
- High-resolution PNG files (150 DPI)
- Useful for visual quality assessment

#### Histograms:
- Grid layout: Rows = Noise levels, Columns = Filter types
- Multiple kernel sizes shown as different colors within each subplot
- Shows distribution of metric values across all images
- Helps identify optimal filter-noise combinations

---

## How to Use

### Prerequisites:
```bash
pip install opencv-python numpy scikit-image scipy pandas matplotlib seaborn tqdm
```

### Basic Usage:

1. **Configure Input Path** (Cell 2):
   ```python
   base_path = "./Photos_Subset/Original"  # Update this to your images folder
   ```

2. **Run All Cells** in order (cells must execute sequentially)

3. **Wait for Processing**:
   - Progress bars will show processing status
   - Time depends on number of images and their resolution
   - Typical: 1-5 minutes for 50 images

4. **Check Outputs**:
   - All results automatically saved to `data_results/data/`
   - No manual file saving required

### Advanced Customization:

#### Modify Noise Levels (Cell 6):
```python
POISSON_PARAMS = [
    {"scale": 0.1, "name": "scale_0.1"},
    {"scale": 0.5, "name": "scale_0.5"},
    # Add or remove noise levels as needed
]
```

#### Modify Filter Configurations (Cell 6):
```python
ANSCOMBE_FILTERS = [
    {"filter_type": "gaussian", "kernel_size": 3, "name": "gaussian_k3"},
    # Add custom filter configurations
]
```

#### Change Output Location (Cell 2):
```python
output_base = "custom_output_path"
```

### Input Requirements:

- **Supported formats**: PNG, JPG, JPEG, BMP, TIFF
- **Color or grayscale**: Both handled automatically
- **Naming convention**: 
  - Files ending with `_bw.jpg` or `_bw.png` are treated as black & white
  - All others treated as color
- **Location**: All images in a single folder

---

## Interpreting Results

### Reading the Histograms:

The histogram visualizations show the distribution of metric values organized by:
- **Rows**: Different noise scales (0.1 to 5.0)
- **Columns**: Different filter types (Gaussian, Median, Bilateral)
- **Colors within each subplot**: Different kernel sizes

#### What to Look For:

1. **Higher PSNR/SSIM** = Better denoising quality
2. **Lower MSE/Entropy_Diff** = Better preservation of original
3. **Tight distributions** = Consistent performance across images
4. **Wide distributions** = Performance varies by image content

#### Example Analysis:

**High Noise (scale 0.1)**:
- Expect lower PSNR/SSIM overall (noise is severe)
- Look for which filter gives the highest peak in PSNR histogram
- Larger kernels may perform better

**Low Noise (scale 5.0)**:
- Expect higher PSNR/SSIM overall (noise is mild)
- Smaller kernels may preserve more detail
- All filters should perform reasonably well

### Reading the CSV Files:

Each CSV contains detailed metrics for every processed image:

```csv
Name,Noise_Level,Scale,Filter,Denoised_PSNR,Denoised_SSIM,...,PSNR_Improvement,...
denoised_img1.jpg,scale_0.1,0.1,gaussian_k3,24.5,0.78,...,+5.2,...
```

#### Key Columns:
- **Name**: Image filename
- **Noise_Level**: Which noise configuration was applied
- **Scale**: Numeric noise scale value
- **Filter**: Which denoising filter was used
- **Denoised_[Metric]**: Quality of denoised image vs original
- **[Metric]_Improvement**: How much denoising helped

#### Useful Analyses:

1. **Find best filter for each noise level**:
   ```python
   df.groupby(['Noise_Level', 'Filter'])['Denoised_PSNR'].mean()
   ```

2. **Compare kernel sizes**:
   ```python
   df[df['Filter'].str.contains('gaussian')].groupby('Filter')['SSIM_Improvement'].mean()
   ```

3. **Identify problematic images**:
   ```python
   df[df['PSNR_Improvement'] < 0]  # Denoising made it worse
   ```

### Reading Sample Comparisons:

The sample comparison images show:
- **Top-left**: Original clean image (ground truth)
- **Top-middle**: Noisy version
- **Remaining 6 positions**: Different denoised versions

**Visual Inspection Checklist**:
- Are edges preserved or over-smoothed?
- Is texture detail maintained?
- Are there any artifacts introduced?
- Which filter looks closest to the original?
- Which filter removed the most noise?

---

## Technical Details

### Computational Complexity:

**For N images, P noise levels, F filters**:
- **Noise addition**: O(N × P)
- **Denoising**: O(N × P × F)
- **Metric calculation**: O(N × P × F)
- **Total processing**: O(N × P × F)

**Default configuration**: N images × 5 noise levels × 6 filters = 30N processing operations

### Memory Requirements:

- Images loaded one at a time (memory-efficient)
- Sample images stored in memory for visualization (5 images max)
- Peak memory usage: ~5-10x single image size

### Performance Tips:

1. **Reduce noise levels or filters** for faster processing
2. **Downscale images** if resolution is very high
3. **Process in batches** if you have many images
4. **Use SSD** for faster I/O operations

### Algorithm Details:

**Anscombe Transform**:
- Forward: `y = 2√(x + 3/8)`
- Inverse: `x = (y/2)² - 3/8`
- Valid for Poisson parameter λ > 0
- Approximation improves for larger λ

**Noise Generation**:
- Uses `numpy.random.poisson()` for accurate Poisson distribution
- Scaling controls effective photon count
- Reproducible with fixed random seed

### Limitations:

1. **Assumes ground truth available**: Requires clean original images
2. **Synthetic noise**: Real-world noise may have additional components
3. **Grayscale metrics**: Color images converted to grayscale for PSNR/SSIM
4. **Anscombe approximation**: Works best for moderate to high photon counts

---
