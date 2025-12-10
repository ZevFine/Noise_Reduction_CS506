# Section 1 : Introduction





---

# Section 2 : Pre-Noise Metrics

## 2.1 Approach and Methodology

All original color and grayscale images were evaluated using the full set of image quality metrics to establish a baseline reference. These pre-noise measurements capture the natural sharpness, spatial frequency, contrast range, entropy, and structural characteristics of the clean images without any distortion. This baseline serves as the ground truth, allowing for an objective assessment of how much degradation is introduced by noise and how effectively each denoising technique restores the original image quality.

---

## 2.2 Quantitative Metrics Analysis

### Color Images

**Brightness**  
The mean is further to the right here, which would imply that overall the photos are pretty dark, but there are some brighter outliers.

**Contrast**  
The contrast has a lot of variance, there are sudden spikes, these are because the data sets where the photo has a black background.

**Entropy**  
Overall the photos are not that complex. This most likely stems from the pokemon and pistachio data sets, because they have the black backgrounds. The huge spike at the end is the rest of the images, where the whole image has something going on.

**Sharpness**  
There is a lot of low sharpness, this is because the images we used have pretty low resolution.

**Spatial Frequency**  
Like Sharpness this implies that there are a lot of blurry/low resolution images.

**Dynamic Range**  
Since this is scaled to 1, the contrast between the darkest and lightest pixels are high.

---

### Black and White Images

**Brightness**  
This has more or less the same distribution as color.

**Contrast**  
The contrast is more right skewed than the color, this is because of the lack of color.

**Entropy**  
This is more or less the same distribution as color.

**Sharpness**  
This is more or less the same distribution as color.

**Spatial Frequency**  
This is more or less the same distribution as color.

**Dynamic Range**  
This is more or less the same distribution as color, it has more variance because when the photo went grayscale it doesn't allow as big of a difference between the two extremes.

---

# Section 3 : Salt & Pepper Noise Generation

## 3.1 Approach and Methodology

Salt and pepper noise, also known as impulse noise, is a form of noise characterized by random occurrences of white (salt) and black (pepper) pixels scattered throughout the image. Our implementation added this noise type to evaluate its impact on image quality and to establish a baseline for subsequent denoising algorithms. We systematically tested five different noise intensity levels to understand the relationship between noise amount and image degradation.

### Parameters Table

| Parameters     | Amount | Description     |
|---------------|--------|-----------------|
| amount_0.01   | 1%     | Minimal noise   |
| amount_0.05   | 5%     | Light noise     |
| amount_0.10   | 10%    | Moderate noise  |
| amount_0.15   | 15%    | Heavy noise     |
| amount_0.20   | 20%    | Severe noise    |

---

## 3.2 Quantitative Metrics Analysis

**Noised Image: amount_0.20**

### Histogram: amount_0.20

**PSNR Distribution:** Mean = 7.85 dB, Median = 7.44 dB  
Dramatically reduced from low noise condition  
Degradation: ~11.5 dB decrease (60% reduction)  
Very tight distribution indicates consistent severe degradation  
Values below 10 dB indicate poor signal quality  

**SSIM Distribution:** Mean = 0.06, Median = 0.06  
Severe structural degradation  
Degradation: 0.51 decrease (89% reduction)  
Nearly uniform distribution showing complete loss of structural similarity  
Values approaching 0 indicate minimal resemblance to original  

**MSE Distribution:** Mean = 10,786.65, Median = 11,716.68  
Increase: ~14x higher than low noise  
Bimodal distribution suggests varying error patterns  
Extremely high error values confirm substantial pixel-level differences  

**Entropy Difference:** Mean = 1.02, Median = 0.61  
Significant information content change  
Wide distribution (0 to 2.0) shows variable impact  
Higher entropy differences indicate loss of original information structure  

**Noise Variance:** Mean = 10,090.28, Median = 10,276.23  
Increase: ~13x higher than low noise  
Broader distribution indicates less predictable noise impact  
High variance confirms substantial signal corruption  

**Sharpness:** Mean = 184,070.10, Median = 193,454.91  
Increase: ~10x higher due to noise-induced high-frequency content  
Paradoxically high values reflect noise artifacts rather than true detail  
Distribution shows noise dominates edge detection  

**Spatial Frequency:** Mean = 191.87, Median = 197.32  
Increase: ~3x higher than low noise  
High spatial frequency reflects noise patterns  
Original spatial information masked by noise  

**Dynamic Range:** Preserved at 255.00  
Unchanged regardless of noise level  
Indicates salt and pepper noise utilizes full intensity range  

---

## 3.3 Denoising Goals

For Severe Noise (amount_0.20):

**PSNR:** Achieve 15–20 dB (recovery from 7.85 dB)  
Goal: 2–3x improvement  
Rationale: Above 15 dB makes content recognizable  

**SSIM:** Achieve 0.40–0.60 (recovery from 0.06)  
Goal: Restore basic structural information  

**MSE:** Reduce to 3,000–5,000 (from 10,786.65)  
Goal: 50–70% reduction  

**Entropy Difference:** Reduce to 0.20–0.40 (from 1.02)  
Goal: Restore information structure and balance denoising with detail preservation  

---

# Section 4: Salt & Pepper Denoising – K-Medoids Clustering Technique

## 4.1 Denoising Methodology

### 4.1.1 Algorithm Selection and Rationale

K-medoids was selected for its robustness to outliers, suitability for discrete clustering of salt and pepper pixels, and preservation of valid pixel intensities.

### 4.1.2 Implementation

A two-stage hybrid approach was used:

**Stage 1: Adaptive Impulse Detection and Median Filtering**  
Pixels exceeding intensity thresholds were selectively median filtered to preserve edges while removing extreme impulses.

**Stage 2: K-medoids Refinement**  
The pre-filtered image was clustered using K-medoids with multiple cluster counts to optimize noise removal and detail retention.

---

### 4.1.3 Parameter Configurations

| Configuration | n_clusters | use_median_prefilter |
|---------------|------------|----------------------|
| kmedoids_k3_no_prefilter | 3 | False |
| kmedoids_k5_no_prefilter | 5 | False |
| kmedoids_k8_no_prefilter | 8 | False |
| kmedoids_k3_median | 3 | True |
| kmedoids_k5_median | 5 | True |
| kmedoids_k8_median | 8 | True |
| kmedoids_k10_median | 10 | True |
| kmedoids_k12_median | 12 | True |

---

## 4.2 Results Analysis

### 4.2.1 Visual Comparison of All Parameters

**Without Median Pre-filtering**  
Pure clustering fails catastrophically with PSNR ≈ 8 dB.

**With Median Pre-filtering**  
kmedoids_k8_median produced the best visual and quantitative balance (PSNR ≈ 22.94 dB).

---

### 4.2.2 Heat Map Analysis – Metric Comparison

Normalized heat maps reveal that prefiltered methods dominate across PSNR, SSIM, and Sharpness, while entropy and dynamic range reflect expected trade-offs.

---

## 4.4 Method Limitations and Trade-offs

- Entropy distortion due to clustering  
- Dynamic range compression  
- Detail loss in smooth regions  
- Catastrophic failure without median prefiltering  

---

# Section 5: Speckle Noise Generation

## 5.1 Approach and Methodology

Speckle noise is a multiplicative noise that appears as granular patterns, commonly found in ultrasound imaging, SAR imagery, and laser-based systems. Unlike additive noise types, speckle is signal-dependent and varies with local image intensity. We systematically tested six different noise intensity levels to understand the relationship between noise amount and image degradation.

### Parameters Table

| Parameters      | Amount | Description        |
|-----------------|--------|--------------------|
| intensity_0.5   | 0.5    | Minimal speckle    |
| intensity_1.0   | 1.0    | Light speckle      |
| intensity_1.5   | 1.5    | Moderate speckle   |
| intensity_2.0   | 2.0    | Heavy speckle      |
| intensity_2.5   | 2.5    | Severe speckle     |
| intensity_3.0   | 3.0    | Extreme speckle    |

---

## 5.2 Quantitative Metrics Analysis

**Noised Image: intensity_3.0**

### Noised Image Metric Histogram: intensity_3.0

**PSNR Distribution:** Mean = 12.74 dB, Median = 12.93 dB  
Degradation: 6.3 dB decrease from low noise (33% reduction)  
Normal distribution centered tightly around mean  
Values above 10 dB indicate some content still recoverable  

**SSIM Distribution:** Mean = 0.45, Median = 0.48  
Degradation: 0.18 decrease (29% reduction)  
Bimodal distribution with second peak around 0.80  
Suggests some images maintain reasonable structural similarity  
Multiplicative nature preserves structure better than impulse noise  

**MSE Distribution:** Mean = 4,458.96, Median = 3,326.73  
Increase: 4.3x from low noise  
Right-skewed distribution  
Reflects controlled degradation compared to impulse noise  

**Entropy Difference:** Mean = 0.34, Median = 0.26  
Minimal change from low noise intensity  
Tight distribution near zero  
Speckle maintains information structure better  

**Noise Variance:** Mean = 4,062.41, Median = 2,952.19  
Increase: 4.2x from low noise  
Distribution shows controlled variance increase  
Multiplicative model produces more predictable variance  

**Sharpness:** Mean = 63,235.21, Median = 41,066.96  
Increase: 3.2x from low noise  
Reflects smoother noise pattern of speckle vs. sharp impulses  
Right-skewed distribution indicates variable impact  

**Spatial Frequency:** Mean = 106.86, Median = 91.37  
Increase: 1.7x from low noise  
Broader distribution (40–160 range)  
Speckle adds frequency content more gradually  

**Dynamic Range:** Preserved at 255.00  
Unchanged across all intensity levels  
Clipping maintains full range utilization  

---

## 5.3 Denoising Goals

For High Noise (intensity_3.0):

**PSNR:** Achieve 18–22 dB (recovery from 12.74 dB)  
Goal: 5–9 dB improvement  

**SSIM:** Achieve 0.65–0.80 (recovery from 0.45)  
Goal: Restore to near-moderate quality  

**MSE:** Reduce to 1,000–2,000 (from 4,458.96)  
Goal: 55–78% reduction  
More optimistic than salt & pepper due to better starting metrics  

**Entropy Difference:** Maintain below 0.40 (from 0.34)  
Goal: Preserve already-good information structure  
Avoid introducing artifacts  

---

# Section 6: Speckle Denoising – Log-Domain Non-Local Means

## 6.1 Denoising Methodology

### 6.1.1 Algorithm Selection and Rationale

For speckle noise removal, Log-Domain Non-Local Means (NLM) filtering was selected based on the following properties:

- Logarithmic transformation converts multiplicative noise into an additive form that standard denoising filters can handle.
- Self-similarity exploitation allows NLM to search for repeating structures across the image.
- Patch-based filtering reduces noise using structurally similar neighborhoods rather than local averaging.
- Adaptive weighting ensures that similar patches contribute more strongly to the final estimate.
- Superior edge preservation maintains sharp boundaries and reduces edge blurring.

---

### 6.1.2 Implementation

**Stage 1: Logarithmic Preprocessing**  
The input image is converted to floating-point format and normalized to the [0, 1] range. A logarithmic transform is applied to stabilize multiplicative noise into an additive form and enhance low-intensity variations. The result is normalized back to 8-bit format for compatibility with the denoising filter.

**Stage 2: Non-Local Means Filtering**  
Denoising is performed in the log domain using Non-Local Means filtering. This method averages pixels based on patch similarity across the image rather than relying solely on local neighborhoods. Filter strength, patch size, and search window size control the balance between noise removal and detail preservation. For color images, filtering is applied only to the luminance channel in YCrCb space to preserve chromatic information.

**Stage 3: Inverse Logarithmic Reconstruction**  
After filtering, the image is mapped back to the original intensity domain using an exponential transform. The output is clipped to the valid 8-bit range for visualization and evaluation.

---

### 6.1.3 Parameter Configurations

| Configuration       | h   | template | search |
|---------------------|-----|----------|--------|
| nlm_h14_t7_s21      | 14  | 7        | 21     |
| nlm_h17_t7_s21      | 17  | 7        | 21     |
| nlm_h20_t7_s21      | 20  | 7        | 21     |
| nlm_h23_t7_s21      | 23  | 7        | 21     |
| nlm_h20_t9_s31      | 20  | 9        | 31     |
| nlm_h23_t9_s31      | 23  | 9        | 31     |

---

## 6.2 Results Analysis

### 6.2.1 Visual Comparison of All Parameters

**Low Filter Strength**  
nlm_h14_t7_s21 (PSNR: 13.11 dB): Minimal improvement, significant residual speckle visible  
nlm_h17_t7_s21 (PSNR: 13.70 dB): Slight improvement but still heavily corrupted  
Analysis: Insufficient filter strength for intensity_3.0 speckle  

**Over-Smoothing Region**  
nlm_h23_t7_s21 (PSNR: 14.62 dB): Higher PSNR but visible loss of texture  
nlm_h20_t9_s31 (PSNR: 14.45 dB): Larger patches cause excessive smoothing  
nlm_h23_t9_s31 (PSNR: 14.54 dB): Over-smoothed, plastic appearance  
Analysis: Excess smoothing sacrifices natural texture  

---

### 6.2.2 Heat Map Analysis – Metric Comparison

**PSNR Column:** Progressive greening from h14 → h23  
**SSIM Column:** Wide variation; higher h reduces structural fidelity  
**MSE Column:** Consistently red across all parameters  
**Entropy_Diff:** Strong green across all methods  
**Sharpness:** Peaks around h20, drops for h23  
**Dynamic_Range:** Consistently preserved  

**Best Parameter Performance Profile**

Strengths (green): PSNR (0.94), SSIM (0.93), Sharpness (high), Noise_Variance (low)  
Moderate (yellow): Entropy_Diff (0.42), Dynamic_Range (0.75)  
Trade-offs visible: Slight entropy increase for superior noise removal  

---

## 6.3 Other Examples

(See additional visual best-result comparisons for intensity_1.0 and intensity_2.0.)

---

## 6.4 Method Limitations and Trade-offs

1. **Multiplicative Noise Complexity**
   - **Limitation:** Multiplicative coupling between signal and noise makes separation difficult.  
   - **Evidence:** PSNR improvement (~1.64 dB) is far less than for salt & pepper (≈15.09 dB).  
   - **Conclusion:** Log-domain NLM mitigates but does not fully resolve the multiplicative nature of speckle.

2. **Dense, Correlated Noise Pattern**
   - **Limitation:** At intensity_3.0, speckle introduces dense granular texture across the entire image.  
   - **Evidence:** Visual comparison shows residual granularity throughout the scene.  
   - **Conclusion:** Patch-based methods struggle when essentially every pixel is affected by structured noise.

3. **Texture vs. Noise Ambiguity**
   - **Limitation:** Speckle patterns can resemble genuine surface texture.  
   - **Evidence:** Reduced sharpness (e.g., 13,251) indicates some real texture is removed along with noise.  
   - **Conclusion:** There is no perfect solution; improving noise removal inevitably risks texture loss.

4. **Parameter Sensitivity and Diminishing Returns**
   - **Limitation:** Beyond \(h = 20\), stronger smoothing yields marginal PSNR gains but worse perceptual quality.  
   - **Evidence:** h23 gives only ~0.24 dB higher PSNR but noticeably lower sharpness (10,527 vs. 13,251).  
   - **Conclusion:** Log-domain NLM approaches a performance ceiling at extreme speckle levels.


