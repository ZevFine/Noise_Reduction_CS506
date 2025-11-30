#!/usr/bin/env python3
"""
COMPREHENSIVE DENOISING COMPARISON SCRIPT
==========================================================
Tests 4 noise types × 4 denoising methods = 16 combinations

HYPOTHESIS TESTING:
i.   Gaussian Noise → Frequency Domain (FFT) is best
ii.  Poisson Noise → Anscombe Transform is best  
iii. Salt & Pepper Noise → Clustering/Median is best
iv.  Speckle Noise → Log Transform + Gaussian is best

How to use:
    python denoise.py                           # runs on built-in sample 'camera'
    python denoise.py path/to/your_image.jpg   # process single image
    python denoise.py path/to/image/folder/    # process all images in folder

Dependencies:
    numpy, matplotlib, scikit-image, scipy, pandas, pywavelets
    pip install numpy matplotlib scikit-image scipy pandas PyWavelets


"""
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# Try imports
try:
    from skimage import img_as_float, color, io, util, exposure, data
    from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
    from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle
    from skimage.util import random_noise, img_as_ubyte
    from skimage.filters import median, gaussian
    from skimage.morphology import disk
    from skimage.exposure import rescale_intensity
except Exception as e:
    raise RuntimeError(f"scikit-image required: pip install scikit-image\n{e}")

try:
    import scipy.ndimage as ndi
    from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
    from scipy.ndimage import gaussian_filter
except Exception as e:
    raise RuntimeError(f"scipy required: pip install scipy\n{e}")

try:
    import pandas as pd
except Exception as e:
    raise RuntimeError(f"pandas required: pip install pandas\n{e}")

# Try psutil (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

def load_image(img_path):
    """Load image as float; keep original dynamic range (may exceed 1.0)."""
    img = img_as_float(io.imread(img_path))
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img

def compute_entropy(img):
    """Entropy computed on clipped [0,1] scaled to 8-bit histogram."""
    img_clipped = np.clip(img, 0.0, 1.0)
    img8 = (rescale_intensity(img_clipped, out_range=(0,255)).astype(np.uint8)).ravel()
    hist, _ = np.histogram(img8, bins=256, range=(0,255), density=True)
    hist = hist + 1e-12
    return -np.sum(hist * np.log2(hist))

def ssim_compat(a, b, **kwargs):
    """Wrapper for SSIM supporting older/newer scikit-image arg names."""
    try:
        return ssim(a, b, **kwargs)
    except TypeError:
        if 'channel_axis' in kwargs:
            ca = kwargs.pop('channel_axis')
            try:
                return ssim(a, b, multichannel=(ca is not None), **kwargs)
            except TypeError:
                return ssim(np.mean(a, axis=2) if a.ndim == 3 else a,
                            np.mean(b, axis=2) if b.ndim == 3 else b, **kwargs)
        else:
            return ssim(np.mean(a, axis=2) if a.ndim == 3 else a,
                        np.mean(b, axis=2) if b.ndim == 3 else b, **kwargs)

def estimate_sigma_compat(img):
    """Wrapper for estimate_sigma across scikit-image versions."""
    try:
        return estimate_sigma(img, channel_axis=None)
    except TypeError:
        try:
            return estimate_sigma(img, multichannel=False)
        except Exception:
            return np.std(img - gaussian_filter(img, sigma=1.0))

def safe_normalize_for_display(img):
    """Safely normalize image for display, handling edge cases."""
    img_clipped = np.clip(img, 0, None)
    max_val = img_clipped.max()
    if max_val > 1.0 and max_val > 1e-10:  # Avoid division by near-zero
        return np.clip(img_clipped / max_val, 0, 1)
    return np.clip(img_clipped, 0, 1)

###############################################################################
# NOISE GENERATORS
###############################################################################

def add_gaussian_noise(img, sigma=0.06):
    img_float = img_as_float(img)
    return random_noise(img_float, mode='gaussian', var=sigma**2)

def add_poisson_noise(img, peak=255.0):
    img = img_as_float(img)
    img_clipped = np.clip(img, 0.0, None)
    img_max = img_clipped.max()
    if img_max <= 0:
        return img_clipped.copy()
    normalize_back = False
    if img_max > 1.0:
        img_norm = img_clipped / img_max
        normalize_back = True
    else:
        img_norm = img_clipped
    lam = np.clip(img_norm, 0.0, 1.0) * float(peak)
    noisy = np.random.poisson(lam).astype(np.float32) / float(peak)
    noisy = np.clip(noisy, 0.0, 1.0)
    if normalize_back:
        noisy = noisy * img_max
    return noisy

def add_salt_pepper(img, amount=0.06):
    img_float = img_as_float(img)
    return random_noise(img_float, mode='s&p', amount=amount)

def add_speckle(img, var=0.04):
    img_float = img_as_float(img)
    return random_noise(img_float, mode='speckle', var=var)

###############################################################################
# DENOISING METHODS
###############################################################################

def denoise_frequency_domain(noisy):
    if noisy.ndim == 3:
        result = np.zeros_like(noisy)
        for c in range(noisy.shape[2]):
            result[:,:,c] = denoise_frequency_domain_single_channel(noisy[:,:,c])
        return np.clip(result, 0, None)
    else:
        return denoise_frequency_domain_single_channel(noisy)

def denoise_frequency_domain_single_channel(noisy):
    noisy = np.asarray(noisy, dtype=float)
    rows, cols = noisy.shape
    try:
        sigma_est = float(estimate_sigma_compat(np.clip(noisy, 0.0, 1.0)))
    except Exception:
        sigma_est = float(np.std(noisy - gaussian_filter(noisy, sigma=1.0)))
    sigma_norm = min(1.0, sigma_est / 0.2)
    cutoff_fraction = 0.15 + 0.35 * (1.0 - sigma_norm)
    cutoff = max(2.0, min(rows, cols) * cutoff_fraction)
    # FIX: Ensure sigma is never too small to avoid division by zero
    sigma = max(cutoff, 0.1)
    f_transform = fft2(noisy)
    f_shift = fftshift(f_transform)
    y, x = np.ogrid[:rows, :cols]
    crow, ccol = rows // 2, cols // 2
    dist2 = (y - crow)**2 + (x - ccol)**2
    mask = np.exp(-dist2 / (2.0 * sigma * sigma))
    f_shift_filtered = f_shift * mask
    f_ishift = ifftshift(f_shift_filtered)
    img_back = ifft2(f_ishift)
    img_back = np.real(img_back)
    return np.clip(img_back, 0, None)

def anscombe_transform(img):
    """
    Anscombe variance-stabilizing transform for Poisson noise.
    Formula: y = 2*sqrt(x + 3/8)
    """
    img = np.clip(img, 0.0, None)
    return 2.0 * np.sqrt(img + 3.0/8.0)

def inverse_anscombe_transform_naive(trans_img):
    """
    Naive algebraic inverse of Anscombe transform.
    Formula: x = (y/2)² - 3/8
    
    This is the simple mathematical inverse, but is BIASED for low intensities.
    Use for comparison or educational purposes.
    """
    y = np.array(trans_img, dtype=float)
    return np.clip((y / 2.0)**2 - 3.0/8.0, 0.0, None)

def inverse_anscombe_transform_exact(trans_img):
    """
    Mäkitalo & Foi (2011) asymptotically unbiased inverse.
    Formula: x = (y/2)² - 1/8
    
    This provides better accuracy than the naive inverse, especially for 
    low-intensity Poisson noise. Recommended for practical denoising.
    
    Reference: Mäkitalo, M., & Foi, A. (2011). Optimal inversion of the 
    Anscombe transformation in low-count Poisson image denoising.
    IEEE Trans. on Image Processing, 20(1), 99-109.
    """
    y = np.array(trans_img, dtype=float)
    return np.clip((y / 2.0)**2 - 1.0/8.0, 0.0, None)

def denoise_nl_means_compat(noisy, h, patch_kw):
    try:
        return denoise_nl_means(noisy, h=h, fast_mode=True, channel_axis=None, **patch_kw)
    except TypeError:
        try:
            return denoise_nl_means(noisy, h=h, fast_mode=True, multichannel=False, **patch_kw)
        except Exception:
            return gaussian_filter(noisy, sigma=1.0)

def denoise_anscombe_naive(noisy):
    """Anscombe denoising with naive inverse (-3/8)."""
    if noisy.ndim == 3:
        result = np.zeros_like(noisy)
        for c in range(noisy.shape[2]):
            result[:,:,c] = denoise_anscombe_single_channel(noisy[:,:,c], use_exact=False)
        return np.clip(result, 0, None)
    else:
        return denoise_anscombe_single_channel(noisy, use_exact=False)

def denoise_anscombe_exact(noisy):
    """Anscombe denoising with exact inverse (-1/8, Mäkitalo & Foi 2011)."""
    if noisy.ndim == 3:
        result = np.zeros_like(noisy)
        for c in range(noisy.shape[2]):
            result[:,:,c] = denoise_anscombe_single_channel(noisy[:,:,c], use_exact=True)
        return np.clip(result, 0, None)
    else:
        return denoise_anscombe_single_channel(noisy, use_exact=True)

def denoise_anscombe_single_channel(noisy, use_exact=True):
    """
    Denoise using Anscombe transform with choice of inverse.
    
    Args:
        noisy: Noisy image channel
        use_exact: If True, use Mäkitalo & Foi exact inverse (-1/8)
                   If False, use naive algebraic inverse (-3/8)
    """
    noisy = np.clip(noisy, 0.0, None)
    trans = anscombe_transform(noisy)
    try:
        sigma_est = estimate_sigma_compat(trans)
    except Exception:
        sigma_est = np.std(trans - gaussian_filter(trans, sigma=1.0))
        if sigma_est <= 0:
            sigma_est = 0.01
    h = 1.15 * sigma_est
    patch_kw = dict(patch_size=5, patch_distance=6)
    try:
        denoised_trans = denoise_nl_means_compat(trans, h=h, patch_kw=patch_kw)
    except Exception:
        denoised_trans = gaussian_filter(trans, sigma=1.0)
    
    # Choose inverse based on parameter
    if use_exact:
        result = inverse_anscombe_transform_exact(denoised_trans)
    else:
        result = inverse_anscombe_transform_naive(denoised_trans)
    
    return np.clip(result, 0, None)

def median_compat(img, radius=2):
    try:
        return median(img, footprint=disk(radius), channel_axis=2 if img.ndim==3 else None)
    except TypeError:
        if img.ndim == 3:
            out = np.zeros_like(img)
            for c in range(img.shape[2]):
                out[:,:,c] = median(img[:,:,c], footprint=disk(radius))
            return out
        else:
            return median(img, footprint=disk(radius))

def denoise_median_clustering(noisy):
    if noisy.ndim == 3:
        denoised = median_compat(noisy, radius=2)
        denoised = median_compat(denoised, radius=1)
        return np.clip(denoised, 0, None)
    else:
        denoised = median_compat(noisy, radius=2)
        denoised = median_compat(denoised, radius=1)
        return np.clip(denoised, 0, None)

def denoise_log_gaussian(noisy):
    if noisy.ndim == 3:
        result = np.zeros_like(noisy)
        for c in range(noisy.shape[2]):
            result[:,:,c] = denoise_log_gaussian_single_channel(noisy[:,:,c])
        return np.clip(result, 0, None)
    else:
        return denoise_log_gaussian_single_channel(noisy)

def denoise_log_gaussian_single_channel(noisy):
    # FIX: Increased epsilon for better stability with salt-and-pepper
    epsilon = 1e-3
    noisy_safe = np.clip(noisy, epsilon, None)
    log_img = np.log(noisy_safe)
    log_denoised = gaussian_filter(log_img, sigma=1.0)
    denoised = np.exp(log_denoised)
    return np.clip(denoised, 0, None)

def denoise_tv_wrapper(noisy, weight=0.1):
    if noisy.ndim == 3:
        out = np.zeros_like(noisy)
        for c in range(noisy.shape[2]):
            try:
                out[:,:,c] = denoise_tv_chambolle(noisy[:,:,c], weight=weight)
            except Exception:
                out[:,:,c] = gaussian_filter(noisy[:,:,c], sigma=0.5)
        return np.clip(out, 0, None)
    else:
        try:
            return np.clip(denoise_tv_chambolle(noisy, weight=weight), 0, None)
        except Exception:
            return np.clip(gaussian_filter(noisy, sigma=0.5), 0, None)

###############################################################################
# MAIN PROCESSING (computes metrics, saves images + error maps)
###############################################################################

def process_image(img_path, out_dir="./denoising_outputs", show_plots=False):
    if isinstance(img_path, str) and os.path.isfile(img_path):
        original = load_image(img_path)
        image_name = os.path.splitext(os.path.basename(img_path))[0]
    else:
        original = img_as_float(data.camera())
        image_name = "camera"

    orig_max = original.max() if original.size else 1.0
    is_color = (original.ndim == 3)

    noise_funcs = {
        'gaussian': add_gaussian_noise,
        'poisson': add_poisson_noise,
        'salt_pepper': add_salt_pepper,
        'speckle': add_speckle
    }

    denoise_methods = {
        'frequency_domain': denoise_frequency_domain,
        'anscombe_naive': denoise_anscombe_naive,      # Uses -3/8 inverse
        'anscombe_exact': denoise_anscombe_exact,      # Uses -1/8 inverse (Mäkitalo & Foi)
        'median_clustering': denoise_median_clustering,
        'log_gaussian': denoise_log_gaussian,
        'tv_chambolle': denoise_tv_wrapper
    }

    noisy_versions = {}
    for noise_name, noise_func in noise_funcs.items():
        if noise_name == 'poisson':
            noisy_versions[noise_name] = noise_func(original, peak=255.0)
        else:
            noisy_versions[noise_name] = noise_func(original)

    results = {}
    for noise_name, noisy_img in noisy_versions.items():
        results[noise_name] = {}
        for method_name, denoise_func in denoise_methods.items():
            try:
                denoised = denoise_func(noisy_img)
                results[noise_name][method_name] = denoised
            except Exception as e:
                print(f"  Warning: {method_name} failed on {noise_name}: {e}")
                results[noise_name][method_name] = noisy_img

    rows = []
    for noise_name, noisy_img in noisy_versions.items():
        for method_name, denoised_img in results[noise_name].items():
            orig = np.clip(original, 0.0, None)
            noisy = np.clip(noisy_img, 0.0, None)
            den = np.clip(denoised_img, 0.0, None)

            data_range_val = float(max(orig.max(), 1.0))
            try:
                if is_color:
                    ssim_noisy = ssim_compat(orig, noisy, data_range=data_range_val, channel_axis=2)
                    psnr_noisy = psnr(orig, noisy, data_range=data_range_val)
                    ssim_denoised = ssim_compat(orig, den, data_range=data_range_val, channel_axis=2)
                    psnr_denoised = psnr(orig, den, data_range=data_range_val)
                else:
                    ssim_noisy = ssim_compat(orig, noisy, data_range=data_range_val)
                    psnr_noisy = psnr(orig, noisy, data_range=data_range_val)
                    ssim_denoised = ssim_compat(orig, den, data_range=data_range_val)
                    psnr_denoised = psnr(orig, den, data_range=data_range_val)
            except Exception as e:
                print(f"  Warning: SSIM/PSNR computation failed for {noise_name} / {method_name}: {e}")
                ssim_noisy = 0.0
                psnr_noisy = 0.0
                ssim_denoised = 0.0
                psnr_denoised = 0.0

            if is_color:
                try:
                    orig_gray = color.rgb2gray(orig)
                    noisy_gray = color.rgb2gray(noisy)
                    denoised_gray = color.rgb2gray(den)
                except Exception:
                    orig_gray = np.mean(orig, axis=2)
                    noisy_gray = np.mean(noisy, axis=2)
                    denoised_gray = np.mean(den, axis=2)
            else:
                orig_gray = orig
                noisy_gray = noisy
                denoised_gray = den

            orig_flat = orig.ravel()
            noisy_flat = noisy.ravel()
            den_flat = den.ravel()

            pixel_abs_error = float(np.mean(np.abs(den_flat - orig_flat)))
            pixel_mse = float(np.mean((den_flat - orig_flat)**2))
            improved_fraction = float(np.mean(np.abs(den_flat - orig_flat) < np.abs(noisy_flat - orig_flat)))
            
            # FIX: Better NaN/Inf handling for correlation
            try:
                if orig_flat.size > 1 and np.std(orig_flat) > 0 and np.std(den_flat) > 0:
                    corr = np.corrcoef(orig_flat, den_flat)[0,1]
                    pixel_corr = 0.0 if (np.isnan(corr) or np.isinf(corr)) else float(corr)
                else:
                    pixel_corr = 0.0
            except Exception:
                pixel_corr = 0.0

            max_pixel_error = float(np.max(np.abs(den_flat - orig_flat)))
            p99_pixel_error = float(np.percentile(np.abs(den_flat - orig_flat), 99.0))

            entry = {
                'image_name': image_name,
                'noise_type': noise_name,
                'denoise_method': method_name,
                'ssim_noisy': float(ssim_noisy),
                'psnr_noisy': float(psnr_noisy),
                'ssim_denoised': float(ssim_denoised),
                'psnr_denoised': float(psnr_denoised),
                'ssim_improvement': float(ssim_denoised - ssim_noisy),
                'psnr_improvement': float(psnr_denoised - psnr_noisy),
                'entropy_orig': float(compute_entropy(orig_gray)),
                'entropy_noisy': float(compute_entropy(noisy_gray)),
                'entropy_denoised': float(compute_entropy(denoised_gray)),
                'noise_var_noisy': float(np.var((noisy - orig).ravel())),
                'noise_std_noisy': float(np.std((noisy - orig).ravel())),
                'noise_var_denoised': float(np.var((den - orig).ravel())),
                'noise_std_denoised': float(np.std((den - orig).ravel())),
                'pixel_abs_error': pixel_abs_error,
                'pixel_mse': pixel_mse,
                'improved_fraction': improved_fraction,
                'pixel_corr': pixel_corr,
                'max_pixel_error': max_pixel_error,
                'p99_pixel_error': p99_pixel_error
            }
            rows.append(entry)

    df = pd.DataFrame(rows)

    os.makedirs(out_dir, exist_ok=True)
    denoised_dir = os.path.join(out_dir, "denoised_images")
    plots_dir = os.path.join(out_dir, "plots")
    csv_dir = os.path.join(out_dir, "csv_data")
    errmaps_dir = os.path.join(out_dir, "error_maps")
    os.makedirs(denoised_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(errmaps_dir, exist_ok=True)

    for noise_name in results:
        for method_name, denoised_img in results[noise_name].items():
            fname = os.path.join(denoised_dir, f"{image_name}_{noise_name}_{method_name}.png")
            try:
                to_save = denoised_img
                if orig_max > 1.0:
                    save_img = np.clip(to_save / orig_max, 0.0, 1.0)
                else:
                    save_img = np.clip(to_save, 0.0, 1.0)
                io.imsave(fname, img_as_ubyte(save_img))
            except Exception:
                try:
                    io.imsave(fname, (rescale_intensity(np.clip(denoised_img, 0, 1), out_range=(0,255)).astype(np.uint8)))
                except Exception as e:
                    print(f"  Warning: Could not save denoised image {fname}: {e}")

            try:
                den = np.clip(denoised_img, 0.0, None)
                orig_for_err = np.clip(original, 0.0, None)
                dyn = max(orig_for_err.max(), 1.0)
                err_map = np.abs(den - orig_for_err) / dyn
                err_name = os.path.join(errmaps_dir, f"{image_name}_{noise_name}_{method_name}_err.png")
                io.imsave(err_name, img_as_ubyte(np.clip(err_map, 0.0, 1.0)))
            except Exception:
                pass

    if show_plots:
        create_comparison_plot(original, noisy_versions, results, image_name, plots_dir)

    csv_path = os.path.join(csv_dir, f"{image_name}_full_results.csv")
    df.to_csv(csv_path, index=False)

    return df, out_dir

###############################################################################
# VISUALIZATION & ANALYSIS (create_aggregate_visualization included)
###############################################################################

def create_comparison_plot(original, noisy_versions, results, image_name, plots_dir):
    noise_types = list(noisy_versions.keys())
    method_names = list(results[noise_types[0]].keys())
    n_cols = 2 + len(method_names)
    n_rows = len(noise_types)
    is_color = (original.ndim == 3)
    cmap_choice = None if is_color else 'gray'
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f'Comprehensive Denoising Comparison: {image_name}',
                 fontsize=28, fontweight='bold', y=0.995)
    for i, noise_name in enumerate(noise_types):
        # FIX: Use safe_normalize_for_display
        axes[i, 0].imshow(safe_normalize_for_display(original),
                          cmap=cmap_choice, interpolation='bilinear')
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=22, fontweight='bold', pad=15)
        axes[i, 0].set_ylabel(noise_name.replace('_', ' ').title(),
                             fontsize=20, fontweight='bold', rotation=90, labelpad=15)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(safe_normalize_for_display(noisy_versions[noise_name]),
                          cmap=cmap_choice, interpolation='bilinear')
        if i == 0:
            axes[i, 1].set_title('Noisy', fontsize=22, fontweight='bold', pad=15)
        axes[i, 1].axis('off')
        
        for j, method_name in enumerate(method_names):
            den_img = results[noise_name][method_name]
            axes[i, 2+j].imshow(safe_normalize_for_display(den_img),
                                cmap=cmap_choice, interpolation='bilinear')
            if i == 0:
                display_name = method_name.replace('_', '\n').title()
                axes[i, 2+j].set_title(display_name, fontsize=22, fontweight='bold', pad=15)
            axes[i, 2+j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig_path = os.path.join(plots_dir, f"{image_name}_full_comparison.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"  Saved comparison figure: {fig_path}")
    plt.close()

def create_aggregate_visualization(combined_df, out_dir):
    noise_types = combined_df['noise_type'].unique()
    methods = combined_df['denoise_method'].unique()
    ssim_matrix = np.zeros((len(noise_types), len(methods)))
    psnr_matrix = np.zeros((len(noise_types), len(methods)))
    pixerr_matrix = np.zeros((len(noise_types), len(methods)))
    for i, noise in enumerate(noise_types):
        for j, method in enumerate(methods):
            subset = combined_df[(combined_df['noise_type'] == noise) &
                                (combined_df['denoise_method'] == method)]
            ssim_matrix[i, j] = subset['ssim_denoised'].mean()
            psnr_matrix[i, j] = subset['psnr_denoised'].mean()
            pixerr_matrix[i, j] = subset['pixel_abs_error'].mean()
    fig, axes = plt.subplots(1, 3, figsize=(36, 12))
    method_labels = [m.replace('_', '\n').title() for m in methods]
    noise_labels = [n.replace('_', ' ').title() for n in noise_types]
    im1 = axes[0].imshow(ssim_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[0].set_xticks(np.arange(len(methods)))
    axes[0].set_yticks(np.arange(len(noise_types)))
    axes[0].set_xticklabels(method_labels, fontsize=14, fontweight='bold')
    axes[0].set_yticklabels(noise_labels, fontsize=14, fontweight='bold')
    axes[0].set_title('SSIM Performance Matrix\n(Higher is Better)', fontsize=18, fontweight='bold')
    for i in range(len(noise_types)):
        for j in range(len(methods)):
            row_max = ssim_matrix[i, :].max()
            is_best = (ssim_matrix[i, j] == row_max)
            text_color = 'white' if ssim_matrix[i, j] < 0.5 else 'black'
            text_weight = 'bold' if is_best else 'normal'
            text_str = f'{ssim_matrix[i, j]:.4f}'
            if is_best:
                text_str = f'★ {text_str} ★'
            axes[0].text(j, i, text_str, ha="center", va="center", color=text_color, fontsize=12, fontweight=text_weight)
    cbar1 = plt.colorbar(im1, ax=axes[0], pad=0.02)
    cbar1.ax.tick_params(labelsize=12)
    cbar1.set_label('SSIM', fontsize=12, fontweight='bold')
    im2 = axes[1].imshow(psnr_matrix, cmap='RdYlGn', aspect='auto')
    axes[1].set_xticks(np.arange(len(methods)))
    axes[1].set_yticks(np.arange(len(noise_types)))
    axes[1].set_xticklabels(method_labels, fontsize=14, fontweight='bold')
    axes[1].set_yticklabels(noise_labels, fontsize=14, fontweight='bold')
    axes[1].set_title('PSNR Performance Matrix (dB)\n(Higher is Better)', fontsize=18, fontweight='bold')
    for i in range(len(noise_types)):
        for j in range(len(methods)):
            row_max = psnr_matrix[i, :].max()
            is_best = (psnr_matrix[i, j] == row_max)
            text_color = 'white' if psnr_matrix[i, j] < 15 else 'black'
            text_weight = 'bold' if is_best else 'normal'
            text_str = f'{psnr_matrix[i, j]:.2f}'
            if is_best:
                text_str = f'★ {text_str} ★'
            axes[1].text(j, i, text_str, ha="center", va="center", color=text_color, fontsize=12, fontweight=text_weight)
    cbar2 = plt.colorbar(im2, ax=axes[1], pad=0.02)
    cbar2.ax.tick_params(labelsize=12)
    cbar2.set_label('PSNR (dB)', fontsize=12, fontweight='bold')
    im3 = axes[2].imshow(pixerr_matrix, cmap='RdYlGn_r', aspect='auto')
    axes[2].set_xticks(np.arange(len(methods)))
    axes[2].set_yticks(np.arange(len(noise_types)))
    axes[2].set_xticklabels(method_labels, fontsize=14, fontweight='bold')
    axes[2].set_yticklabels(noise_labels, fontsize=14, fontweight='bold')
    axes[2].set_title('Mean Absolute Pixel Error\n(Lower is Better)', fontsize=18, fontweight='bold')
    for i in range(len(noise_types)):
        for j in range(len(methods)):
            row_min = pixerr_matrix[i, :].min()
            is_best = (pixerr_matrix[i, j] == row_min)
            text_color = 'white' if pixerr_matrix[i, j] > (np.max(pixerr_matrix)/2.0) else 'black'
            text_weight = 'bold' if is_best else 'normal'
            text_str = f'{pixerr_matrix[i, j]:.4f}'
            if is_best:
                text_str = f'★ {text_str} ★'
            axes[2].text(j, i, text_str, ha="center", va="center", color=text_color, fontsize=12, fontweight=text_weight)
    cbar3 = plt.colorbar(im3, ax=axes[2], pad=0.02)
    cbar3.ax.tick_params(labelsize=12)
    cbar3.set_label('Mean Abs Pixel Error', fontsize=12, fontweight='bold')
    plt.tight_layout()
    heatmap_path = os.path.join(out_dir, "aggregate_heatmap.png")
    plt.savefig(heatmap_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved performance heatmap: {heatmap_path}")
    plt.close()

def _worker_process_image(img_path, out_dir, show_plots):
    try:
        df, _ = process_image(img_path, out_dir, show_plots)
        return df
    except Exception as e:
        print(f"  ERROR processing {os.path.basename(img_path)}: {e}")
        return None

def auto_detect_workers():
    cpu_count = os.cpu_count() or 1
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        if available_gb < 2:
            return 1
        elif available_gb < 4:
            return min(2, cpu_count)
        elif available_gb < 8:
            return min(4, max(1, cpu_count - 1))
        else:
            return max(1, cpu_count - 1)
    if cpu_count <= 2:
        return 1
    elif cpu_count <= 4:
        return 2
    else:
        return max(1, cpu_count - 1)

def process_folder(folder_path, out_dir="./denoising_outputs", workers=None,
                   save_individual_plots=True, max_plots=10):
    image_files = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp'):
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    if not image_files:
        print(f"No images found in {folder_path}")
        return None, out_dir  # FIX: Consistent return
    print(f"\nFound {len(image_files)} images in {folder_path}")
    if workers is None:
        if len(image_files) < 20:
            workers = 1
            print("Auto-detected: Using sequential processing (< 20 images)")
        else:
            workers = auto_detect_workers()
            print(f"Auto-detected: Using {workers} worker(s) based on system resources")
    save_plots_for = set()
    if save_individual_plots and len(image_files) > max_plots:
        print(f"Large dataset detected: Creating plots for {max_plots} representative images")
        step = max(1, len(image_files) // max_plots)
        for i in range(max_plots):
            save_plots_for.add(i * step)
    elif save_individual_plots:
        save_plots_for = set(range(len(image_files)))
    print("="*70)
    workers = int(max(1, workers))
    if workers == 1:
        print("Running in SEQUENTIAL mode (processing one image at a time)")
    else:
        print(f"Running in PARALLEL mode with {workers} worker processes")
    os.makedirs(out_dir, exist_ok=True)
    tasks = []
    for idx, img_path in enumerate(image_files):
        should_save = (idx in save_plots_for) and save_individual_plots
        tasks.append((img_path, out_dir, should_save))
    all_results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_info = {
            executor.submit(_worker_process_image, t[0], t[1], t[2]): (idx, t[0])
            for idx, t in enumerate(tasks, start=1)
        }
        completed = 0
        total = len(future_to_info)
        for future in as_completed(future_to_info):
            idx, img_path = future_to_info[future]
            completed += 1
            img_name = os.path.basename(img_path)
            try:
                res = future.result()
                if res is None:
                    print(f"  [{completed}/{total}] ✗ {img_name} (failed)")
                else:
                    all_results.append(res)
                    print(f"  [{completed}/{total}] ✓ {img_name}")
            except Exception as e:
                print(f"  [{completed}/{total}] ✗ {img_name}: {e}")
    if not all_results:
        print("\n❌ No images processed successfully")
        return None, out_dir  # FIX: Consistent return
    combined_df = pd.concat(all_results, ignore_index=True)
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS")
    print("="*70)
    analyze_results(combined_df, out_dir)
    return combined_df, out_dir  # FIX: Consistent return (2 values)

def analyze_results(combined_df, out_dir):
    print("\n📊 BEST METHOD FOR EACH NOISE TYPE (Testing Hypotheses):")
    print("-" * 70)
    for noise_type in combined_df['noise_type'].unique():
        subset = combined_df[combined_df['noise_type'] == noise_type]
        avg_by_method = subset.groupby('denoise_method').agg({
            'ssim_denoised': 'mean',
            'psnr_denoised': 'mean',
            'pixel_abs_error': 'mean',
            'improved_fraction': 'mean'
        }).round(4)
        best_method_ssim = avg_by_method['ssim_denoised'].idxmax()
        best_method_psnr = avg_by_method['psnr_denoised'].idxmax()
        best_method_pixel = avg_by_method['pixel_abs_error'].idxmin()
        print(f"\n{noise_type.upper()} Noise:")
        print(f"  Best by SSIM: {best_method_ssim} (SSIM: {avg_by_method.loc[best_method_ssim, 'ssim_denoised']:.4f})")
        print(f"  Best by PSNR: {best_method_psnr} (PSNR: {avg_by_method.loc[best_method_psnr, 'psnr_denoised']:.2f} dB)")
        print(f"  Best by Pixel Abs Error: {best_method_pixel} (AbsErr: {avg_by_method.loc[best_method_pixel, 'pixel_abs_error']:.4f})")
        print("\n  All methods:")
        for method in avg_by_method.index:
            ssim_v = avg_by_method.loc[method, 'ssim_denoised']
            psnr_v = avg_by_method.loc[method, 'psnr_denoised']
            px_v = avg_by_method.loc[method, 'pixel_abs_error']
            marker = "⭐" if method in [best_method_ssim, best_method_psnr, best_method_pixel] else "  "
            print(f"  {marker} {method:20s}: SSIM={ssim_v:.4f}, PSNR={psnr_v:.2f} dB, AbsErr={px_v:.4f}")
    print("\n" + "="*70)
    print("🏆 OVERALL BEST DENOISING METHOD:")
    print("-" * 70)
    overall_avg = combined_df.groupby('denoise_method').agg({
        'ssim_denoised': ['mean', 'std'],
        'psnr_denoised': ['mean', 'std'],
        'pixel_abs_error': ['mean', 'std']
    }).round(4)
    overall_avg = overall_avg.sort_values(('ssim_denoised', 'mean'), ascending=False)
    print(overall_avg)
    best_overall = overall_avg.index[0]
    print(f"\n⭐ WINNER: {best_overall}")
    print(f"   Average SSIM: {overall_avg.loc[best_overall, ('ssim_denoised', 'mean')]:.4f}")
    print(f"   Average PSNR: {overall_avg.loc[best_overall, ('psnr_denoised', 'mean')]:.2f} dB")
    
    # Add analysis comparing naive vs exact Anscombe
    print("\n" + "="*70)
    print("🔬 ANSCOMBE INVERSE COMPARISON (Naive -3/8 vs Exact -1/8):")
    print("-" * 70)
    anscombe_methods = [m for m in combined_df['denoise_method'].unique() if 'anscombe' in m]
    if len(anscombe_methods) >= 2:
        for noise_type in combined_df['noise_type'].unique():
            print(f"\n{noise_type.upper()} Noise:")
            for method in anscombe_methods:
                subset = combined_df[(combined_df['noise_type'] == noise_type) & 
                                    (combined_df['denoise_method'] == method)]
                if len(subset) > 0:
                    avg_ssim = subset['ssim_denoised'].mean()
                    avg_psnr = subset['psnr_denoised'].mean()
                    avg_err = subset['pixel_abs_error'].mean()
                    variant = "NAIVE (-3/8)" if "naive" in method else "EXACT (-1/8, Mäkitalo & Foi)"
                    print(f"  {variant:30s}: SSIM={avg_ssim:.4f}, PSNR={avg_psnr:.2f} dB, AbsErr={avg_err:.4f}")
    
    print("\n" + "="*70)
    print("Saving detailed results...")
    aggregate_path = os.path.join(out_dir, "aggregate_statistics.csv")
    overall_avg.to_csv(aggregate_path)
    print(f"✓ Saved aggregate statistics: {aggregate_path}")
    best_per_noise = []
    for noise_type in combined_df['noise_type'].unique():
        subset = combined_df[combined_df['noise_type'] == noise_type]
        avg_by_method = subset.groupby('denoise_method').agg({
            'ssim_denoised': 'mean',
            'psnr_denoised': 'mean',
            'pixel_abs_error': 'mean'
        })
        best_method = avg_by_method['ssim_denoised'].idxmax()
        best_per_noise.append({
            'noise_type': noise_type,
            'best_method': best_method,
            'ssim': avg_by_method.loc[best_method, 'ssim_denoised'],
            'psnr': avg_by_method.loc[best_method, 'psnr_denoised'],
            'pixel_abs_error': avg_by_method.loc[best_method, 'pixel_abs_error']
        })
    best_per_noise_df = pd.DataFrame(best_per_noise)
    best_path = os.path.join(out_dir, "best_method_per_noise.csv")
    best_per_noise_df.to_csv(best_path, index=False)
    print(f"✓ Saved best methods per noise type: {best_path}")
    combined_path = os.path.join(out_dir, "all_results_combined.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"✓ Saved all combined results: {combined_path}")
    create_aggregate_visualization(combined_df, out_dir)
    print("\n" + "="*70)
    print("✓ ALL DONE! Check the following files:")
    print("="*70)
    print(f"  1. {aggregate_path}")
    print(f"  2. {best_path}")
    print(f"  3. {combined_path}")
    print(f"  4. {os.path.join(out_dir, 'aggregate_heatmap.png')}")
    print("="*70)

###############################################################################
# ENTRYPOINT
###############################################################################

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isdir(path):
            print(f"Processing folder: {path}")
            combined_df, out_dir = process_folder(path)
            if combined_df is not None:
                print(f"\n✓ Processed folder: {path}")
                print(f"✓ Results saved to: {out_dir}")
                sys.exit(0)
            else:
                print("✗ Processing failed or found no images.")
                sys.exit(2)
        elif os.path.isfile(path):
            print(f"Processing file: {path}")
            df, out_dir = process_image(path, show_plots=True)
            print(f"\n✓ Processed file: {path}")
            print(f"✓ Results saved to: {out_dir}")
            sys.exit(0)
        else:
            print(f"Error: {path} is not a valid file or directory")
            sys.exit(1)
    else:
        print("No input provided: using built-in sample image (camera).")
        df, out_dir = process_image(None, show_plots=True)
        print(f"\n✓ Results saved to: {out_dir}")
        sys.exit(0)