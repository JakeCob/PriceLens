"""
Image Preprocessing Techniques for Card Detection and Identification

Fast, optimized image enhancement functions designed for real-time processing.
Each function targets specific image quality issues common in card scanning.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def normalize_lighting(image: np.ndarray, target_brightness: int = 127) -> np.ndarray:
    """
    Fast brightness normalization using mean luminance.

    Normalizes overall image brightness to a target value using a lookup table (LUT)
    for maximum speed. Useful for handling dark or overexposed frames.

    Args:
        image: Input BGR image
        target_brightness: Target mean brightness (0-255), default 127 (mid-gray)

    Returns:
        Brightness-normalized BGR image

    Performance: ~0.5-1ms for 1280x720
    """
    if image is None or image.size == 0:
        return image

    # Convert to grayscale for brightness calculation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    # Avoid division by zero for extremely dark images
    if mean_brightness < 10:
        logger.debug(f"Image too dark (mean={mean_brightness:.1f}), skipping normalization")
        return image

    # Calculate scale factor
    scale = target_brightness / mean_brightness
    scale = np.clip(scale, 0.5, 2.5)  # Limit adjustment range to avoid artifacts

    # Apply using LUT for speed (much faster than pixel-wise operations)
    lut = np.clip(np.arange(256) * scale, 0, 255).astype(np.uint8)

    # Apply LUT to each channel
    result = cv2.LUT(image, lut)

    return result


def apply_fast_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: tuple = (4, 4)
) -> np.ndarray:
    """
    Fast CLAHE (Contrast Limited Adaptive Histogram Equalization) for detection.

    Uses smaller tile grid for speed while still providing local contrast enhancement.
    Operates on the L channel of LAB color space to preserve colors.

    Args:
        image: Input BGR image
        clip_limit: Contrast limiting threshold (higher = more contrast)
        tile_size: Grid size for local histogram equalization

    Returns:
        Contrast-enhanced BGR image

    Performance: ~1-2ms for 1280x720 with (4,4) tiles
    """
    if image is None or image.size == 0:
        return image

    # Convert to LAB color space (only enhance L channel)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.apply(l)

    # Reconstruct image
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return result


def apply_enhanced_clahe(
    image: np.ndarray,
    clip_limit: float = 3.0,
    tile_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Enhanced CLAHE for identification with more aggressive settings.

    Uses larger tile grid and higher clip limit for better feature detection
    at the cost of slightly more processing time.

    Args:
        image: Input BGR image
        clip_limit: Contrast limiting threshold (3.0 is more aggressive)
        tile_size: Grid size (8,8 provides finer local adaptation)

    Returns:
        Contrast-enhanced BGR image

    Performance: ~1.5-2ms for card crop (~300x420)
    """
    if image is None or image.size == 0:
        return image

    # Same as fast_clahe but with different parameters
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return result


def white_balance_gray_world(image: np.ndarray) -> np.ndarray:
    """
    Gray World white balance algorithm.

    Assumes the average color of a scene should be gray. Scales each channel
    to make the average equal across R, G, B. Simple but effective for most
    indoor lighting conditions.

    Args:
        image: Input BGR image

    Returns:
        White-balanced BGR image

    Performance: ~0.5-1ms for card crop
    """
    if image is None or image.size == 0:
        return image

    result = image.astype(np.float32)

    # Calculate mean of each channel
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])

    # Calculate overall gray average
    avg_gray = (avg_b + avg_g + avg_r) / 3

    # Avoid division by zero
    if avg_gray < 1:
        return image

    # Scale each channel to match gray average
    if avg_b > 0:
        result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / avg_b), 0, 255)
    if avg_g > 0:
        result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / avg_g), 0, 255)
    if avg_r > 0:
        result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / avg_r), 0, 255)

    return result.astype(np.uint8)


def reduce_glare(
    image: np.ndarray,
    threshold: int = 240,
    blur_kernel: int = 5
) -> np.ndarray:
    """
    Reduce specular highlights (glare) on holographic cards.

    Detects very bright regions (glare) and inpaints them using surrounding pixels.
    Particularly useful for holographic and foil Pokemon cards which often have
    bright reflections that interfere with OCR and feature matching.

    Args:
        image: Input BGR image
        threshold: Brightness threshold for glare detection (0-255)
        blur_kernel: Inpainting radius

    Returns:
        Glare-reduced BGR image

    Performance: ~2-3ms for card crop
    """
    if image is None or image.size == 0:
        return image

    # Convert to grayscale for glare detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect glare regions (very bright areas)
    _, glare_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Check if any glare detected
    if np.sum(glare_mask) == 0:
        return image

    # Dilate mask to cover full glare regions
    kernel = np.ones((3, 3), np.uint8)
    glare_mask = cv2.dilate(glare_mask, kernel, iterations=2)

    # Inpaint the glare regions using surrounding pixels
    # INPAINT_TELEA is fast and produces good results
    result = cv2.inpaint(image, glare_mask, blur_kernel, cv2.INPAINT_TELEA)

    return result


def remove_shadows(image: np.ndarray) -> np.ndarray:
    """
    Remove shadows using morphological operations.

    Estimates background illumination using morphological opening, then divides
    the original image by this estimate to normalize lighting. Effective for
    cards photographed at angles where one side is shadowed.

    Args:
        image: Input BGR image

    Returns:
        Shadow-corrected BGR image

    Performance: ~2-3ms for card crop
    """
    if image is None or image.size == 0:
        return image

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Estimate background illumination using morphological opening
    # Large kernel approximates the "shadow-free" background
    kernel_size = max(l.shape[0] // 4, 15)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd for morphological operations

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(l, cv2.MORPH_OPEN, kernel)

    # Divide original by background to normalize illumination
    # Add small epsilon to avoid division by zero
    normalized = cv2.divide(
        l.astype(np.float32),
        background.astype(np.float32) + 1,
        scale=255
    )
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # Reconstruct image
    lab_corrected = cv2.merge([normalized, a, b])
    result = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    return result


def histogram_stretch(image: np.ndarray, percentile_low: int = 2, percentile_high: int = 98) -> np.ndarray:
    """
    Stretch histogram to full dynamic range using percentiles.

    Uses percentiles instead of min/max to avoid being affected by outliers.
    Improves contrast without clipping important image data.

    Args:
        image: Input BGR image
        percentile_low: Lower percentile for black point
        percentile_high: Upper percentile for white point

    Returns:
        Histogram-stretched BGR image

    Performance: ~1ms for card crop
    """
    if image is None or image.size == 0:
        return image

    # Convert to LAB to stretch only luminance
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Calculate percentile values
    p_low = np.percentile(l, percentile_low)
    p_high = np.percentile(l, percentile_high)

    # Avoid division by zero
    if p_high <= p_low:
        return image

    # Stretch histogram
    l_stretched = np.clip((l.astype(np.float32) - p_low) * 255 / (p_high - p_low), 0, 255)
    l_stretched = l_stretched.astype(np.uint8)

    # Reconstruct
    lab_stretched = cv2.merge([l_stretched, a, b])
    result = cv2.cvtColor(lab_stretched, cv2.COLOR_LAB2BGR)

    return result
