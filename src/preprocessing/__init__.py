"""
Image Preprocessing Module for PriceLens

Provides image enhancement techniques for improving card detection and identification.
"""

from src.preprocessing.enhancer import ImageEnhancer
from src.preprocessing.techniques import (
    normalize_lighting,
    apply_fast_clahe,
    apply_enhanced_clahe,
    white_balance_gray_world,
    reduce_glare,
    remove_shadows,
)

__all__ = [
    "ImageEnhancer",
    "normalize_lighting",
    "apply_fast_clahe",
    "apply_enhanced_clahe",
    "white_balance_gray_world",
    "reduce_glare",
    "remove_shadows",
]
