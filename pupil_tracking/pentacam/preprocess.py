"""Pentacam input handling.

Single source of truth for turning an arbitrary Pentacam input -- a cropped
eye image, a raw grayscale cross-section, or a full device screenshot with UI
chrome -- into a clean ocular image plus the region that actually contains
the eye.

Design constraints honoured here:

* No layout coordinates are assumed.  The eye region is derived from image
  content only, so the same code path serves cropped images and screenshots.
* Purely deterministic: no randomness, no learned models, no global mutable
  state.
* Source images are never mutated in place.

``build_ui_mask`` is the shared implementation used both by
:class:`~pupil_tracking.pentacam.detector.PentacamIrisDetector` and by
:mod:`pupil_tracking.pentacam.reference`; there is only one copy of it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a grayscale view of ``image`` without mutating the source.

    Accepts single-channel arrays (returned as-is) and BGR/BGRA arrays
    (converted).  The returned array is never a view of ``image`` unless
    ``image`` is already grayscale and ``copy`` is not requested, so callers
    that write into the result cannot corrupt their input.
    """
    if image is None:
        raise ValueError("image is None")
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return image[:, :, 0]
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"unsupported image shape: {image.shape}")


def build_ui_mask(gray: np.ndarray) -> np.ndarray:
    """Create a mask where valid ocular image is 255 and UI chrome/text is 0.

    Detects and suppresses:
        - Sharp horizontal/vertical UI line overlays and crosshairs
        - Bright annotation text (characters, numbers)
        - Pure black letterboxing/chrome margins

    The rule is purely content-based (black level, near-white level, long thin
    morphological line responses), so it applies to cropped images, raw
    cross-sections and device screenshots alike without any hardcoded layout.
    """
    h, w = gray.shape[:2]
    usable = np.ones((h, w), dtype=np.uint8) * 255

    # Pure black border/chrome margin
    black_thresh = 5
    usable[gray <= black_thresh] = 0

    # Highly saturated synthetic pure white text / graphics (e.g. > 252)
    white_thresh = 252
    white_pixels = gray >= white_thresh
    if np.any(white_pixels):
        # Dilate text/lines to eliminate anti-aliased feather edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated_white = cv2.dilate(white_pixels.astype(np.uint8), kernel)
        usable[dilated_white > 0] = 0

    # Detect horizontal and vertical crosshair / reticle lines
    # Using morphological opening with thin long kernels
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    edges = cv2.Canny(gray, 100, 200)
    h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)

    lines_mask = cv2.bitwise_or(h_lines, v_lines)
    if np.any(lines_mask):
        lines_dilated = cv2.dilate(lines_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        usable[lines_dilated > 0] = 0

    return usable


def content_bbox(
    ui_mask: np.ndarray,
    min_fraction: float = 0.02,
) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box ``(x, y, w, h)`` of retained (non-UI) image content.

    Returns ``None`` when too little of the frame survives UI suppression to
    call anything an eye region.  ``min_fraction`` is the minimum fraction of
    total pixels that must remain usable.
    """
    ys, xs = np.nonzero(ui_mask)
    if ys.size == 0:
        return None
    total = float(ui_mask.size)
    if (ys.size / total) < float(min_fraction):
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, (x1 - x0 + 1), (y1 - y0 + 1)


def normalize_gray(gray_roi: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Robustly contrast-stretch a grayscale ROI to the full uint8 range.

    Percentile clipping keeps specular highlights and black UI chrome from
    compressing the iris contrast.  Returns ``float32`` in ``[0, 255]``.
    Deterministic and dependency-free.
    """
    if gray_roi is None or gray_roi.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    src = gray_roi.astype(np.float32, copy=False)
    lo = float(np.percentile(src, p_low))
    hi = float(np.percentile(src, p_high))
    if hi - lo < 1e-6:
        # Flat ROI: return zeros rather than amplifying sensor noise.
        return np.zeros_like(src)

    out = (src - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0.0, 255.0)


def crop(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop ``image`` to ``bbox`` clipped to the frame; never mutates input."""
    h, w = image.shape[:2]
    x, y, bw, bh = bbox
    x0 = max(0, min(int(x), w))
    y0 = max(0, min(int(y), h))
    x1 = max(x0, min(int(x) + int(bw), w))
    y1 = max(y0, min(int(y) + int(bh), h))
    return image[y0:y1, x0:x1].copy()
