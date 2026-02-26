# Copyright 2026 Gabriel Zo-Hasina Rasatavohary / ZONOVA Research.
# All rights reserved.
#
# This source code is licensed under the ZONOVA Research Non-Commercial
# Research License v1.0 found in the LICENSE file in the root directory
# of this source tree. Commercial use requires written authorization.
# Contact: zo@research.zonova.io

"""Ink density and Shannon entropy for residual zones.

These two metrics form the second component of the residual signal
vector ψ_R.  High ink density in a residual zone suggests visible content
that was missed; high entropy suggests visual complexity inconsistent
with an empty background.
"""

from __future__ import annotations

import cv2
import numpy as np

from .models import BBox


def compute_ink_density(
    page_image: np.ndarray,
    zone_bbox: BBox,
    zone_mask: np.ndarray,
    threshold: int = 200,
) -> float:
    """Ratio of dark ("ink") pixels inside a residual zone.

    A pixel is considered ink if its grayscale intensity is below
    *threshold* (darker = ink).

    Returns:
        Value in ``[0, 1]``.
    """
    y1, y2 = int(zone_bbox.y_min), int(zone_bbox.y_max) + 1
    x1, x2 = int(zone_bbox.x_min), int(zone_bbox.x_max) + 1
    crop = page_image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    zone_pixels = gray[zone_mask > 0]
    if len(zone_pixels) == 0:
        return 0.0
    return int(np.sum(zone_pixels < threshold)) / len(zone_pixels)


def compute_entropy(
    page_image: np.ndarray,
    zone_bbox: BBox,
    zone_mask: np.ndarray,
    bins: int = 256,
) -> float:
    """Normalised Shannon entropy of the grayscale intensity histogram.

    ``H_norm = −Σ p_i log₂(p_i) / log₂(bins)``

    Returns:
        Value in ``[0, 1]`` — 0 means uniform background, 1 means
        maximally varied content.
    """
    y1, y2 = int(zone_bbox.y_min), int(zone_bbox.y_max) + 1
    x1, x2 = int(zone_bbox.x_min), int(zone_bbox.x_max) + 1
    crop = page_image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    zone_pixels = gray[zone_mask > 0]
    if len(zone_pixels) == 0:
        return 0.0
    hist, _ = np.histogram(zone_pixels, bins=bins, range=(0, 256))
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 0.0
    h = -np.sum(probs * np.log2(probs))
    return float(h / np.log2(bins))


def compute_residual_entropy(
    image: np.ndarray,
    residual_mask: np.ndarray,
) -> float:
    """Global normalised entropy over the entire residual space.

    This is a page-level metric (as opposed to the per-zone version above).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    pixels = gray[residual_mask == 255]
    if len(pixels) == 0:
        return 0.0
    hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 0.0
    h = -np.sum(probs * np.log2(probs))
    return float(h / np.log2(256))
