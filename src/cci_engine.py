# Copyright 2026 Gabriel Zo-Hasina Rasatavohary / ZONOVA Research.
# All rights reserved.
#
# This source code is licensed under the ZONOVA Research Non-Commercial
# Research License v1.0 found in the LICENSE file in the root directory
# of this source tree. Commercial use requires written authorization.
# Contact: zo@research.zonova.io

"""CCI engine — compute the residual signal vector ψ_R for a page.

This module orchestrates the full pipeline:
    1. Build the conservative residual mask (Ω_res^{+δ})
    2. Run Ghost OCR on the residual space
    3. Extract connected residual zones
    4. Compute ink density and Shannon entropy per zone
    5. Aggregate into ρ_ghost, ρ_entropy, and ψ_R
"""

from __future__ import annotations

import logging

import numpy as np

from .ghost_ocr import ghost_ocr_char_count
from .ink_entropy import compute_entropy, compute_ink_density
from .models import BBox, PageCCIResult, ResidualZone
from .residual import compute_residual_mask, extract_residual_zones

logger = logging.getLogger(__name__)


def compute_cci_from_image(
    page_image: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    delta: int = 10,
    min_zone_area: int = 100,
    ocr_lang: str = "eng",
    page_number: int = 1,
) -> PageCCIResult:
    """Compute the CCI residual vector for a single page image.

    This is the main entry point for the demonstration pipeline.

    Args:
        page_image: RGB image ``(H, W, 3)`` as a NumPy array.
        bboxes: Detected bounding boxes ``(x_min, y_min, x_max, y_max)``
                in **pixel** coordinates.
        delta: Geometric tolerance (δ) in pixels.
        min_zone_area: Minimum connected-component area to keep.
        ocr_lang: Tesseract language string.
        page_number: Page number for reporting.

    Returns:
        A :class:`PageCCIResult` with all computed metrics.
    """
    img_h, img_w = page_image.shape[:2]

    # 1. Conservative residual mask
    residual_mask = compute_residual_mask(img_h, img_w, bboxes, delta=delta)

    # 2. Ghost OCR on the full residual surface
    total_ghost_chars, ghost_text = ghost_ocr_char_count(
        page_image, residual_mask, lang=ocr_lang,
    )
    logger.info("  Ghost OCR: %d chars detected outside layout", total_ghost_chars)

    # 3. Connected residual zones
    raw_zones = extract_residual_zones(residual_mask, min_area=min_zone_area)
    logger.info("  %d residual zones extracted", len(raw_zones))

    zones: list[ResidualZone] = []
    weighted_entropies: list[tuple[float, float]] = []

    for bbox, mask in raw_zones:
        ink = compute_ink_density(page_image, bbox, mask)
        ent = compute_entropy(page_image, bbox, mask)
        zones.append(ResidualZone(bbox=bbox, mask=mask, ink_density=ink, entropy=ent))
        weighted_entropies.append((ent, float(np.count_nonzero(mask))))

    # Store ghost text in the first zone for convenience
    if zones and ghost_text:
        z0 = zones[0]
        zones[0] = ResidualZone(
            bbox=z0.bbox, mask=z0.mask,
            ink_density=z0.ink_density, entropy=z0.entropy,
            ghost_text=ghost_text,
        )

    # 4. Aggregate scores -----------------------------------------------

    # ρ_ghost
    estimated_detected = _estimate_detected_chars(bboxes)
    if total_ghost_chars == 0:
        rho_ghost = 1.0
    else:
        rho_ghost = 1.0 - total_ghost_chars / (total_ghost_chars + estimated_detected)

    # ρ_entropy (area-weighted mean)
    if weighted_entropies:
        total_area = sum(a for _, a in weighted_entropies)
        mean_entropy = (
            sum(e * a for e, a in weighted_entropies) / total_area
            if total_area > 0 else 0.0
        )
        rho_entropy = 1.0 - mean_entropy
    else:
        rho_entropy = 1.0

    psi_residual = rho_ghost * rho_entropy

    # Residual area ratio
    total_residual_px = sum(np.count_nonzero(z.mask) for z in zones)
    residual_ratio = total_residual_px / (img_h * img_w) if img_h * img_w > 0 else 0.0

    logger.info(
        "  CCI page %d: ψ_R=%.4f (ρ_ghost=%.4f, ρ_entropy=%.4f), "
        "residual=%.1f%%, ghost_chars=%d",
        page_number, psi_residual, rho_ghost, rho_entropy,
        residual_ratio * 100, total_ghost_chars,
    )

    return PageCCIResult(
        page_number=page_number,
        rho_ghost=rho_ghost,
        rho_entropy=rho_entropy,
        psi_residual=psi_residual,
        num_residual_zones=len(zones),
        total_residual_area_ratio=residual_ratio,
        zones=zones,
    )


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _estimate_detected_chars(
    bboxes: list[tuple[int, int, int, int]],
    char_area: float = 150.0,
    fill_ratio: float = 0.6,
) -> int:
    """Rough heuristic: total bbox area × fill ratio / char area.

    A character occupies ~10×15 px at 200 DPI ⇒ char_area ≈ 150 px².
    """
    total = 0
    for x1, y1, x2, y2 in bboxes:
        total += int((x2 - x1) * (y2 - y1) * fill_ratio / char_area)
    return max(total, 1)
