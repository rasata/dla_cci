# Copyright 2026 Gabriel Zo-Hasina Rasatavohary / ZONOVA Research.
# All rights reserved.
#
# This source code is licensed under the ZONOVA Research Non-Commercial
# Research License v1.0 found in the LICENSE file in the root directory
# of this source tree. Commercial use requires written authorization.
# Contact: zo@research.zonova.io

"""Residual mask computation and connected-component extraction.

The residual mask represents the areas of a document page that are *not*
covered by any detected bounding box.  These uncovered zones are then
analysed by Ghost OCR and entropy computation to assess layout completeness.
"""

from __future__ import annotations

import cv2
import numpy as np

from .models import BBox, LayoutElement


def compute_residual_mask(
    image_height: int,
    image_width: int,
    bboxes: list[tuple[int, int, int, int]],
    delta: int = 10,
) -> np.ndarray:
    """Build the conservative (δ-eroded) residual mask.

    Each bounding box is shrunk by *delta* pixels on every side before
    being subtracted from the page.  Pixels that remain uncovered after
    this conservative erosion form the residual space Ω_res^{+δ}.

    Args:
        image_height: Page image height in pixels.
        image_width: Page image width in pixels.
        bboxes: Detected regions as ``(x_min, y_min, x_max, y_max)``
                in pixel coordinates.
        delta: Geometric tolerance in pixels (default 10 px at 200–300 DPI).

    Returns:
        Binary mask ``uint8 (H, W)`` — 255 = residual, 0 = covered.
    """
    mask = np.full((image_height, image_width), 255, dtype=np.uint8)
    for x1, y1, x2, y2 in bboxes:
        # δ-erosion: only mark the strict interior as covered
        ex1 = max(0, min(x1 + delta, image_width))
        ey1 = max(0, min(y1 + delta, image_height))
        ex2 = max(0, min(x2 - delta, image_width))
        ey2 = max(0, min(y2 - delta, image_height))
        if ex2 > ex1 and ey2 > ey1:
            mask[ey1:ey2, ex1:ex2] = 0
    return mask


def compute_residual_mask_from_elements(
    image_height: int,
    image_width: int,
    elements: list[LayoutElement],
    layout_width: int,
    layout_height: int,
    delta: int = 10,
) -> np.ndarray:
    """Build residual mask from :class:`LayoutElement` objects.

    Coordinates are projected from layout space to image space before
    computing the mask.
    """
    scale_x = image_width / layout_width
    scale_y = image_height / layout_height
    bboxes: list[tuple[int, int, int, int]] = []
    for elem in elements:
        bboxes.append((
            int(elem.bbox.x_min * scale_x),
            int(elem.bbox.y_min * scale_y),
            int(elem.bbox.x_max * scale_x),
            int(elem.bbox.y_max * scale_y),
        ))
    return compute_residual_mask(image_height, image_width, bboxes, delta=delta)


def extract_residual_zones(
    residual_mask: np.ndarray,
    min_area: int = 100,
) -> list[tuple[BBox, np.ndarray]]:
    """Extract connected components from the residual mask.

    Args:
        residual_mask: Binary mask (255 = residual).
        min_area: Minimum area in pixels to keep a zone.

    Returns:
        List of ``(BBox, sub_mask)`` for each qualifying zone.
    """
    num_labels, labels = cv2.connectedComponents(residual_mask, connectivity=8)
    zones: list[tuple[BBox, np.ndarray]] = []
    for label_id in range(1, num_labels):
        component_mask = (labels == label_id).astype(np.uint8) * 255
        area = int(np.count_nonzero(component_mask))
        if area < min_area:
            continue
        ys, xs = np.where(component_mask > 0)
        bbox = BBox(
            x_min=float(xs.min()),
            y_min=float(ys.min()),
            x_max=float(xs.max()),
            y_max=float(ys.max()),
        )
        y1, y2 = int(bbox.y_min), int(bbox.y_max) + 1
        x1, x2 = int(bbox.x_min), int(bbox.x_max) + 1
        zones.append((bbox, component_mask[y1:y2, x1:x2]))
    return zones
