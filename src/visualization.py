# Copyright 2026 Gabriel Zo-Hasina Rasatavohary / ZONOVA Research.
# All rights reserved.
#
# This source code is licensed under the ZONOVA Research Non-Commercial
# Research License v1.0 found in the LICENSE file in the root directory
# of this source tree. Commercial use requires written authorization.
# Contact: zo@research.zonova.io

"""Visualization utilities for the Ghost OCR demonstration.

Produces annotated page images showing:
- Detected bounding boxes (green)
- Residual zones (red overlay)
- Ghost text annotations
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import PageCCIResult


def save_ghost_ocr_visualization(
    image: np.ndarray,
    residual_mask: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    output_path: Path,
    ghost_text: str = "",
) -> Path:
    """Save an annotated image highlighting detected vs. residual zones.

    Args:
        image: Page image ``(H, W, 3)`` RGB.
        residual_mask: Binary mask (255 = residual).
        bboxes: Detected bounding boxes.
        output_path: Destination path for the PNG.
        ghost_text: Optional ghost text to overlay.

    Returns:
        The path to the saved image.
    """
    vis = image.copy()

    # Semi-transparent red overlay on residual zones
    overlay = vis.copy()
    overlay[residual_mask == 255] = [255, 200, 200]
    vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

    # Green rectangles for detected bounding boxes
    for x1, y1, x2, y2 in bboxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 0), 2)

    # If ghost text was found, mark the top-left corner
    if ghost_text:
        cv2.putText(
            vis, f"GHOST: {len(ghost_text)} chars",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 0, 0), 2,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), vis_bgr)
    return output_path
