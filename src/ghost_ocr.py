# Copyright 2026 Gabriel Zo-Hasina Rasatavohary / ZONOVA Research.
# All rights reserved.
#
# This source code is licensed under the ZONOVA Research Non-Commercial
# Research License v1.0 found in the LICENSE file in the root directory
# of this source tree. Commercial use requires written authorization.
# Contact: zo@research.zonova.io

"""Ghost OCR — OCR on residual (uncovered) zones via Tesseract.

The core idea: blank out every region that the layout model claims to have
detected, then run OCR on whatever remains.  Any text found in the residual
space is direct evidence of an omission.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytesseract


def ghost_ocr_page(
    page_image: np.ndarray,
    residual_mask: np.ndarray,
    lang: str = "eng",
) -> str:
    """Run Tesseract on the residual surface only.

    Covered regions (mask == 0) are painted white so that Tesseract
    ignores them entirely.

    Args:
        page_image: Page image ``(H, W, 3)`` RGB.
        residual_mask: Binary mask ``(H, W)`` uint8 — 255 = residual.
        lang: Tesseract language string.

    Returns:
        Cleaned text detected in uncovered zones.
    """
    image = page_image.copy()
    image[residual_mask == 0] = 255
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray, lang=lang, config="--psm 6")
    return _clean_text(text)


def ghost_ocr_char_count(
    page_image: np.ndarray,
    residual_mask: np.ndarray,
    lang: str = "eng",
) -> tuple[int, str]:
    """Count characters detected by Ghost OCR outside detected regions.

    Returns:
        ``(char_count, cleaned_text)`` — 0 chars means the layout fully
        covers all readable content.
    """
    text = ghost_ocr_page(page_image, residual_mask, lang=lang)
    char_count = len(text.replace("\n", "").replace(" ", ""))
    return char_count, text


def _clean_text(text: str) -> str:
    """Remove OCR noise: keep only lines with ≥ 3 characters."""
    lines = text.strip().splitlines()
    return "\n".join(ln.strip() for ln in lines if len(ln.strip()) >= 3)
