# Copyright 2026 Gabriel Zo-Hasina Rasatavohary / ZONOVA Research.
# All rights reserved.
#
# This source code is licensed under the ZONOVA Research Non-Commercial
# Research License v1.0 found in the LICENSE file in the root directory
# of this source tree. Commercial use requires written authorization.
# Contact: zo@research.zonova.io

"""Data structures for the CCI residual vector pipeline.

These dataclasses model the bounding boxes, layout elements, residual zones,
and per-page CCI scores used throughout the Ghost OCR demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box (absolute coordinates)."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class LayoutElement:
    """A single element detected by a layout analysis model."""

    label: str
    confidence: float
    bbox: BBox
    source: str = "original"


@dataclass
class ResidualZone:
    """A connected component of the residual mask (uncovered area)."""

    bbox: BBox
    mask: np.ndarray
    ink_density: float = 0.0
    entropy: float = 0.0
    ghost_text: str = ""

    @property
    def ghost_text_length(self) -> int:
        return len(self.ghost_text)

    @property
    def area_pixels(self) -> int:
        return int(np.count_nonzero(self.mask))


@dataclass
class PageCCIResult:
    """CCI residual-vector result for one page."""

    page_number: int
    rho_ghost: float
    rho_entropy: float
    psi_residual: float
    num_residual_zones: int
    total_residual_area_ratio: float
    zones: list[ResidualZone] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "rho_ghost": round(self.rho_ghost, 6),
            "rho_entropy": round(self.rho_entropy, 6),
            "psi_residual": round(self.psi_residual, 6),
            "num_residual_zones": self.num_residual_zones,
            "total_residual_area_ratio": round(self.total_residual_area_ratio, 6),
            "zones": [
                {
                    "bbox": {
                        "x_min": z.bbox.x_min,
                        "y_min": z.bbox.y_min,
                        "x_max": z.bbox.x_max,
                        "y_max": z.bbox.y_max,
                    },
                    "ink_density": round(z.ink_density, 6),
                    "entropy": round(z.entropy, 6),
                    "ghost_text": z.ghost_text,
                    "ghost_text_length": z.ghost_text_length,
                    "area_pixels": z.area_pixels,
                }
                for z in self.zones
            ],
        }
