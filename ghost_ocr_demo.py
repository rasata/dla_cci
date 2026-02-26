#!/usr/bin/env python3
# Copyright 2026 Gabriel Zo-Hasina Rasatavohary / ZONOVA Research.
# All rights reserved.
#
# This source code is licensed under the ZONOVA Research Non-Commercial
# Research License v1.0 found in the LICENSE file in the root directory
# of this source tree. Commercial use requires written authorization.
# Contact: zo@research.zonova.io

"""
Ghost OCR — Proof-of-Concept Demonstration
===========================================

Reference implementation accompanying the position paper:

    Rasatavohary, G. Z.-H. (2026). "Towards a Metrology of Exhaustiveness
    in Document Analysis: A Systemic Framework for Layout Completeness
    Assessment."  ZONOVA Research / MatrixAI Programme.

This script demonstrates the Ghost OCR technique (Algorithm 1 in the paper):
    1. Load a document page image and its layout bounding boxes.
    2. Build the conservative residual mask (Ω_res^{+δ}).
    3. Run Tesseract OCR on the residual space only.
    4. Compute ρ_ghost, ρ_entropy, and the residual vector ψ_R.
    5. Output a report and an annotated visualization.

Usage
-----
    # With a layout JSON file
    python ghost_ocr_demo.py --image page.png --layout layout.json

    # With bounding boxes on the command line
    python ghost_ocr_demo.py --image page.png \\
        --bboxes "50,80,400,120;50,130,400,600"

    # Full options
    python ghost_ocr_demo.py --image page.png --layout layout.json \\
        --delta 10 --lang eng --output results/ -v

Layout JSON format
------------------
    {
      "elements": [
        {"bbox": [x_min, y_min, x_max, y_max], "label": "Text"},
        {"bbox": [x_min, y_min, x_max, y_max], "label": "Figure"},
        ...
      ]
    }

Dependencies: pytesseract, Pillow, numpy, opencv-python  (see requirements.txt)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.cci_engine import compute_cci_from_image
from src.residual import compute_residual_mask
from src.visualization import save_ghost_ocr_visualization

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Input parsing helpers
# -----------------------------------------------------------------------

def parse_bboxes_from_json(path: Path) -> list[tuple[int, int, int, int]]:
    """Load bounding boxes from a JSON file.

    Supports two common formats:
    - ``{"elements": [{"bbox": [x1, y1, x2, y2]}, ...]}``
    - ``{"annotations": [{"bbox": [x1, y1, x2, y2]}, ...]}``
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    bboxes: list[tuple[int, int, int, int]] = []
    for elem in data.get("elements", data.get("annotations", [])):
        if "bbox" in elem:
            b = elem["bbox"]
            if len(b) == 4:
                bboxes.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
    return bboxes


def parse_bboxes_from_string(s: str) -> list[tuple[int, int, int, int]]:
    """Parse ``"x1,y1,x2,y2;x1,y1,x2,y2;..."`` into a list of boxes."""
    bboxes: list[tuple[int, int, int, int]] = []
    for part in s.split(";"):
        coords = part.strip().split(",")
        if len(coords) == 4:
            bboxes.append(tuple(int(c.strip()) for c in coords))
    return bboxes


# -----------------------------------------------------------------------
# Omission simulation (for controlled experiments)
# -----------------------------------------------------------------------

def simulate_omissions(
    bboxes: list[tuple[int, int, int, int]],
    remove_indices: list[int],
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Remove specific bboxes to create controlled omissions.

    Args:
        bboxes: Full set of ground-truth bounding boxes.
        remove_indices: Indices of boxes to remove (0-based).

    Returns:
        ``(degraded_bboxes, removed_bboxes)``
    """
    removed = [bboxes[i] for i in remove_indices if i < len(bboxes)]
    kept = [b for i, b in enumerate(bboxes) if i not in set(remove_indices)]
    return kept, removed


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ghost-ocr-demo",
        description=(
            "Ghost OCR proof of concept — detect text in residual "
            "(uncovered) zones of a document layout."
        ),
    )
    parser.add_argument("--image", type=Path, required=True,
                        help="Page image (PNG, JPG, TIFF).")
    parser.add_argument("--layout", type=Path, default=None,
                        help="Layout JSON (see docstring for format).")
    parser.add_argument("--bboxes", type=str, default=None,
                        help='Bounding boxes: "x1,y1,x2,y2;..."')
    parser.add_argument("--delta", type=int, default=10,
                        help="Geometric tolerance δ in pixels (default: 10).")
    parser.add_argument("--lang", type=str, default="eng",
                        help="Tesseract language (default: eng).")
    parser.add_argument("--output", type=Path, default=Path("results"),
                        help="Output directory (default: results/).")
    parser.add_argument("--remove", type=str, default=None,
                        help="Simulate omissions: comma-separated bbox indices to remove.")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load image
    if not args.image.exists():
        logger.error("Image not found: %s", args.image)
        sys.exit(1)
    image = np.array(Image.open(args.image).convert("RGB"))
    h, w = image.shape[:2]
    logger.info("Image loaded: %s (%d × %d)", args.image.name, w, h)

    # Load bounding boxes
    if args.layout:
        bboxes = parse_bboxes_from_json(args.layout)
    elif args.bboxes:
        bboxes = parse_bboxes_from_string(args.bboxes)
    else:
        logger.error("Provide either --layout or --bboxes.")
        sys.exit(1)
    logger.info("Loaded %d bounding boxes", len(bboxes))

    # Optional: simulate omissions
    removed_bboxes: list[tuple[int, int, int, int]] = []
    if args.remove:
        indices = [int(i.strip()) for i in args.remove.split(",")]
        bboxes, removed_bboxes = simulate_omissions(bboxes, indices)
        logger.info(
            "Simulated omissions: removed %d boxes (indices: %s)",
            len(removed_bboxes), indices,
        )

    # Run CCI pipeline
    result = compute_cci_from_image(
        page_image=image,
        bboxes=bboxes,
        delta=args.delta,
        ocr_lang=args.lang,
    )

    # Save report
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "source_image": str(args.image),
        "image_size": [w, h],
        "num_bboxes": len(bboxes),
        "num_removed": len(removed_bboxes),
        "delta_px": args.delta,
        "residual_area_ratio": round(result.total_residual_area_ratio, 4),
        "rho_ghost": round(result.rho_ghost, 4),
        "rho_entropy": round(result.rho_entropy, 4),
        "psi_residual": round(result.psi_residual, 4),
        "num_residual_zones": result.num_residual_zones,
        "ghost_text": result.zones[0].ghost_text if result.zones else "",
    }
    report_path = args.output / "ghost_ocr_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Save visualization
    residual_mask = compute_residual_mask(h, w, bboxes, delta=args.delta)
    vis_path = args.output / f"{args.image.stem}_ghost_ocr.png"
    save_ghost_ocr_visualization(
        image, residual_mask, bboxes, vis_path,
        ghost_text=report["ghost_text"],
    )

    # Console output
    print()
    print("=" * 62)
    print("  GHOST OCR — Proof of Concept Results")
    print("=" * 62)
    print(f"  Image           : {args.image.name} ({w} × {h})")
    print(f"  Bounding boxes  : {len(bboxes)} (+ {len(removed_bboxes)} removed)")
    print(f"  Delta (δ)       : {args.delta} px")
    print(f"  Residual area   : {result.total_residual_area_ratio * 100:.1f}%")
    print("-" * 62)
    print(f"  ρ_ghost         : {result.rho_ghost:.4f}")
    print(f"  ρ_entropy       : {result.rho_entropy:.4f}")
    print(f"  ψ_R             : {result.psi_residual:.4f}")
    print("-" * 62)

    ghost_text = report["ghost_text"]
    if ghost_text:
        chars = len(ghost_text.replace("\n", "").replace(" ", ""))
        print(f"  Ghost text found ({chars} chars) — omission evidence:")
        for line in ghost_text.splitlines()[:10]:
            print(f"    > {line}")
        if ghost_text.count("\n") > 10:
            print(f"    ... ({ghost_text.count(chr(10)) - 10} more lines)")
    else:
        print("  No ghost text detected — layout appears complete.")

    print("-" * 62)
    print(f"  Report : {report_path}")
    print(f"  Visual : {vis_path}")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
