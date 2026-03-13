# DLA-CCI: Ghost OCR & Completeness Confidence Index

**Reference implementation** accompanying the position paper:

> Rasatavohary, G. Z.-H. (2026). *Towards a Metrology of Exhaustiveness in Document Analysis: A Systemic Framework for Layout Completeness Assessment.* ZONOVA Research / MatrixAI Programme.

---

## Overview

Current Document Layout Analysis (DLA) metrics — mAP, IoU, F1 — measure the quality of **what was detected** but remain silent on **what was missed**. This repository provides a minimal, self-contained implementation of the **Ghost OCR** technique and the **residual signal vector** (ψ_R) of the **Completeness Confidence Index (CCI)** framework proposed in the paper.

### What is Ghost OCR?

Ghost OCR is a simple yet effective technique: after a layout model has produced its detections, we **blank out all detected regions** and **run OCR on whatever remains**. Any text found in the residual space is direct evidence that the layout model missed something.

```
┌──────────────────────────────┐
│  Document Page               │
│  ┌────────────┐              │
│  │ Detected   │  ← covered  │
│  │ (blanked)  │              │
│  └────────────┘              │
│        Ghost text here! ←──── residual (OCR runs here)
│  ┌────────────┐              │
│  │ Detected   │  ← covered  │
│  └────────────┘              │
│              Missed footer ←─ residual
└──────────────────────────────┘
```

### Metrics computed

| Metric | Description | Range |
|--------|-------------|-------|
| **ρ_ghost** | Absence of ghost text in residual zones | [0, 1] — 1 = no ghost text |
| **ρ_entropy** | Low visual complexity in residual zones | [0, 1] — 1 = uniform background |
| **ψ_R** | Residual signal vector = ρ_ghost × ρ_entropy | [0, 1] — 1 = high completeness confidence |

---

## Quick Start

### Prerequisites

- **Python 3.12+**
- **Tesseract OCR** installed and on `PATH`
  ```bash
  # macOS
  brew install tesseract

  # Ubuntu/Debian
  sudo apt-get install tesseract-ocr

  # Windows — see https://github.com/tesseract-ocr/tesseract
  ```

### Installation

```bash
git clone https://github.com/rasata/dla_cci.git
cd dla_cci
pip install -r requirements.txt
```

### Basic usage

```bash
# With a layout JSON file
python ghost_ocr_demo.py \
    --image examples/my_page.png \
    --layout examples/sample_layout.json \
    --output results/

# With bounding boxes specified on the command line
python ghost_ocr_demo.py \
    --image examples/my_page.png \
    --bboxes "50,80,400,120;50,140,400,600" \
    --output results/
```

### Simulate omissions (controlled experiment)

To validate that Ghost OCR can detect known omissions, use `--remove` to deliberately drop bounding boxes:

```bash
# Remove bbox indices 2 and 4 from the layout to simulate omissions
python ghost_ocr_demo.py \
    --image examples/my_page.png \
    --layout examples/sample_layout.json \
    --remove "2,4" \
    --output results/
```

The output will show whether Ghost OCR recovers text from the removed regions.

---

## Output

The script produces two files in the output directory:

### 1. `ghost_ocr_report.json`

```json
{
  "source_image": "examples/my_page.png",
  "image_size": [612, 792],
  "num_bboxes": 5,
  "num_removed": 0,
  "delta_px": 10,
  "residual_area_ratio": 0.3241,
  "rho_ghost": 0.9812,
  "rho_entropy": 0.9654,
  "psi_residual": 0.9472,
  "num_residual_zones": 12,
  "ghost_text": ""
}
```

### 2. `<image_name>_ghost_ocr.png`

Annotated visualization:
- **Green rectangles** — detected bounding boxes
- **Red overlay** — residual zones (uncovered areas)
- Ghost text count displayed if omissions are detected

---

## Experiment Protocol

As described in Section 5 of the paper, we propose the following validation protocol applicable to any public DLA dataset (PubLayNet, DocLayNet, etc.):

1. **Select** *k* pages with known complete ground-truth annotations.
2. **Simulate omissions** by removing *m* bounding boxes per page (`--remove`).
3. **Run Ghost OCR** on the degraded layout.
4. **Measure detection rate**: what fraction of deliberately omitted text regions produce ghost text?

```bash
# Example: systematic experiment on page_042.png
# Full layout has 8 elements; remove elements 1, 3, 5 one at a time

python ghost_ocr_demo.py --image page_042.png --layout full_layout.json \
    --output results/baseline/

python ghost_ocr_demo.py --image page_042.png --layout full_layout.json \
    --remove "1" --output results/omit_1/

python ghost_ocr_demo.py --image page_042.png --layout full_layout.json \
    --remove "3" --output results/omit_3/

python ghost_ocr_demo.py --image page_042.png --layout full_layout.json \
    --remove "5" --output results/omit_5/
```

Compare `psi_residual` across runs: a significant drop when removing a text-bearing element validates the principle.

---

## Project Structure

```
dla_cci/
├── README.md               ← this file
├── LICENSE                  ← Non-Commercial Research License v1.0
├── requirements.txt         ← Python dependencies
├── .gitignore
├── ghost_ocr_demo.py        ← main entry point
├── src/
│   ├── __init__.py
│   ├── models.py            ← data structures (BBox, PageCCIResult, …)
│   ├── residual.py          ← residual mask computation
│   ├── ghost_ocr.py         ← Ghost OCR (Tesseract on residual zones)
│   ├── ink_entropy.py       ← ink density + Shannon entropy
│   ├── cci_engine.py        ← orchestrator: ρ_ghost × ρ_entropy → ψ_R
│   └── visualization.py     ← annotated image generation
├── examples/
│   └── sample_layout.json   ← example layout for testing
└── results/                 ← output directory (generated)
```

---

## Algorithm (Paper Reference)

This implementation corresponds to **Algorithm 1** in the paper (Section 5.1):

```
Input:  Page image I (H × W), detected boxes M = {R₁, …, Rₙ}, tolerance δ
Output: Residual signal score ψ_R ∈ [0, 1], ghost text T_ghost

Phase 1 — Conservative residual mask:
    For each Rᵢ, compute δ-eroded box Rᵢ⁻ᵟ
    Mask M ← page \ ∪ Rᵢ⁻ᵟ

Phase 2 — Ghost OCR:
    Blank covered regions in I
    T_ghost ← Tesseract(I_masked)

Phase 3 — Residual entropy:
    H_res ← Shannon entropy of pixel distribution in residual
    ρ_entropy ← 1 − H_res / log₂(256)

Phase 4 — Score:
    ρ_ghost ← 1 − |T_ghost| / (|T_ghost| + |T_detected|)
    ψ_R ← ρ_ghost × ρ_entropy
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--delta` | 10 | Geometric tolerance δ in pixels. Conservative erosion of bounding boxes before masking. At 200–300 DPI, 10 px ≈ 0.85–1.27 mm. |
| `--lang` | `eng` | Tesseract language code. Use `fra+eng` for French+English, etc. |
| `--remove` | — | Comma-separated bbox indices to remove (omission simulation). |
| `--output` | `results/` | Output directory for report and visualization. |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{rasatavohary2026metrology,
  title   = {Towards a Metrology of Exhaustiveness in Document Analysis:
             A Systemic Framework for Layout Completeness Assessment},
  author  = {Rasatavohary, Gabriel Zo-Hasina},
  year    = {2026},
  month   = feb,
  note    = {Position Paper. ZONOVA Research / MatrixAI Programme},
  url     = {https://github.com/rasata/dla_cci}
}
```

---

## License

Copyright 2026 Gabriel Zo-Hasina Rasatavohary / ZONOVA Research. All rights reserved.

This code is released under the **ZONOVA Research Non-Commercial Research License v1.0** — see [LICENSE](LICENSE).

| Use case | Permitted? |
|---|---|
| Academic research & education | **Yes** — free |
| Non-commercial testing & benchmarking | **Yes** — free |
| Publication of results (with citation) | **Yes** — free |
| Redistribution for research (with license) | **Yes** — free |
| **Commercial use** | **No** — requires written authorization |

For commercial licensing inquiries, contact: **zo@research.zonova.io**

## Acknowledgments

- **ZONOVA Research / MatrixAI Programme** — R&D framework for intelligent document processing.
- **Prof. Emeritus Ioan Roxin** (Université de Franche-Comté) — pre-submission peer review of the paper.

## Citation
If you use this code or the CCI framework in your research, please cite the following position paper:

**Rasatavohary, G. Z.-H. (2026). Towards a Metrology of Exhaustiveness in Document Analysis: A Systemic Framework for Layout Completeness Assessment. engrXiv. https://engrxiv.org/preprint/view/6568**
