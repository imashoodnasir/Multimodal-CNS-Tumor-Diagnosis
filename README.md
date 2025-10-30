# Adaptive Vision–Language Transformer (AVLT) — CNS Tumor Diagnosis

This repository provides a reproducible PyTorch implementation of the **Adaptive Vision–Language Transformer (AVLT)**
for multimodal CNS tumor diagnosis using MRI (T1/T1ce/T2/FLAIR) and clinical/genomic text.

It includes:
- Vision encoder (CNN stem + ViT via `timm`)
- Language encoder (ClinicalBERT via `transformers`)
- Cross-attention + adaptive gating fusion
- Student–teacher self-distillation with cross-modal alignment
- Training/evaluation scripts with ablations, ROC/Confusion, and Grad-CAM
- A **synthetic data fallback** so you can run a quick sanity check without datasets

> **Note**: Replace the synthetic pipeline with your datasets (BraTS OS, TCGA-GBM/LGG, REMBRANDT, GLASS) by pointing paths in `configs/base.yaml`.

---

## Quick Start (synthetic data sanity check)

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/base.yaml --synthetic true --max_steps 50
python scripts/eval.py  --config configs/base.yaml
```

This produces logs and figures under `outputs/` (ROC, confusion, Grad-CAM).

## Real Data Usage (outline)

Organize your data as:

```
data/
  brats/
    imagesTr/  # NIfTI or PNG slices (preprocessed)
    labelsTr/  # CSV with {id, label} for OS bins or tumor type
    textTr/    # per-patient .txt or a CSV column "report"
```

Then run:

```bash
python scripts/train.py --config configs/base.yaml --data_root /path/to/data --dataset brats
python scripts/eval.py  --config configs/base.yaml --data_root /path/to/data --dataset brats
```

## Repository Layout

```
configs/base.yaml              # Main hyperparameters
requirements.txt               # Dependencies
src/avlt/models/encoders.py    # Vision & text encoders
src/avlt/models/fusion.py      # Cross-attention + adaptive gating
src/avlt/models/avlt.py        # Full AVLT model (student/teacher)
src/avlt/data/dataset.py       # Real + synthetic datasets
src/avlt/train/losses.py       # Alignment, distillation, classification
src/avlt/train/engine.py       # Train loop (AMP, EMA)
src/avlt/utils/metrics.py      # ACC, F1, AUC, C-index
src/avlt/viz/plots.py          # ROC, confusion, Grad-CAM
scripts/train.py               # CLI training entry
scripts/eval.py                # CLI evaluation entry
scripts/infer.py               # Single-case inference
```

## Citation

If you use this code, please cite the paper section in your thesis/manuscript.

---

## License
Apache-2.0
