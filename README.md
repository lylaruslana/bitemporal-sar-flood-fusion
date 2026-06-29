# Bi-Temporal Sentinel-1 Early Fusion for SAR Flood Mapping

Code for the paper:
**"Bi-Temporal Sentinel-1 Early Fusion for Improved SAR Flood Mapping: Extending a Single-Temporal Encoder Benchmark"**
IC3INA 2026 — Lyla Ruslana Aini et al., BRIN

---

## Overview

This repository contains training scripts for three model variants evaluated on the Indonesian subset of the [SenForFlood](https://huggingface.co/datasets/matosak/SenForFlood) dataset:

| Variant | Description | Input channels |
|---|---|---|
| **Baseline** | Single-temporal U-Net (during-event only) | 5 (VV, VH, VV/VH, DEM, slope) |
| **Variant A** | Bi-temporal early fusion (pre + during concat) | 8 (3 pre + 3 during + DEM + slope) |
| **Variant B** | Siamese U-Net with feature-level difference fusion | 3+3+2 |

All models use `segmentation_models_pytorch` with encoder depth 5, decoder channels (256,128,64,32,16), and are trained from scratch (no ImageNet pretraining). Evaluation metric: flood-class IoU (class 1), 5-fold cross-validation, SEED=2026.

---

## Requirements

```bash
pip install torch torchvision segmentation-models-pytorch scikit-learn tqdm matplotlib
```

Dataset must be downloaded to `~/SenForFlood/DFO/` using `SenForFloodMini.py` (not included; see [SenForFlood on HuggingFace](https://huggingface.co/datasets/matosak/SenForFlood)).

---

## Training Scripts

### Baseline — Single-Temporal (EfficientNet-B3, fixed)

```bash
python sensorflood_baseline.py
```

Trains U-Net + EfficientNet-B3 on 5-channel during-event input. Saves checkpoints to `Models/unet_efficientnetb3_Indonesia_baseline/`.

---

### Baseline — Generic (any encoder)

```bash
python sensorflood_baseline_generic.py \
    --encoder efficientnet-b1 \
    --model-id unet_efficientnetb1_Indonesia_baseline
```

**Arguments:**

| Argument | Description |
|---|---|
| `--encoder` | SMP encoder name (e.g. `resnet34`, `efficientnet-b1`) |
| `--model-id` | Output folder name under `Models/` |

---

### Variant A — Bi-Temporal Early Fusion (EfficientNet-B3, fixed)

```bash
python sensorflood.py
```

Trains U-Net + EfficientNet-B3 on 8-channel bi-temporal input. Saves to `Models/unet_efficientnetb3_Indonesia_bitemporal/`.

---

### Variant A — Generic Early Fusion (any encoder)

```bash
python sensorflood_earlyfusion.py \
    --encoder resnet34 \
    --model-id unet_resnet34_Indonesia_bitemporal
```

**Arguments:**

| Argument | Description |
|---|---|
| `--encoder` | SMP encoder name |
| `--model-id` | Output folder name under `Models/` |

---

### Variant B — Siamese U-Net (EfficientNet-B3, fixed)

```bash
python sensorflood_siamese.py
```

Trains Siamese U-Net + EfficientNet-B3 with `|F_dur - F_pre|` feature difference at each encoder level. Saves to `Models/unet_efficientnetb3_Indonesia_siamese/`.

---

### SegFormer Variants (Experimental)

```bash
# Baseline
python sensorflood_segformer_baseline.py

# Bi-temporal early fusion
python sensorflood_segformer_bitemporal.py
```

---

### Chip-Split Variants (Corrected CV)

These scripts use chip-level CV splitting (no spatial leakage between tiles from the same 512×512 chip):

```bash
python sensorflood_baseline_chipsplit.py
python sensorflood_earlyfusion_chipsplit.py
python sensorflood_siamese_chipsplit.py
```

---

## Utility Scripts

### Cross-Validation Splits — `cv_splits.py`

Provides three CV split strategies:

| Strategy | Description |
|---|---|
| `TILE` | Original tile-level split (may have spatial leakage) |
| `CHIP` | Chip-level split — no chip crosses train/val boundary |
| `EVENT` | Event-level split — strictest geographic generalization |

```python
from cv_splits import make_folds, print_split_report

folds = make_folds(dataset, strategy="CHIP", n_splits=5, seed=2026)
print_split_report(dataset, folds, strategy="CHIP")
```

---

### ΔVV Threshold Baseline — `threshold_baseline.py`

Non-ML rule-based baseline using VV backscatter change signal:

```bash
cd ic3ina26/
python threshold_baseline.py
```

Sweeps thresholds on training split and evaluates on validation split. Saves results to `threshold_baseline_results.csv`.

---

### Plot Learning Curves — `plot_learning_curves.py`

```bash
python plot_learning_curves.py
```

Generates `Figures/learning_curves_backbone.png` and `Figures/learning_curves_ablation.png` from saved model checkpoints.

---

### Qualitative Comparison — `visualize_results.py`

```bash
python visualize_results.py
```

Generates `Figures/qualitative_comparison.png` — side-by-side SAR chips with ground truth and predictions for all three variants.

---

## SLURM Scripts

For cluster submission (adjust `--nodelist` and paths as needed):

| Script | Purpose |
|---|---|
| `run_slurm_baseline.sh` | Submit Baseline EffB3 job |
| `run_slurm.sh` | Submit Variant A EffB3 job |
| `run_slurm_siamese.sh` | Submit Variant B Siamese EffB3 job |
| `run_slurm_backbone_ablation.sh` | Submit backbone sensitivity jobs (ResNet34/50, EffB1/B2) sequentially |
| `run_slurm_all_segformer.sh` | Submit SegFormer baseline + bi-temporal jobs |
| `run_slurm_baseline_effb2.sh` | Submit Baseline EffB2 job |

```bash
sbatch run_slurm_baseline.sh
sbatch run_slurm.sh
sbatch run_slurm_siamese.sh
```

---

## Results

5-fold cross-validation flood-class IoU on the Indonesian subset (SEED=2026):

| Model | Backbone | Mean IoU | Std |
|---|---|---|---|
| Baseline | EfficientNet-B3 | 0.5277 | ±0.0087 |
| Variant A (Early Fusion) | EfficientNet-B1 | **0.9161** | ±0.0079 |
| Variant A (Early Fusion) | EfficientNet-B3 | 0.9104 | ±0.0087 |
| Variant B (Siamese) | EfficientNet-B3 | 0.6200 | ±0.0200 |

Full per-fold results: [`results_all.csv`](results_all.csv) · [`results_pivot.csv`](results_pivot.csv)

---

## Citation

```bibtex
@inproceedings{aini2026bitemporal,
  author    = {Aini, Lyla Ruslana and others},
  title     = {Bi-Temporal Sentinel-1 Early Fusion for Improved SAR Flood Mapping:
               Extending a Single-Temporal Encoder Benchmark},
  booktitle = {Proceedings of the International Conference on Computer, Control,
               Informatics and its Applications (IC3INA)},
  year      = {2026}
}
```
