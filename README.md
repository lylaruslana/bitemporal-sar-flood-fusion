# Bi-Temporal Sentinel-1 Early Fusion for SAR Flood Mapping

Code for the paper:
**"Bi-Temporal Sentinel-1 Early Fusion for Improved SAR Flood Mapping: Extending a Single-Temporal Encoder Benchmark"**
IC3INA 2026 — Lyla Ruslana Aini et al., BRIN

---

## Overview

Three U-Net variants evaluated on the Indonesian subset of SenForFlood (9 DFO events, 1,372 tiles 256×256):

| Variant | Description | Input channels |
|---|---|---|
| **Baseline** | Single-temporal U-Net (during-event only) | 5 (VV, VH, VV/VH, DEM, slope) |
| **Variant A** | Bi-temporal early fusion (pre + during concatenated) | 8 (3 pre + 3 during + DEM + slope) |
| **Variant B** | Siamese U-Net with feature-level difference fusion | 3 + 3 + 2 |

Encoder: EfficientNet-B3 (fixed) or any `segmentation_models_pytorch` encoder via `--encoder` argument.
Metric: flood-class IoU (class 1), 5-fold cross-validation, SEED=2026.

---

## Dataset

Download the SenForFlood dataset from HuggingFace and filter to the Indonesia subset (9 DFO flood events):

```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="matosak/SenForFlood",
    repo_type="dataset",
    local_dir="~/SenForFlood",
    allow_patterns="DFO/Indonesia/**"
)
EOF
```

The scripts expect data at `~/SenForFlood/DFO/` with subdirectories per event (e.g. `DFO_1688_From_20160122/`).
Each tile is a `.npz` file containing keys: `s1_before_flood`, `s1_during_flood`, `terrain`, `flood_mask`.

---

## Requirements

```bash
pip install torch torchvision segmentation-models-pytorch scikit-learn tqdm matplotlib
```

---

## Training Scripts

### Baseline — Single-Temporal (EfficientNet-B3)

```bash
python sensorflood_baseline.py
```

Trains U-Net + EfficientNet-B3 on 5-channel during-event input. Saves checkpoints to `Models/unet_efficientnetb3_Indonesia_baseline/`.

---

### Variant A — Bi-Temporal Early Fusion

```bash
# Reproduce paper result (EfficientNet-B3)
python sensorflood_earlyfusion.py \
    --encoder efficientnet-b3 \
    --model-id unet_efficientnetb3_Indonesia_bitemporal

# Other encoders used in backbone sensitivity analysis
python sensorflood_earlyfusion.py --encoder resnet34      --model-id unet_resnet34_Indonesia_bitemporal
python sensorflood_earlyfusion.py --encoder resnet50      --model-id unet_resnet50_Indonesia_bitemporal
python sensorflood_earlyfusion.py --encoder efficientnet-b1 --model-id unet_efficientnetb1_Indonesia_bitemporal
python sensorflood_earlyfusion.py --encoder efficientnet-b2 --model-id unet_efficientnetb2_Indonesia_bitemporal
```

| Argument | Description |
|---|---|
| `--encoder` | SMP encoder name (e.g. `resnet34`, `efficientnet-b3`) |
| `--model-id` | Output folder name under `Models/` |

---

### Variant B — Siamese U-Net (EfficientNet-B3)

```bash
python sensorflood_siamese.py
```

Trains Siamese U-Net + EfficientNet-B3 with `|F_dur − F_pre|` feature difference at each encoder level. Saves to `Models/unet_efficientnetb3_Indonesia_siamese/`.

---

## Utility Scripts

### `cv_splits.py` — Cross-Validation Splits

Three CV split strategies:

| Strategy | Description |
|---|---|
| `TILE` | Tile-level split (default; may have spatial leakage within chips) |
| `CHIP` | Chip-level split — no chip spans both train and val |
| `EVENT` | Event-level split — strictest geographic generalization |

```python
from cv_splits import make_folds, print_split_report

folds = make_folds(dataset, strategy="CHIP", n_splits=5, seed=2026)
print_split_report(dataset, folds, strategy="CHIP")
```

---

### `threshold_baseline.py` — ΔVV Threshold Baseline

Non-ML baseline using VV backscatter change (pre vs. during):

```bash
python threshold_baseline.py
```

Sweeps thresholds on the training split, reports best IoU on validation split.

---

### `plot_learning_curves.py` — Learning Curves

```bash
python plot_learning_curves.py
```

Reads saved checkpoints from `Models/` and writes `Figures/learning_curves_backbone.png` and `Figures/learning_curves_ablation.png`.

---

### `visualize_results.py` — Qualitative Comparison

```bash
python visualize_results.py
```

Generates `Figures/qualitative_comparison.png` — side-by-side SAR chips with ground truth and model predictions.

---

## Results

5-fold cross-validation flood-class IoU, Indonesian subset, SEED=2026:

| Model | Backbone | Mean IoU | Std |
|---|---|---|---|
| Baseline | EfficientNet-B3 | 0.5277 | ±0.0087 |
| Variant A (Early Fusion) | EfficientNet-B1 | **0.9161** | ±0.0079 |
| Variant A (Early Fusion) | EfficientNet-B3 | 0.9104 | ±0.0087 |
| Variant B (Siamese) | EfficientNet-B3 | 0.6200 | ±0.0200 |

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
