"""
Task 3: ΔVV Threshold Baseline
================================
Non-ML baseline using only the ΔVV = VV_dur - VV_pre signal to classify
flood (class 1) and permanent water (class 2).

Logic
-----
  permanent water : vv_dur < thresh_water  AND  |delta_vv| < 3 dB  (already dark)
  flood           : delta_vv  < thresh_flood AND  vv_dur < -10 dB  (became dark)
  background      : everything else

Evaluation
----------
Uses the SAME tile-level KFold indices as the original training scripts so
results are directly comparable. Threshold sweep uses TRAINING split only;
best threshold is then applied to val.

Run
---
    cd ic3ina26/
    python threshold_baseline.py
"""

import sys, os
import numpy as np
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent))

from SenForFloodMini import SenForFloodMini
from cv_splits import make_folds, print_split_report
from sklearn.model_selection import KFold
from tqdm import tqdm

SEED = 2026

# ── Threshold sweep grid ──────────────────────────────────────────────────────
THRESH_FLOOD_GRID  = np.arange(-1.0, -6.5, -0.5)   # -1.0 to -6.0 dB
THRESH_WATER_GRID  = np.arange(-12.0, -18.5, -1.0)  # -12.0 to -18.0 dB
THRESH_DUR_FLOOD   = -10.0   # vv_dur must be this dark to be flood (fixed)
THRESH_DELTA_WATER =  3.0    # |ΔVV| < this to be permanent water (fixed)

# ── CV strategy: match original paper (TILE) for direct comparison ────────────
CV_SPLIT_STRATEGY = "TILE"


def iou_class(pred, gt, cls):
    """Flood-class IoU for a single class label."""
    tp = np.sum((pred == cls) & (gt == cls))
    fp = np.sum((pred == cls) & (gt != cls))
    fn = np.sum((pred != cls) & (gt == cls))
    denom = tp + fp + fn
    return tp / denom if denom > 0 else float('nan')


def predict(vv_pre, vv_dur, thresh_flood, thresh_water):
    """
    vv_pre, vv_dur : float arrays, raw (already in linear or dB — must be same unit).
    Returns integer array: 0=background, 1=flood, 2=permanent water.
    """
    delta = vv_dur - vv_pre
    pred = np.zeros_like(vv_dur, dtype=np.int64)

    # Permanent water: already dark before AND stays dark (small delta)
    perm = (vv_dur < thresh_water) & (np.abs(delta) < THRESH_DELTA_WATER)
    pred[perm] = 2

    # Flood: big drop AND dark during (not already water)
    flood = (delta < thresh_flood) & (vv_dur < THRESH_DUR_FLOOD) & (~perm)
    pred[flood] = 1

    return pred


def sweep_thresholds(subset_indices, dataset):
    """Find best (thresh_flood, thresh_water) on a given set of tile indices."""
    best_iou = -1
    best_tf = THRESH_FLOOD_GRID[0]
    best_tw = THRESH_WATER_GRID[0]

    for tf in THRESH_FLOOD_GRID:
        for tw in THRESH_WATER_GRID:
            ious = []
            for idx in subset_indices:
                s1a, s1b, te, fm = dataset[idx]
                vv_pre = s1a[0]   # VV channel of pre-event
                vv_dur = s1b[0]   # VV channel of during-event
                gt     = fm[0].astype(np.int64)
                pred   = predict(vv_pre, vv_dur, tf, tw)
                iou    = iou_class(pred, gt, cls=1)
                if not np.isnan(iou):
                    ious.append(iou)
            mean_iou = np.mean(ious) if ious else 0.0
            if mean_iou > best_iou:
                best_iou = mean_iou
                best_tf  = tf
                best_tw  = tw

    return best_tf, best_tw, best_iou


def evaluate_fold(val_indices, dataset, thresh_flood, thresh_water):
    """Evaluate on val split with fixed thresholds. Returns (flood_iou, water_iou)."""
    flood_ious = []
    water_ious = []
    for idx in val_indices:
        s1a, s1b, te, fm = dataset[idx]
        vv_pre = s1a[0]
        vv_dur = s1b[0]
        gt     = fm[0].astype(np.int64)
        pred   = predict(vv_pre, vv_dur, thresh_flood, thresh_water)
        fi = iou_class(pred, gt, cls=1)
        wi = iou_class(pred, gt, cls=2)
        if not np.isnan(fi):
            flood_ious.append(fi)
        if not np.isnan(wi):
            water_ious.append(wi)
    return np.mean(flood_ious), np.mean(water_ious)


def main():
    print("Loading dataset...")
    ds = SenForFloodMini(
        '../../SenForFlood/DFO',
        countries=['Indonesia'],
        data_to_include=['s1_before_flood', 's1_during_flood', 'terrain', 'flood_mask'],
        percentile_scale_bttm=5,
        percentile_scale_top=95,
        chip_size=256,
        use_data_augmentation=False,   # no augmentation for threshold baseline
    )
    print(f"Total tiles: {len(ds)}")

    folds = make_folds(ds, strategy=CV_SPLIT_STRATEGY, n_splits=5, seed=SEED)
    print_split_report(ds, folds, strategy=CV_SPLIT_STRATEGY)

    results = []
    print(f"\nThreshold sweep grid: flood={THRESH_FLOOD_GRID}, water={THRESH_WATER_GRID}")
    print(f"Fixed: vv_dur_max={THRESH_DUR_FLOOD} dB, |ΔVV|_water<{THRESH_DELTA_WATER} dB\n")

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"{'='*50}")
        print(f"Fold {fold_idx+1}/5 | sweeping on {len(train_idx)} train tiles...")

        # Use only training split for threshold sweep
        best_tf, best_tw, train_iou = sweep_thresholds(train_idx, ds)
        print(f"  Best thresh_flood={best_tf:.1f} dB, thresh_water={best_tw:.1f} dB "
              f"(train flood IoU={train_iou:.4f})")

        # Evaluate on val with best thresholds from train
        val_flood_iou, val_water_iou = evaluate_fold(val_idx, ds, best_tf, best_tw)
        print(f"  Val flood IoU={val_flood_iou:.4f}  |  Val water IoU={val_water_iou:.4f}")

        results.append({
            'fold': fold_idx + 1,
            'best_thresh_flood': best_tf,
            'best_thresh_water': best_tw,
            'val_flood_iou': val_flood_iou,
            'val_water_iou': val_water_iou,
        })

    # Summary
    fold_flood_ious = [r['val_flood_iou'] for r in results]
    fold_water_ious = [r['val_water_iou'] for r in results]
    print(f"\n{'='*50}")
    print(f"ΔVV Threshold Baseline — 5-Fold CV Summary ({CV_SPLIT_STRATEGY} split)")
    print(f"{'='*50}")
    print(f"{'Fold':<6} {'thresh_flood':>12} {'thresh_water':>12} {'flood_IoU':>10} {'water_IoU':>10}")
    for r in results:
        print(f"{r['fold']:<6} {r['best_thresh_flood']:>12.1f} {r['best_thresh_water']:>12.1f} "
              f"{r['val_flood_iou']:>10.4f} {r['val_water_iou']:>10.4f}")
    print(f"{'Mean':<6} {'':>12} {'':>12} "
          f"{np.mean(fold_flood_ious):>10.4f} {np.mean(fold_water_ious):>10.4f}")
    print(f"{'Std':<6} {'':>12} {'':>12} "
          f"{np.std(fold_flood_ious):>10.4f} {np.std(fold_water_ious):>10.4f}")

    # Save results
    import csv
    out_path = 'threshold_baseline_results.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({
            'fold': 'mean',
            'best_thresh_flood': '',
            'best_thresh_water': '',
            'val_flood_iou': np.mean(fold_flood_ious),
            'val_water_iou': np.mean(fold_water_ious),
        })
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
