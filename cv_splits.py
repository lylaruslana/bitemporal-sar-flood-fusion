"""
CV split strategies for SenForFloodMini tile-level datasets.

Background
----------
SenForFloodMini expands each 512x512 chip into 4 non-overlapping 256x256 tiles
before shuffling (SenForFloodMini.py line 29-33). The original training scripts
apply KFold directly to the shuffled tile indices, so tiles from the same chip
can land in both train and val — a spatial leakage bug.

Three strategies are provided:

  TILE    — original tile-level split (LEAKY, kept for comparison only)
  CHIP    — chip-level split (primary fix, no chip crosses train/val)
  EVENT   — event-level split (strictest, tests geographic generalization)

Usage
-----
    from cv_splits import make_folds, print_split_report

    folds = make_folds(dataset, strategy="CHIP", n_splits=5, seed=2026)
    print_split_report(dataset, folds, strategy="CHIP")

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_subset = Subset(dataset, train_idx)
        val_subset   = Subset(dataset, val_idx)
"""

import os
import re
from collections import defaultdict
from sklearn.model_selection import KFold, GroupKFold


def _chip_key(sample_id):
    """Unique chip identifier: the flood_mask .tif path (without tile index)."""
    return sample_id[0]


def _event_key(sample_id):
    """DFO event extracted from the path, e.g. 'DFO_4330_Indonesia'."""
    parts = sample_id[0].replace("\\", "/").split("/")
    for part in parts:
        if re.match(r"DFO_\d+", part):
            return part
    raise ValueError(f"Cannot extract event from path: {sample_id[0]}")


def make_folds(dataset, strategy="CHIP", n_splits=5, seed=2026):
    """
    Return a list of (train_indices, val_indices) tuples.

    Parameters
    ----------
    dataset   : SenForFloodMini instance (already constructed)
    strategy  : "TILE" | "CHIP" | "EVENT"
    n_splits  : number of CV folds (ignored for EVENT when n_splits > n_events)
    seed      : random seed (used for TILE and CHIP strategies)
    """
    samples = dataset.samples_ids
    n = len(samples)
    indices = list(range(n))

    if strategy == "TILE":
        # --- Strategy C: original tile-level split (LEAKY) ---
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(indices))

    elif strategy == "CHIP":
        # --- Strategy A: chip-level split (primary fix) ---
        # Group tile indices by chip path
        chip_to_tiles = defaultdict(list)
        for idx, sid in enumerate(samples):
            chip_to_tiles[_chip_key(sid)].append(idx)

        chip_ids = list(chip_to_tiles.keys())
        chip_kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        folds = []
        for train_chips, val_chips in chip_kf.split(chip_ids):
            train_idx = []
            val_idx = []
            for ci in train_chips:
                train_idx.extend(chip_to_tiles[chip_ids[ci]])
            for ci in val_chips:
                val_idx.extend(chip_to_tiles[chip_ids[ci]])
            folds.append((train_idx, val_idx))
        return folds

    elif strategy == "EVENT":
        # --- Strategy B: event-level split (strictest) ---
        event_to_tiles = defaultdict(list)
        for idx, sid in enumerate(samples):
            event_to_tiles[_event_key(sid)].append(idx)

        events = list(event_to_tiles.keys())
        n_events = len(events)
        actual_splits = min(n_splits, n_events)
        if actual_splits < n_splits:
            print(f"[cv_splits] EVENT strategy: only {n_events} events available, "
                  f"using {actual_splits}-fold instead of {n_splits}-fold.")

        # Build group labels for GroupKFold
        groups = []
        for idx, sid in enumerate(samples):
            groups.append(_event_key(sid))

        gkf = GroupKFold(n_splits=actual_splits)
        return list(gkf.split(indices, groups=groups))

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose TILE, CHIP, or EVENT.")


def print_split_report(dataset, folds, strategy):
    """Print a verification summary before training starts."""
    samples = dataset.samples_ids
    n = len(samples)
    n_splits = len(folds)

    # Unique chips and events in the full dataset
    all_chips  = set(_chip_key(s)  for s in samples)
    all_events = set(_event_key(s) for s in samples)

    print("\n" + "=" * 60)
    print(f"  CV SPLIT REPORT — strategy: {strategy}")
    print("=" * 60)
    print(f"  Total tiles  : {n}")
    print(f"  Unique chips : {len(all_chips)}  (expected ~{n // 4})")
    print(f"  Unique events: {len(all_events)}")
    print(f"  Folds        : {n_splits}")
    print("-" * 60)

    leakage_detected = False
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_set = set(train_idx)
        val_set   = set(val_idx)

        # Check for index overlap
        idx_overlap = train_set & val_set
        assert not idx_overlap, f"Fold {fold_idx+1}: tile index appears in both train and val!"

        # Check chip leakage
        train_chips = set(_chip_key(samples[i]) for i in train_idx)
        val_chips   = set(_chip_key(samples[i]) for i in val_idx)
        chip_leak   = train_chips & val_chips

        # Check event leakage
        train_events = set(_event_key(samples[i]) for i in train_idx)
        val_events   = set(_event_key(samples[i]) for i in val_idx)
        event_leak   = train_events & val_events

        leak_str = ""
        if chip_leak:
            leak_str += f"  ⚠ {len(chip_leak)} chips shared between train/val"
            leakage_detected = True
        if event_leak:
            leak_str += f"  ⚠ {len(event_leak)} events shared: {sorted(event_leak)}"

        print(f"  Fold {fold_idx+1}: train={len(train_idx):4d} tiles "
              f"({len(train_chips):3d} chips, {len(train_events)} events) | "
              f"val={len(val_idx):4d} tiles "
              f"({len(val_chips):3d} chips, {len(val_events)} events)"
              + (leak_str if leak_str else "  ✓ no chip leakage"))

    print("-" * 60)
    if leakage_detected:
        print("  *** LEAKAGE DETECTED — use CHIP or EVENT strategy for clean CV ***")
    else:
        print("  ✓ No chip leakage across any fold.")
    print("=" * 60 + "\n")
