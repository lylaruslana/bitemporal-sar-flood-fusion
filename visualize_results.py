import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import KFold
from SenForFloodMini import SenForFloodMini
import segmentation_models_pytorch as smp

SEED      = 2026
DEVICE    = torch.device('cpu')
BEST_FOLD = 2
N_SHOW    = 4

CLASS_COLORS = np.array([
    [0.15, 0.15, 0.15],  # 0 background → gelap
    [1.00, 1.00, 1.00],  # 1 flood      → putih
    [0.55, 0.55, 0.55],  # 2 perm. water → abu
])

# ── Model loaders ─────────────────────────────────────────────────────────────

def load_unet(in_channels, weights_path):
    model = smp.Unet(
        encoder_name="efficientnet-b3", encoder_depth=5,
        decoder_channels=(256, 128, 64, 32, 16),
        in_channels=in_channels, classes=3,
    )
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=False))
    model.eval()
    return model

def load_segformer(in_channels, weights_path):
    model = smp.Segformer(
        encoder_name="mit_b2", encoder_weights=None,
        in_channels=in_channels, classes=3,
    )
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=False))
    model.eval()
    return model

def load_siamese(weights_path):
    sys.path.insert(0, str(Path(__file__).parent))
    from sensorflood_siamese import SiameseChangeNet
    model = SiameseChangeNet()
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=False))
    model.eval()
    return model

# ── Cek model yang tersedia ───────────────────────────────────────────────────

baseline_path   = f'Models/unet_efficientnetb3_Indonesia_baseline/fold{BEST_FOLD}/best_model.pt'
variant_a_path  = f'Models/unet_efficientnetb3_Indonesia_bitemporal/fold{BEST_FOLD}/best_model.pt'
siamese_path    = f'Models/unet_efficientnetb3_Indonesia_siamese/fold{BEST_FOLD}/best_model.pt'
segformer_path  = f'Models/segformer_mitb2_Indonesia_bitemporal/fold{BEST_FOLD}/best_model.pt'

USE_SIAMESE    = os.path.exists(siamese_path)
USE_SEGFORMER  = os.path.exists(segformer_path)

print("Loading models...")
model_base = load_unet(5, baseline_path)
model_va   = load_unet(8, variant_a_path)
model_vb   = load_siamese(siamese_path)      if USE_SIAMESE   else None
model_sf   = load_segformer(8, segformer_path) if USE_SEGFORMER else None
print(f"  Baseline        : OK")
print(f"  Variant A       : OK")
print(f"  Variant B       : {'OK' if USE_SIAMESE   else 'tidak ditemukan'}")
print(f"  SegFormer Var A : {'OK' if USE_SEGFORMER else 'tidak ditemukan'}")

# ── Dataset ───────────────────────────────────────────────────────────────────

def get_val_set(fold_idx):
    ds = SenForFloodMini(
        '../../SenForFlood/DFO', countries=['Indonesia'],
        data_to_include=['s1_before_flood', 's1_during_flood', 'terrain', 'flood_mask'],
        percentile_scale_bttm=5, percentile_scale_top=95,
        chip_size=256, use_data_augmentation=False,
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for i, (_, val_idx) in enumerate(kf.split(range(len(ds)))):
        if i == fold_idx:
            return ds, val_idx

print("Loading dataset...")
ds, val_idx = get_val_set(BEST_FOLD - 1)
print(f"  Val chips: {len(val_idx)}")

# ── Helper ────────────────────────────────────────────────────────────────────

def to_tensor(x):
    return torch.tensor(x) if not isinstance(x, torch.Tensor) else x

def colorize(mask):
    img = np.zeros((*mask.shape, 3))
    for c, color in enumerate(CLASS_COLORS):
        img[mask == c] = color
    return img

def error_map(pred, gt):
    """TP=hijau, FN=merah, FP=biru, TN=abu gelap (flood class only)."""
    img = np.full((*pred.shape, 3), [0.15, 0.15, 0.15])
    img[(pred == 1) & (gt == 1)] = [0.10, 0.80, 0.15]   # TP hijau
    img[(pred != 1) & (gt == 1)] = [0.90, 0.10, 0.10]   # FN merah
    img[(pred == 1) & (gt != 1)] = [0.15, 0.35, 0.90]   # FP biru
    return img

def iou1(pred, gt):
    tp = ((pred == 1) & (gt == 1)).sum()
    fp = ((pred == 1) & (gt != 1)).sum()
    fn = ((pred != 1) & (gt == 1)).sum()
    return float(tp) / (tp + fp + fn + 1e-6)

def norm_sar(band):
    lo, hi = np.percentile(band, 2), np.percentile(band, 98)
    return np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)

# ── Inference & chip selection ────────────────────────────────────────────────

print("Running inference...")
candidates = []

for idx in val_idx:
    s1a, s1b, te, fm = [to_tensor(x) for x in ds[idx]]
    gt = fm.squeeze().numpy().astype(int)

    # Baseline
    inp_b = torch.cat([s1b[:3], te], dim=0).unsqueeze(0).float()
    with torch.no_grad():
        pred_base = model_base(inp_b).argmax(1).squeeze().numpy()

    # Variant A
    inp_a = torch.cat([s1a[:3], s1b[:3], te], dim=0).unsqueeze(0).float()
    with torch.no_grad():
        pred_va = model_va(inp_a).argmax(1).squeeze().numpy()

    # Variant B
    if USE_SIAMESE:
        with torch.no_grad():
            pred_vb = model_vb(
                s1a[:3].unsqueeze(0).float(),
                s1b[:3].unsqueeze(0).float(),
                te.unsqueeze(0).float(),
            ).argmax(1).squeeze().numpy()
    else:
        pred_vb = None

    # SegFormer Variant A
    if USE_SEGFORMER:
        with torch.no_grad():
            pred_sf = model_sf(inp_a).argmax(1).squeeze().numpy()
    else:
        pred_sf = None

    iou_base = iou1(pred_base, gt)
    iou_va   = iou1(pred_va,   gt)
    delta    = iou_va - iou_base

    if gt.sum() > 500 and delta > 0.2:
        candidates.append((delta, s1a, s1b, gt, pred_base, pred_va, pred_vb, pred_sf, iou_base, iou_va))

candidates.sort(reverse=True)
print(f"Kandidat (delta>0.2, flood>500px): {len(candidates)}")

if len(candidates) < N_SHOW:
    print("Threshold diturunkan ke delta>0.05...")
    for idx in val_idx:
        s1a, s1b, te, fm = [to_tensor(x) for x in ds[idx]]
        gt = fm.squeeze().numpy().astype(int)
        inp_b = torch.cat([s1b[:3], te], dim=0).unsqueeze(0).float()
        inp_a = torch.cat([s1a[:3], s1b[:3], te], dim=0).unsqueeze(0).float()
        with torch.no_grad():
            pred_base = model_base(inp_b).argmax(1).squeeze().numpy()
            pred_va   = model_va(inp_a).argmax(1).squeeze().numpy()
            pred_vb   = model_vb(s1a[:3].unsqueeze(0).float(),
                                  s1b[:3].unsqueeze(0).float(),
                                  te.unsqueeze(0).float()
                                 ).argmax(1).squeeze().numpy() if USE_SIAMESE else None
        iou_base = iou1(pred_base, gt)
        iou_va   = iou1(pred_va,   gt)
        delta    = iou_va - iou_base
        if USE_SEGFORMER:
            with torch.no_grad():
                pred_sf = model_sf(inp_a).argmax(1).squeeze().numpy()
        else:
            pred_sf = None
        if gt.sum() > 200 and delta > 0.05:
            candidates.append((delta, s1a, s1b, gt, pred_base, pred_va, pred_vb, pred_sf, iou_base, iou_va))
    candidates.sort(reverse=True)

selected = candidates[:N_SHOW]

# ── Plot ──────────────────────────────────────────────────────────────────────

col_titles = [
    'Pre-event S1\n(VV band)',
    'During-event S1\n(VV band)',
    'Ground Truth',
    'Baseline\n(during-only)',
    'Variant A\n(early fusion)',
]
if USE_SIAMESE:
    col_titles.append('Variant B\n(Siamese diff)')
if USE_SEGFORMER:
    col_titles.append('SegFormer\n(early fusion)')
N_COLS = len(col_titles)

fig, axes = plt.subplots(N_SHOW, N_COLS, figsize=(N_COLS * 3.2, N_SHOW * 3.4))

title = 'Bi-Temporal Change Detection — Qualitative Comparison'
fig.suptitle(title, fontsize=15, fontweight='bold')

for ci, ct in enumerate(col_titles):
    axes[0, ci].set_title(ct, fontsize=11, fontweight='bold')

for row, (delta, s1a, s1b, gt, pred_base, pred_va, pred_vb, pred_sf, iou_base, iou_va) in enumerate(selected):
    ax = axes[row]
    col = 0

    ax[col].imshow(norm_sar(s1a[0].numpy()), cmap='gray')
    ax[col].set_ylabel(f'Chip {row+1}', fontsize=8)
    col += 1

    ax[col].imshow(norm_sar(s1b[0].numpy()), cmap='gray'); col += 1
    ax[col].imshow(colorize(gt)); col += 1

    def annotate_iou(a, text):
        a.text(0.5, 0.03, text, transform=a.transAxes,
               ha='center', va='bottom', fontsize=11, fontweight='bold', color='white',
               bbox=dict(facecolor='#111111', alpha=0.65, pad=1.5, edgecolor='none'))

    ax[col].imshow(error_map(pred_base, gt))
    annotate_iou(ax[col], f'IoU={iou_base:.3f}')
    col += 1

    ax[col].imshow(error_map(pred_va, gt))
    annotate_iou(ax[col], f'IoU={iou_va:.3f}')
    col += 1

    if USE_SIAMESE and pred_vb is not None:
        iou_vb = iou1(pred_vb, gt)
        ax[col].imshow(error_map(pred_vb, gt))
        annotate_iou(ax[col], f'IoU={iou_vb:.3f}')
        col += 1

    if USE_SEGFORMER and pred_sf is not None:
        iou_sf = iou1(pred_sf, gt)
        ax[col].imshow(error_map(pred_sf, gt))
        annotate_iou(ax[col], f'IoU={iou_sf:.3f}')
        col += 1

    for a in ax:
        a.axis('off')

# Legend — GT uses class colors, prediction columns use error-map colors
patches_gt = [
    mpatches.Patch(color=CLASS_COLORS[0], label='Background'),
    mpatches.Patch(facecolor=CLASS_COLORS[1], edgecolor='gray', label='Flood'),
    mpatches.Patch(color=CLASS_COLORS[2], label='Permanent Water'),
]
patches_pred = [
    mpatches.Patch(color=[0.10, 0.80, 0.15], label='TP (flood correctly detected)'),
    mpatches.Patch(color=[0.90, 0.10, 0.10], label='FN (flood missed)'),
    mpatches.Patch(color=[0.15, 0.35, 0.90], label='FP (false alarm)'),
    mpatches.Patch(color=[0.15, 0.15, 0.15], label='TN (background)'),
]
divider = mpatches.Patch(color='none', label='  |  ')
all_handles = patches_gt + [divider] + patches_pred
fig.legend(handles=all_handles, loc='lower center', ncol=len(all_handles), fontsize=8,
           bbox_to_anchor=(0.5, -0.04), framealpha=0.9,
           handlelength=1.2, handletextpad=0.4, columnspacing=0.8)

fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.09, hspace=0.02, wspace=0.003)
os.makedirs('Figures', exist_ok=True)
out = 'Figures/qualitative_comparison.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nFigure tersimpan: {out}")

print("\n=== Summary ===")
header = f"{'Chip':<6} {'Baseline':>10} {'Variant A':>10} {'Δ A-Base':>10}"
if USE_SIAMESE:   header += f" {'Variant B':>10}"
if USE_SEGFORMER: header += f" {'SegFormer':>10}"
print(header)
for i, (delta, s1a, s1b, gt, pred_base, pred_va, pred_vb, pred_sf, iou_base, iou_va) in enumerate(selected):
    row_str = f"{i+1:<6} {iou_base:>10.4f} {iou_va:>10.4f} {delta:>+10.4f}"
    if USE_SIAMESE   and pred_vb is not None: row_str += f" {iou1(pred_vb, gt):>10.4f}"
    if USE_SEGFORMER and pred_sf is not None: row_str += f" {iou1(pred_sf, gt):>10.4f}"
    print(row_str)
