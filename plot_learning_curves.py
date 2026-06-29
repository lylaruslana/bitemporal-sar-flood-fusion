import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

N_FOLDS  = 5
N_EPOCHS = 50

ABLATION_MODELS = {
    'UNet Baseline\n(during-only)': {
        'path': 'Models/unet_efficientnetb3_Indonesia_baseline',
        'color': '#e74c3c', 'ls': '--',
    },
    'UNet Variant B\n(Siamese diff)': {
        'path': 'Models/unet_efficientnetb3_Indonesia_siamese',
        'color': '#3498db', 'ls': '-.',
    },
    'SegFormer Baseline\n(during-only)': {
        'path': 'Models/segformer_mitb2_Indonesia_baseline',
        'color': '#e67e22', 'ls': '--',
    },
    'SegFormer Variant A\n(early fusion)': {
        'path': 'Models/segformer_mitb2_Indonesia_bitemporal',
        'color': '#9b59b6', 'ls': '-',
    },
    'UNet Variant A\n(early fusion)': {
        'path': 'Models/unet_efficientnetb3_Indonesia_bitemporal',
        'color': '#2ecc71', 'ls': '-',
    },
}

BACKBONE_MODELS = {
    'ResNet-34': {
        'path': 'Models/unet_resnet34_Indonesia_bitemporal',
        'color': '#e74c3c', 'ls': '-.',
    },
    'ResNet-50': {
        'path': 'Models/unet_resnet50_Indonesia_bitemporal',
        'color': '#e67e22', 'ls': '--',
    },
    'EfficientNet-B1': {
        'path': 'Models/unet_efficientnetb1_Indonesia_bitemporal',
        'color': '#2ecc71', 'ls': '-',
    },
    'EfficientNet-B2': {
        'path': 'Models/unet_efficientnetb2_Indonesia_bitemporal',
        'color': '#3498db', 'ls': '-',
    },
    'EfficientNet-B3': {
        'path': 'Models/unet_efficientnetb3_Indonesia_bitemporal',
        'color': '#9b59b6', 'ls': '-',
    },
}


def load_curves(model_path):
    all_iou, all_loss = [], []
    for fold in range(1, N_FOLDS + 1):
        fp = f'{model_path}/fold{fold}/training_curves.npy'
        if not os.path.exists(fp):
            continue
        curves = np.load(fp, allow_pickle=True).item()
        all_iou.append(curves['IoU']['eval'])
        all_loss.append(curves['Loss']['eval'])
    return np.array(all_iou), np.array(all_loss)


def plot_group(models_dict, title, out_path, ylim_iou=(0, 1.0)):
    models = {k: v for k, v in models_dict.items()
              if os.path.exists(f"{v['path']}/fold1/training_curves.npy")}
    if not models:
        print(f"  Tidak ada model ditemukan untuk {title}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    epochs = np.arange(1, N_EPOCHS + 1)

    for label, cfg in models.items():
        iou_arr, loss_arr = load_curves(cfg['path'])
        if len(iou_arr) == 0:
            continue

        iou_mean  = iou_arr.mean(0)
        iou_std   = iou_arr.std(0)
        loss_mean = loss_arr.mean(0)
        loss_std  = loss_arr.std(0)
        short     = label.replace('\n', ' ')

        axes[0].plot(epochs, iou_mean, color=cfg['color'], ls=cfg['ls'],
                     lw=2, label=f'{short} ({iou_mean.max():.4f})')
        axes[0].fill_between(epochs, iou_mean - iou_std, iou_mean + iou_std,
                              color=cfg['color'], alpha=0.12)

        axes[1].plot(epochs, loss_mean, color=cfg['color'], ls=cfg['ls'],
                     lw=2, label=short)
        axes[1].fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std,
                              color=cfg['color'], alpha=0.12)

    for ax in axes:
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_xlim(1, N_EPOCHS)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(fontsize=9, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_title('Validation Flood IoU', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Flood-class IoU (class 1)', fontsize=11)
    axes[0].set_ylim(*ylim_iou)
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    axes[1].set_title('Validation Loss (DiceCE)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=11)

    plt.tight_layout()
    os.makedirs('Figures', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Tersimpan: {out_path}")


def print_summary(models_dict, title):
    print(f"\n=== {title} ===")
    for label, cfg in models_dict.items():
        iou_arr, _ = load_curves(cfg['path'])
        if len(iou_arr) == 0:
            continue
        best = iou_arr.max(axis=1)
        short = label.replace('\n', ' ')
        print(f"  {short:<35} {best.mean():.4f} ± {best.std():.4f} "
              f"| {' '.join(f'{x:.4f}' for x in best)}")


if __name__ == '__main__':
    plot_group(
        ABLATION_MODELS,
        title='Learning Curves — Ablation Study (5-Fold CV)',
        out_path='Figures/learning_curves_ablation.png',
        ylim_iou=(0, 1.0),
    )

    plot_group(
        BACKBONE_MODELS,
        title='Learning Curves — Backbone Variation in Early Fusion Model (5-Fold CV)',
        out_path='Figures/learning_curves_backbone.png',
        ylim_iou=(0.5, 1.0),
    )

    print_summary(ABLATION_MODELS, 'Ablation Study')
    print_summary(BACKBONE_MODELS, 'Backbone Sensitivity')
