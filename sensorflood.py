import torch
import torch.nn as nn
import random
import numpy as np

from pathlib import Path
import sys
sys.path.insert(1, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import time

from sklearn.model_selection import KFold
from SenForFloodMini import SenForFloodMini
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 2026

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
# ─────────────────────────────────────────────────────────────────────────────

# Diagnostics
print("=== System Check ===")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
print("MPS available:", torch.backends.mps.is_available())
print("====================\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print('Device:', DEVICE)

class DiceCELoss(nn.Module):
    def __init__(self, ce_weight=None):
        super().__init__()
        self.dice = DiceLoss(mode='multiclass', from_logits=True)
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
    
    def forward(self, logits, targets):
        return 0.5 * self.dice(logits, targets) + 0.5 * self.ce(logits, targets)

def train(model_id):
    senforfloodmini = SenForFloodMini(
        '../../SenForFlood/DFO',
        countries=['Indonesia'],
        data_to_include=['s1_before_flood', 's1_during_flood', 'terrain', 'flood_mask'],
        percentile_scale_bttm=5,
        percentile_scale_top=95,
        chip_size=256,
        use_data_augmentation=True
    )
    print(f'Total Chips: {len(senforfloodmini)}')

    max_epoch = 50
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    indices = list(range(len(senforfloodmini)))
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f'\n{"="*40}')
        print(f'Fold {fold+1}/5 | Train: {len(train_idx)} | Val: {len(val_idx)}')
        print(f'{"="*40}')

        # ── Model (reinit every fold) ─────────────────────────────────────
        # torch.manual_seed(SEED)
        # torch.cuda.manual_seed(SEED)
        # model = smp.Segformer(
        #     encoder_name="mit_b2",
        #     encoder_weights=None,
        #     in_channels=5,
        #     classes=3
        # ).to(DEVICE)
        
        model = smp.Unet(
            encoder_name="efficientnet-b3",
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
            in_channels=8,
            classes=3
        ).to(DEVICE)
        
        # model = smp.Unet(
        #     encoder_name="resnext101_32x4d",
        #     encoder_weights="ssl",
        #     encoder_depth=5,
        #     decoder_channels=(256, 128, 64, 32, 16),
        #     in_channels=5,
        #     classes=3
        # ).to(DEVICE)
        
        # model = smp.UnetPlusPlus(
        #     encoder_name="resnet34",
        #     # encoder_weights="imagenet",
        #     encoder_depth=5,
        #     decoder_channels=(256, 128, 64, 32, 16),
        #     in_channels=5,
        #     classes=3
        # ).to(DEVICE)

        loss_fn = DiceCELoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

        training_curves = {'Loss': {'train': [], 'eval': []},
                           'IoU':  {'train': [], 'eval': []}}

        # ── DataLoaders ───────────────────────────────────────────────────
        generator = torch.Generator().manual_seed(SEED + fold)

        train_set = torch.utils.data.Subset(senforfloodmini, train_idx)
        val_set   = torch.utils.data.Subset(senforfloodmini, val_idx)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=64, shuffle=True, drop_last=True,
            num_workers=4, pin_memory=True, prefetch_factor=2,
            worker_init_fn=seed_worker, generator=generator
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=32, shuffle=False, drop_last=False,
            num_workers=4, pin_memory=True, prefetch_factor=2,
            worker_init_fn=seed_worker
        )

        # ── Folder ────────────────────────────────────────────────────────
        model_folder = f'Models/{model_id}/fold{fold+1}'
        os.makedirs(model_folder, exist_ok=True)

        best_fold_iou = 0
        step = 0

        for epoch in range(max_epoch):

            # ── Train ─────────────────────────────────────────────────────
            model.train()
            for ind, (s1a, s1b, te, fm) in enumerate(train_loader):
                input_train = torch.cat([s1a[:, :3, :, :], s1b[:, :3, :, :], te], dim=1).float().to(DEVICE)
                output_train = fm.squeeze(1).to(torch.int64).to(DEVICE)

                output_ = model(input_train)
                loss = loss_fn(output_, output_train)
                model.zero_grad()
                loss.backward()
                opt.step()

                preds = output_.argmax(1)
                tp, fp, fn, tn = smp.metrics.get_stats(
                    preds, output_train,
                    mode='multiclass',
                    num_classes=3
                )
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='none')
                train_iou = iou[:, 1].mean().item()

                training_curves['Loss']['train'].append(loss.item())
                training_curves['IoU']['train'].append(train_iou)

                print(f'Fold {fold+1} | Epoch {epoch+1} | Step {step} | '
                      f'train_loss: {loss.item():.5f} | train_iou: {train_iou:.5f}', end='\r')
                step += 1

            # ── Validate ──────────────────────────────────────────────────
            model.eval()
            val_ious = []
            val_losses = []
            for s1a, s1b, te, fm in val_loader:
                input_eval = torch.cat([s1a[:, :3, :, :], s1b[:, :3, :, :], te], dim=1).float().to(DEVICE)
                output_eval = fm.squeeze(1).to(torch.int64).to(DEVICE)

                with torch.no_grad():
                    output_ = model(input_eval)
                    loss_eval = loss_fn(output_, output_eval)
                    preds = output_.argmax(1)

                    tp, fp, fn, tn = smp.metrics.get_stats(
                        preds, output_eval,
                        mode='multiclass',
                        num_classes=3
                    )
                    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='none')
                    val_ious.append(iou[:, 1].mean().item())
                    val_losses.append(loss_eval.item())

            epoch_iou = np.mean(val_ious)
            epoch_val_loss = np.mean(val_losses)
            training_curves['Loss']['eval'].append(epoch_val_loss)
            training_curves['IoU']['eval'].append(epoch_iou)

            print(f'\nFold {fold+1} | Epoch {epoch+1} | '
                  f'flood_iou: {epoch_iou:.5f} | '
                  f'val_loss: {epoch_val_loss:.5f}')

            # ── Save best ─────────────────────────────────────────────────
            if epoch_iou > best_fold_iou:
                best_fold_iou = epoch_iou
                torch.save(model.state_dict(), f'{model_folder}/best_model.pt')
                print(f'  → Best fold model saved (flood_iou: {best_fold_iou:.4f})')

        # ── Save training curves ───────────────────────────────────────────
        np.save(f'{model_folder}/training_curves.npy', training_curves)
        fold_results.append(best_fold_iou)
        print(f'\nFold {fold+1} best IoU: {best_fold_iou:.4f}')

    # ── Final Result ──────────────────────────────────────────────────────────
    print(f'\n{"="*40}')
    print(f'5-Fold CV Results:')
    for i, iou in enumerate(fold_results):
        print(f'  Fold {i+1}: {iou:.4f}')
    print(f'  Mean IoU: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}')
    print(f'{"="*40}')

    np.save(f'Models/{model_id}/fold_results.npy', fold_results)

if __name__ == '__main__':
    print("Starting training...")
    train(model_id='unet_efficientnetb3_Indonesia_bitemporal')