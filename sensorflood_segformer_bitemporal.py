import torch
import torch.nn as nn
import random
import numpy as np
import os
import sys
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent))

from sklearn.model_selection import KFold
from SenForFloodMini import SenForFloodMini
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

SEED = 2026
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    random.seed(SEED + worker_id); np.random.seed(SEED + worker_id)

print("=== System Check ===")
print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("====================\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(mode='multiclass', from_logits=True)
        self.ce   = nn.CrossEntropyLoss()
    def forward(self, logits, targets):
        return 0.5 * self.dice(logits, targets) + 0.5 * self.ce(logits, targets)

def train(model_id):
    senforfloodmini = SenForFloodMini(
        '../../SenForFlood/DFO',
        countries=['Indonesia'],
        data_to_include=['s1_before_flood', 's1_during_flood', 'terrain', 'flood_mask'],
        percentile_scale_bttm=5, percentile_scale_top=95,
        chip_size=256, use_data_augmentation=True
    )
    print(f'Total Chips: {len(senforfloodmini)}')

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(senforfloodmini)))):
        print(f'\n{"="*40}')
        print(f'Fold {fold+1}/5 | Train: {len(train_idx)} | Val: {len(val_idx)}')
        print(f'{"="*40}')

        model = smp.Segformer(
            encoder_name="mit_b2",
            encoder_weights=None,
            in_channels=8,   # 3 pre + 3 during + 2 terrain
            classes=3,
        ).to(DEVICE)

        loss_fn = DiceCELoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
        training_curves = {'Loss': {'train': [], 'eval': []}, 'IoU': {'train': [], 'eval': []}}
        generator = torch.Generator().manual_seed(SEED + fold)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(senforfloodmini, train_idx),
            batch_size=64, shuffle=True, drop_last=True,
            num_workers=4, pin_memory=True, prefetch_factor=2,
            worker_init_fn=seed_worker, generator=generator
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(senforfloodmini, val_idx),
            batch_size=32, shuffle=False, drop_last=False,
            num_workers=4, pin_memory=True, prefetch_factor=2,
            worker_init_fn=seed_worker
        )

        model_folder = f'Models/{model_id}/fold{fold+1}'
        os.makedirs(model_folder, exist_ok=True)
        best_fold_iou = 0
        step = 0

        for epoch in range(50):
            model.train()
            for ind, (s1a, s1b, te, fm) in enumerate(train_loader):
                input_ = torch.cat([s1a[:, :3], s1b[:, :3], te], dim=1).float().to(DEVICE)
                target = fm.squeeze(1).to(torch.int64).to(DEVICE)
                out = model(input_)
                loss = loss_fn(out, target)
                model.zero_grad(); loss.backward(); opt.step()

                preds = out.argmax(1)
                tp, fp, fn, tn = smp.metrics.get_stats(preds, target, mode='multiclass', num_classes=3)
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='none')
                training_curves['Loss']['train'].append(loss.item())
                training_curves['IoU']['train'].append(iou[:, 1].mean().item())
                print(f'Fold {fold+1} | Epoch {epoch+1} | Step {step} | loss: {loss.item():.5f} | iou: {iou[:,1].mean().item():.5f}', end='\r')
                step += 1

            model.eval()
            val_ious, val_losses = [], []
            for s1a, s1b, te, fm in val_loader:
                input_ = torch.cat([s1a[:, :3], s1b[:, :3], te], dim=1).float().to(DEVICE)
                target = fm.squeeze(1).to(torch.int64).to(DEVICE)
                with torch.no_grad():
                    out = model(input_)
                    tp, fp, fn, tn = smp.metrics.get_stats(out.argmax(1), target, mode='multiclass', num_classes=3)
                    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='none')
                    val_ious.append(iou[:, 1].mean().item())
                    val_losses.append(loss_fn(out, target).item())

            epoch_iou = np.mean(val_ious)
            training_curves['Loss']['eval'].append(np.mean(val_losses))
            training_curves['IoU']['eval'].append(epoch_iou)
            print(f'\nFold {fold+1} | Epoch {epoch+1} | flood_iou: {epoch_iou:.5f} | val_loss: {np.mean(val_losses):.5f}')

            if epoch_iou > best_fold_iou:
                best_fold_iou = epoch_iou
                torch.save(model.state_dict(), f'{model_folder}/best_model.pt')
                print(f'  → Best saved (flood_iou: {best_fold_iou:.4f})')

        np.save(f'{model_folder}/training_curves.npy', training_curves)
        fold_results.append(best_fold_iou)
        print(f'\nFold {fold+1} best IoU: {best_fold_iou:.4f}')

    print(f'\n{"="*40}')
    print('5-Fold CV Results:')
    for i, iou in enumerate(fold_results):
        print(f'  Fold {i+1}: {iou:.4f}')
    print(f'  Mean IoU: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}')
    print(f'{"="*40}')
    np.save(f'Models/{model_id}/fold_results.npy', fold_results)

if __name__ == '__main__':
    print("Starting SegFormer Variant A (bi-temporal early fusion, 8ch)...")
    train(model_id='segformer_mitb2_Indonesia_bitemporal')
