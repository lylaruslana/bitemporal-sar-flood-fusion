import torch
import torch.nn as nn
import torch.nn.functional as F
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
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead

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

print("=== System Check ===")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
print("====================\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)


class SiameseChangeNet(nn.Module):
    """
    Variant B: Siamese encoder + absolute-difference fusion.

    Arsitektur:
    - Satu shared EfficientNet-B3 encoder (3ch SAR: VV, VH, VV/VH)
    - Encoder dijalankan dua kali: untuk pre-event dan during-event
    - Di setiap level skip connection: |feat_during - feat_pre| (absolute difference)
    - Terrain (2ch) diinjeksikan di bottleneck via adaptive pooling + concat
    - UnetDecoder memproses difference feature pyramid → segmentasi 3 kelas

    Input:
        s1_pre    : [B, 3, H, W]  — SAR pre-event (VV, VH, VV/VH)
        s1_during : [B, 3, H, W]  — SAR during-event (VV, VH, VV/VH)
        terrain   : [B, 2, H, W]  — DEM + slope

    Output:
        logits    : [B, 3, H, W]  — unnormalized scores per kelas
    """
    def __init__(self, encoder_name='efficientnet-b3', encoder_depth=5,
                 decoder_channels=(256, 128, 64, 32, 16), classes=3,
                 terrain_in=2):
        super().__init__()

        # Shared encoder — satu instance dipakai untuk pre DAN during
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=3, depth=encoder_depth, weights=None
        )
        enc_ch = list(self.encoder.out_channels)  # [3, 40, 32, 48, 136, 384]

        # Terrain masuk di bottleneck: adaptive pool ke ukuran bottleneck, lalu concat
        # enc_ch[-1] = 384 (bottleneck), setelah concat terrain: 384 + terrain_in = 386
        self.terrain_in = terrain_in
        bottleneck_ch = enc_ch[-1] + terrain_in   # 386

        # Modified encoder_channels untuk decoder
        modified_enc_ch = enc_ch[:-1] + [bottleneck_ch]  # [..., 386]

        self.decoder = UnetDecoder(
            encoder_channels=modified_enc_ch,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_norm='batchnorm',
            attention_type=None,
        )

        self.head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )

    def forward(self, s1_pre, s1_during, terrain):
        # Encode pre dan during dengan encoder yang sama (shared weights)
        feat_pre = self.encoder(s1_pre)     # list 6 feature maps
        feat_dur = self.encoder(s1_during)  # list 6 feature maps

        # Absolute difference di setiap level skip connection
        diff = [torch.abs(fd - fp) for fd, fp in zip(feat_dur, feat_pre)]

        # Inject terrain di bottleneck: pool ke spatial size bottleneck, lalu concat
        terrain_pooled = F.adaptive_avg_pool2d(terrain, output_size=diff[-1].shape[2:])
        diff[-1] = torch.cat([diff[-1], terrain_pooled], dim=1)  # [B, 386, 8, 8]

        # Decode difference feature pyramid
        x = self.decoder(diff)
        return self.head(x)


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

        model = SiameseChangeNet(
            encoder_name='efficientnet-b3',
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=3,
            terrain_in=2,
        ).to(DEVICE)

        loss_fn = DiceCELoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

        training_curves = {'Loss': {'train': [], 'eval': []},
                           'IoU':  {'train': [], 'eval': []}}

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

        model_folder = f'Models/{model_id}/fold{fold+1}'
        os.makedirs(model_folder, exist_ok=True)

        best_fold_iou = 0
        step = 0

        for epoch in range(max_epoch):

            # ── Train ─────────────────────────────────────────────────────
            model.train()
            for s1a, s1b, te, fm in train_loader:
                s1_pre    = s1a[:, :3, :, :].float().to(DEVICE)
                s1_during = s1b[:, :3, :, :].float().to(DEVICE)
                terrain   = te.float().to(DEVICE)
                target    = fm.squeeze(1).to(torch.int64).to(DEVICE)

                logits = model(s1_pre, s1_during, terrain)
                loss   = loss_fn(logits, target)
                model.zero_grad()
                loss.backward()
                opt.step()

                preds = logits.argmax(1)
                tp, fp, fn, tn = smp.metrics.get_stats(
                    preds, target, mode='multiclass', num_classes=3)
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='none')
                train_iou = iou[:, 1].mean().item()

                training_curves['Loss']['train'].append(loss.item())
                training_curves['IoU']['train'].append(train_iou)

                print(f'Fold {fold+1} | Epoch {epoch+1} | Step {step} | '
                      f'train_loss: {loss.item():.5f} | train_iou: {train_iou:.5f}', end='\r')
                step += 1

            # ── Validate ──────────────────────────────────────────────────
            model.eval()
            val_ious, val_losses = [], []
            for s1a, s1b, te, fm in val_loader:
                s1_pre    = s1a[:, :3, :, :].float().to(DEVICE)
                s1_during = s1b[:, :3, :, :].float().to(DEVICE)
                terrain   = te.float().to(DEVICE)
                target    = fm.squeeze(1).to(torch.int64).to(DEVICE)

                with torch.no_grad():
                    logits = model(s1_pre, s1_during, terrain)
                    loss_eval = loss_fn(logits, target)
                    preds = logits.argmax(1)

                    tp, fp, fn, tn = smp.metrics.get_stats(
                        preds, target, mode='multiclass', num_classes=3)
                    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='none')
                    val_ious.append(iou[:, 1].mean().item())
                    val_losses.append(loss_eval.item())

            epoch_iou     = np.mean(val_ious)
            epoch_val_loss = np.mean(val_losses)
            training_curves['Loss']['eval'].append(epoch_val_loss)
            training_curves['IoU']['eval'].append(epoch_iou)

            print(f'\nFold {fold+1} | Epoch {epoch+1} | '
                  f'flood_iou: {epoch_iou:.5f} | val_loss: {epoch_val_loss:.5f}')

            if epoch_iou > best_fold_iou:
                best_fold_iou = epoch_iou
                torch.save(model.state_dict(), f'{model_folder}/best_model.pt')
                print(f'  → Best fold model saved (flood_iou: {best_fold_iou:.4f})')

        np.save(f'{model_folder}/training_curves.npy', training_curves)
        fold_results.append(best_fold_iou)
        print(f'\nFold {fold+1} best IoU: {best_fold_iou:.4f}')

    print(f'\n{"="*40}')
    print(f'5-Fold CV Results:')
    for i, iou in enumerate(fold_results):
        print(f'  Fold {i+1}: {iou:.4f}')
    print(f'  Mean IoU: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}')
    print(f'{"="*40}')

    np.save(f'Models/{model_id}/fold_results.npy', fold_results)


if __name__ == '__main__':
    print("Starting Variant B (Siamese + difference fusion) training...")
    train(model_id='unet_efficientnetb3_Indonesia_siamese')
