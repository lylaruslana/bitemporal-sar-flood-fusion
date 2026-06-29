#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --mem=39G
#SBATCH --nodelist=a100
#SBATCH --output=/home/lyla001/RP-ORKM/logs/%x_%j.out
#SBATCH --error=/home/lyla001/RP-ORKM/logs/%x_%j.err

cd /home/lyla001/RP-ORKM/ic3ina26
PYTHON=/home/lyla001/RP-ORKM/.venv/bin/python

echo "========================================"
echo "Baseline UNet EfficientNet-B2 (5ch, during only)"
echo "========================================"
$PYTHON sensorflood_baseline_generic.py \
    --encoder efficientnet-b2 \
    --model-id unet_efficientnetb2_Indonesia_baseline

echo "========================================"
echo "SELESAI"
echo "========================================"
