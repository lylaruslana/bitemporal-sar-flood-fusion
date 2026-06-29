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
echo "RUN 1/4: UNet ResNet-34 (early fusion, 8ch)"
echo "========================================"
$PYTHON sensorflood_earlyfusion.py \
    --encoder resnet34 \
    --model-id unet_resnet34_Indonesia_bitemporal

echo "========================================"
echo "RUN 2/4: UNet ResNet-50 (early fusion, 8ch)"
echo "========================================"
$PYTHON sensorflood_earlyfusion.py \
    --encoder resnet50 \
    --model-id unet_resnet50_Indonesia_bitemporal

echo "========================================"
echo "RUN 3/4: UNet EfficientNet-B1 (early fusion, 8ch)"
echo "========================================"
$PYTHON sensorflood_earlyfusion.py \
    --encoder efficientnet-b1 \
    --model-id unet_efficientnetb1_Indonesia_bitemporal

echo "========================================"
echo "RUN 4/4: UNet EfficientNet-B2 (early fusion, 8ch)"
echo "========================================"
$PYTHON sensorflood_earlyfusion.py \
    --encoder efficientnet-b2 \
    --model-id unet_efficientnetb2_Indonesia_bitemporal

echo "========================================"
echo "SEMUA SELESAI"
echo "========================================"
