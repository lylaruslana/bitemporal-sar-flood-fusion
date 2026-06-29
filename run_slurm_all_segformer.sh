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
echo "RUN 1/3: SegFormer Baseline (5ch)"
echo "========================================"
$PYTHON sensorflood_segformer_baseline.py

echo "========================================"
echo "RUN 2/3: SegFormer Variant A (8ch)"
echo "========================================"
$PYTHON sensorflood_segformer_bitemporal.py

echo "========================================"
echo "RUN 3/3: Update learning curves & viz"
echo "========================================"
$PYTHON plot_learning_curves.py
$PYTHON visualize_results.py

echo "========================================"
echo "SEMUA SELESAI"
echo "========================================"
