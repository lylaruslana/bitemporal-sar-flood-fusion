#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --mem=39G
#SBATCH --nodelist=a100
#SBATCH --output=/home/lyla001/RP-ORKM/logs/%x_%j.out
#SBATCH --error=/home/lyla001/RP-ORKM/logs/%x_%j.err

cd /home/lyla001/RP-ORKM/ic3ina26
/home/lyla001/RP-ORKM/.venv/bin/python sensorflood_segformer_baseline.py
