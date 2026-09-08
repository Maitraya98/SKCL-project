#!/bin/bash -l
#SBATCH --job-name=train-skcl-robotcar
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# RECONSTRUCTED job script. Trains on day_train only; evaluates on both
# day_test and night_test every few epochs (domain-shift Part 2 experiment).

module add python
conda activate skcl
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

cd ~/skcl-project
python3 train_skcl_robotcar.py \
    --data_root $WORK/data/robotcar_extracted \
    --graph graph_robotcar_real.json \
    --epochs 100 --batch_size 64 \
    --eval_every 5 \
    --out_dir results/skcl_robotcar
