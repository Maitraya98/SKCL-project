#!/bin/bash -l
#SBATCH --job-name=train-concutmix-cifar
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# RECONSTRUCTED job script -- verify flags/paths against your cluster copy.
# ConCutMix's public repo required 5 compatibility fixes before it would
# run at all (see NOTES.md) -- this launches the patched local copy.

module add python
conda activate skcl
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

cd ~/skcl-project/ConCutMix
python3 train.py \
    --dataset cifar100 --data_root $WORK/data \
    --imb_factor 0.01 \
    --epochs 200 --batch_size 128 \
    --out_dir ../results/concutmix_cifar100
