#!/bin/bash -l
#SBATCH --job-name=train-skcl-cifar
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# RECONSTRUCTED job script -- exact flags/paths approximate; verify against
# your cluster copy. ~30-40 min for 200 epochs on an RTX 2080 Ti per past logs.

module add python
conda activate skcl
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

cd ~/skcl-project
python3 train_skcl_cifar.py \
    --dataset cifar100 --data_root $WORK/data \
    --imb_factor 0.01 \
    --graph graph_cifar100_real.json \
    --epochs 200 --batch_size 128 \
    --tau 0.1 --tau_prime 0.2 \
    --out_dir results/skcl_cifar100
