#!/bin/bash -l
#SBATCH --job-name=train-bcl-robotcar
#SBATCH --gres=gpu:1
#SBATCH --partition=work
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# RECONSTRUCTED job script -- the BCL/SKCL RobotCar comparison pair for
# the day/night domain-shift experiment.

module add python
conda activate skcl
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

cd ~/skcl-project
python3 train_bcl_robotcar.py \
    --data_root $WORK/data/robotcar_extracted \
    --epochs 100 --batch_size 64 \
    --eval_every 5 \
    --out_dir results/bcl_robotcar
