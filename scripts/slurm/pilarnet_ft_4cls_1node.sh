#!/bin/bash
#SBATCH --job-name=sonata_dist
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --account=neutrino:cider-nu
#SBATCH --partition=ampere
#SBATCH --output=slurm_logs/%j_%n_%x.txt

set -e

export PYTHONFAULTHANDLER=1

SINGULARITY_IMAGE_PATH=/sdf/group/neutrino/images/develop.sif

# Get current date and time in format YYYY-MM-DD_HH-MM
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M")

TRAIN_PATH=/sdf/home/y/youngsam/sw/dune/representations/Pointcept/scripts/train.sh
COMMAND="sh ${TRAIN_PATH} -m 1 -g 4 -d pilarnet -c semseg-sonata-pilarnet-ft-4cls -n sonata-pilarnet-semseg-ft-4cls-v1-4GPU-100ev-256patch-${CURRENT_DATETIME} -w exp/pilarnet/pretrain-sonata-pilarnet-100k-4GPU-2025-04-26_15-18/model/model_last.pth" # same as in SBATCH config!

srun singularity run --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SINGULARITY_IMAGE_PATH} \
    bash -c "source ~/.bashrc && mamba activate pointcept-torch2.5.0-cu12.4 && ${COMMAND} $1"