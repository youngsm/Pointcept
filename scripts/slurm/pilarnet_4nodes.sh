#!/bin/bash
#SBATCH --job-name=sonata_dist
#SBATCH --nodes=4
#SBATCH --partition=ampere
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6
#SBATCH --mem=192G
#SBATCH --time=1:00:00
#SBATCH --account=mli:cider-ml
#SBATCH --output=slurm_logs/%j_%n_%x.txt

set -e

export PYTHONFAULTHANDLER=1

SINGULARITY_IMAGE_PATH=/sdf/group/neutrino/images/develop.sif

# Get current date and time in format YYYY-MM-DD_HH-MM-SS
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")

# Generate a truly unique port based on job ID, timestamp, and random number
# Use last 4 digits of SLURM_JOB_ID + offset to ensure uniqueness between jobs

if [ -n "$SLURM_NODELIST" ]; then
  # Get first node from the list for master
  export MASTER_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
  export MASTER_ADDR=$MASTER_HOSTNAME
  echo "Master node set to: $MASTER_ADDR"
  fi

# Remove deprecated variable
# unset NCCL_ASYNC_ERROR_HANDLING

# Enhanced NCCL settings for better performance and stability
# export NCCL_DEBUG=INFO  # Detailed NCCL logs for debugging
export NCCL_SOCKET_IFNAME=^docker0,lo  # Use any interface except docker and loopback
# export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
# export NCCL_P2P_DISABLE=0  # Enable P2P
# export NCCL_IB_TIMEOUT=30  # Increased timeout for InfiniBand operations
# export NCCL_SOCKET_NTHREADS=8  # More threads for socket operations
# export NCCL_MIN_NCHANNELS=4  # Use more channels
# export NCCL_BUFFSIZE=4194304  # Larger buffer size

# PyTorch distributed settings
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # PyTorch-specific setting
# export TORCH_DISTRIBUTED_DEBUG=DETAIL  # More detailed distributed logs
# export TORCH_DISTRIBUTED_TIMEOUT=1800  # 30 minutes timeout

# # Disable Gloo's TCP transport auto-tuning to avoid stalling
# export GLOO_SOCKET_TIMEOUT=1200  # 20 minutes timeout for Gloo sockets

TRAIN_PATH=/sdf/home/y/youngsam/sw/dune/representations/Pointcept/scripts/train.sh
COMMAND="sh ${TRAIN_PATH} -m 4 -g 4 -d pilarnet -c pretrain-sonata-pilarnet -n pretrain-sonata-pilarnet-1m-16GPU-${CURRENT_DATETIME}" # same as in SBATCH config!

echo "Using MASTER_ADDR=$MASTER_ADDR and MASTER_PORT=$MASTER_PORT"
echo "Network configuration: NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
export CUDA_LAUNCH_BLOCKING=1

# Launch with a 15 second delay between tasks to avoid simultaneous port binding
srun --kill-on-bad-exit=1 --ntasks-per-node=1 --nodes=$SLURM_NNODES \
  singularity run --nv -B /sdf,/fs,/sdf/scratch,/lscratch,/opt/slurm/ ${SINGULARITY_IMAGE_PATH} \
  bash -c "source ~/.bashrc && mamba activate pointcept-torch2.5.0-cu12.4 && ${COMMAND} $1"