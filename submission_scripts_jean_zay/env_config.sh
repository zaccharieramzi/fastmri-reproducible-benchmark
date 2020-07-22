#!/bin/bash
module purge
module load python/3.7.5 cuda/10.1.2 cudnn/7.6.5.32-cuda-10.1 nccl/2.5.6-2-cuda
conda activate fastmri-tf-2.1.0

export FASTMRI_DATA_DIR=$SCRATCH/
export OASIS_DATA_DIR=$SCRATCH/OASIS_data
export LOGS_DIR=$SCRATCH/
export CHECKPOINTS_DIR=$SCRATCH/
