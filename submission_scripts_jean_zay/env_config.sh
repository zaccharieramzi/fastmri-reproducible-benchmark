#!/bin/bash
module purge
module load python/3.7.5 cuda/10.1.1 cudnn/7.6.5.32-cuda-10.1

export TMPDIR=$SCRATCH/tmp
pip install --no-cache-dir ./
pip install -r requirements.txt

export FASTMRI_DATA_DIR=$SCRATCH/
export OASIS_DATA_DIR=$SCRATCH/
export LOGS_DIR=$SCRATCH/
export CHECKPOINTS_DIR=$SCRATCH/
