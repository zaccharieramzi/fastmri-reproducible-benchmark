#!/bin/bash
module purge
module load python/3.7.5
module load tensorflow-gpu/py3/2.1.0

pip install ./

export FASTMRI_DATA_DIR=$SCRATCH/
export OASIS_DATA_DIR=$SCRATCH/
export LOGS_DIR=$SCRATCH/
export CHECKPOINTS_DIR=$SCRATCH/
