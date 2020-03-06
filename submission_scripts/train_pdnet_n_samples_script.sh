#!/bin/bash
#MSUB -r train_pdnet_n_samples                # Request name
#MSUB -n 2                         # Number of tasks to use
#MSUB -c 2                         # I want 2 cores per task since io might be costly
#MSUB -x
#MSUB -T 86400                      # Elapsed time limit in seconds
#MSUB -o pdnet_n_samples_%I.o              # Standard output. %I is the job id
#MSUB -e pdnet_n_samples_%I.e              # Error output. %I is the job id
#MSUB -q v100               # Queue
#MSUB -Q normal
#MSUB -m scratch,work
#MSUB -@ zaccharie.ramzi@gmail.com:begin,end
#MSUB -A gch0424                  # Project ID

set -x
cd $workspace/fastmri-reproducible-benchmark

. ./submission_scripts/env_config.sh

ccc_mprun -E '--exclusive' -n 1 python3 ./fastmri_recon/training_scripts/pdnet_approach.py -a 4 -e 200 -ns 400 -gpus 0 &
ccc_mprun -E '--exclusive' -n 1 python3 ./fastmri_recon/training_scripts/pdnet_approach.py -a 4 -e 200 -ns 2 -gpus 1 &

wait  # wait for all ccc_mprun(s) to complete.
