#!/bin/bash
#MSUB -r train_pdnet_fine_tune                # Request name
#MSUB -n 2                         # Number of tasks to use
#MSUB -c 2                         # I want 2 cores per task since io might be costly
#MSUB -x
#MSUB -T 86400                      # Elapsed time limit in seconds
#MSUB -o pdnet_fine_tune_%I.o              # Standard output. %I is the job id
#MSUB -e pdnet_fine_tune_%I.e              # Error output. %I is the job id
#MSUB -q v100               # Queue
#MSUB -Q normal
#MSUB -m scratch,work
#MSUB -@ zaccharie.ramzi@gmail.com:begin,end
#MSUB -A gch0424                  # Project ID

set -x
cd $workspace/fastmri-reproducible-benchmark

. ./submission_scripts/env_config.sh

ccc_mprun -E '--exclusive' -n 1 python3 ./fastmri_recon/training_scripts/single_coil/pdnet_approach_fine_tuning.py pdnet_af4_1568384763 -a 4 -c CORPDFS_FBK -gpus 0 &
ccc_mprun -E '--exclusive' -n 1 python3 ./fastmri_recon/training_scripts/single_coil/pdnet_approach_fine_tuning.py pdnet_af4_1568384763 -a 4 -c CORPD_FBK -gpus 1 &

wait  # wait for all ccc_mprun(s) to complete.
