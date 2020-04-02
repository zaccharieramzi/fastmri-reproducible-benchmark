#!/bin/bash
#MSUB -r train_pdnet_huge                # Request name
#MSUB -n 1                         # Number of tasks to use
#MSUB -c 2                         # I want 2 cores per task since io might be costly
#MSUB -x
#MSUB -T 172800                      # Elapsed time limit in seconds
#MSUB -o pdnet_huge_%I.o              # Standard output. %I is the job id
#MSUB -e pdnet_huge_%I.e              # Error output. %I is the job id
#MSUB -q v100               # Queue
#MSUB -Q long
#MSUB -m scratch,work
#MSUB -@ zaccharie.ramzi@gmail.com:begin,end
#MSUB -A gch0424                  # Project ID

set -x
cd $workspace/fastmri-reproducible-benchmark

. ./submission_scripts/env_config.sh

ccc_mprun -E '--exclusive' -n 1 python3 ./fastmri_recon/training_scripts/single_coil/pdnet_approach.py -a 4 -s -i 20 &

wait  # wait for all ccc_mprun(s) to complete.
