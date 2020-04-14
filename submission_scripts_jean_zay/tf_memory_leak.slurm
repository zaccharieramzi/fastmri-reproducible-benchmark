#!/bin/bash
#SBATCH --job-name=tf_memory_leak     # nom du job
#SBATCH --ntasks=1                   # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --gres=gpu:4                 # nombre de GPU à réserver par nœud
#SBATCH --partition=gpu_p2s
#SBATCH --cpus-per-task=4           # nombre de coeurs à réserver par tâche
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-dev         # le calcul va etre long
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=2:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%j.out # nom du fichier de sortie
#SBATCH --error=%x_%j.out  # nom du fichier d'erreur (ici commun avec la sortie)

set -x
cd $WORK/fastmri-reproducible-benchmark

. ./submission_scripts_jean_zay/env_config.sh

srun --ntasks=1 python ./submission_scripts_jean_zay/tf_memory_leak.py&

wait  # wait for all ccc_mprun(s) to complete.
