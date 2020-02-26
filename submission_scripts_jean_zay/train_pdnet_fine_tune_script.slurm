#!/bin/bash
#SBATCH --job-name=pdnet_fine_tune     # nom du job
#SBATCH --ntasks=2                   # nombre de tâche MPI
#SBATCH --ntasks-per-node=2          # nombre de tâche MPI par noeud
#SBATCH --gres=gpu:2                 # nombre de GPU à réserver par nœud
#SBATCH --cpus-per-task=10           # nombre de coeurs à réserver par tâche
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=10:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=pdnet_fine_tune%j.out # nom du fichier de sortie
#SBATCH --error=pdnet_fine_tune%j.out  # nom du fichier d'erreur (ici commun avec la sortie)

set -x
cd $WORK/fastmri-reproducible-benchmark

. ./submission_scripts_jean_zay/env_config.sh

srun python ./fastmri_recon/training_scripts/pdnet_approach_fine_tuning.py pdnet_af4_1568384763 -a 4 -c CORPDFS_FBK -gpus 0 &
srun python ./fastmri_recon/training_scripts/pdnet_approach_fine_tuning.py pdnet_af4_1568384763 -a 4 -c CORPD_FBK -gpus 1 &

wait  # wait for all ccc_mprun(s) to complete.
