#!/bin/bash
#SBATCH --job-name=pdnet_n_samples     # nom du job
#SBATCH --ntasks=4                   # nombre de tâche MPI
#SBATCH --ntasks-per-node=4          # nombre de tâche MPI par noeud
#SBATCH --gres=gpu:4                 # nombre de GPU à réserver par nœud
#SBATCH --cpus-per-task=10           # nombre de coeurs à réserver par tâche
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=pdnet_n_samples%j.out # nom du fichier de sortie
#SBATCH --error=pdnet_n_samples%j.out  # nom du fichier d'erreur (ici commun avec la sortie)

set -x
cd $WORK/fastmri-reproducible-benchmark

. ./submission_scripts_jean_zay/env_config.sh

srun --multi-prog ./submission_scripts_jean_zay/mpmd_pdnet_n_samples.conf
