#!/bin/bash
#SBATCH --job-name=fastmri_test     # nom du job
#SBATCH --ntasks=1                   # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --gres=gpu:1                 # nombre de GPU à réserver par nœud
#SBATCH --cpus-per-task=10           # nombre de coeurs à réserver par tâche
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=00:45:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-dev
#SBATCH --output=fastmri_test%A_%a.out # nom du fichier de sortie
#SBATCH --error=fastmri_test%A_%a.out  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-8

set -x
cd $WORK/fastmri-reproducible-benchmark

. ./submission_scripts_jean_zay/env_config.sh

opt[0]="fastmri_recon/tests/data"
opt[1]="fastmri_recon/tests/models"
opt[2]="fastmri_recon/tests/training_scripts/denoising"
opt[3]="fastmri_recon/tests/training_scripts/multi_coil"
opt[4]="fastmri_recon/tests/training_scripts/dealiasing_train_test.py"
opt[5]="fastmri_recon/tests/training_scripts/ncpdnet_train_test.py"
opt[6]="fastmri_recon/tests/training_scripts/unet_nc_train_test.py"
opt[7]="fastmri_recon/tests/training_scripts/updnet_train_test.py"
opt[8]="fastmri_recon/tests/training_scripts/xpdnet_train_test.py"

srun pytest -s ${opt[$SLURM_ARRAY_TASK_ID]}
