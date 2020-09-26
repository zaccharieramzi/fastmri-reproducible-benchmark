#!/bin/bash
#SBATCH --job-name=xpdnet_brain_train_2     # nom du job
#SBATCH --ntasks=1                   # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --gres=gpu:1                 # nombre de GPU à réserver par nœud
#SBATCH --cpus-per-task=20           # nombre de coeurs à réserver par tâche
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=80:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=xpdnet_brain_train2_%A_%a.out # nom du fichier de sortie
#SBATCH --error=xpdnet_brain_train2_%A_%a.out  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-1
#SBATCH --dependency=afterok:378637

set -x
cd $WORK/fastmri-reproducible-benchmark

. ./submission_scripts_jean_zay/env_config.sh

export RUN_ID_AF4="xpdnet_sense_brain__af4_mssim_rf_sm_MWCNNbig_1601070404"
export RUN_ID_AF8="xpdnet_sense_brain__af8_mssim_rf_sm_MWCNNbig_1601070401"

opt[0]="-a 4 --orig-id ${RUN_ID_AF4}"
opt[1]="-a 8 --orig-id ${RUN_ID_AF8}"

srun python ./fastmri_recon/training_scripts/xpdnet_train.py ${opt[$SLURM_ARRAY_TASK_ID]} -b -l mssim -rfs -e 50 -m MWCNN -s big -eqf -ec 50 &

wait  # wait for all ccc_mprun(s) to complete.
