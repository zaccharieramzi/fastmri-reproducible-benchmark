#!/bin/bash
#SBATCH --job-name=updnet_brain_finetune    # nom du job
#SBATCH --ntasks=1                   # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --gres=gpu:1                 # nombre de GPU à réserver par nœud
#SBATCH --cpus-per-task=20           # nombre de coeurs à réserver par tâche
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=40:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=updnet_brain_finetune%A_%a.out # nom du fichier de sortie
#SBATCH --error=updnet_brain_finetune%A_%a.out  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-9

set -x
cd $WORK/fastmri-reproducible-benchmark

. ./submission_scripts_jean_zay/env_config.sh

RUN_ID_AF4 = ""
RUN_ID_AF8 = ""

# AF4 finetuning
opt[0]="-a 4 -b -l compound_mssim -rfs --n-epochs-orig 200 --orig-id ${RUN_ID_AF4} -e 10 -c AXFLAIR"
opt[1]="-a 4 -b -l compound_mssim -rfs --n-epochs-orig 200 --orig-id ${RUN_ID_AF4} -e 10 -c AXT1POST"
opt[2]="-a 4 -b -l compound_mssim -rfs --n-epochs-orig 200 --orig-id ${RUN_ID_AF4} -e 10 -c AXT1PRE"
opt[3]="-a 4 -b -l compound_mssim -rfs --n-epochs-orig 200 --orig-id ${RUN_ID_AF4} -e 10 -c AXT1"
opt[4]="-a 4 -b -l compound_mssim -rfs --n-epochs-orig 200 --orig-id ${RUN_ID_AF4} -e 10 -c AXT2"
# AF8 finetuning
opt[5]="-a 8 -b -l compound_mssim -rfs --n-epochs-orig 200 --orig-id ${RUN_ID_AF8} -e 10 -c AXFLAIR"
opt[6]="-a 8 -b -l compound_mssim -rfs --n-epochs-orig 200 --orig-id ${RUN_ID_AF8} -e 10 -c AXT1POST"
opt[7]="-a 8 -b -l compound_mssim -rfs --n-epochs-orig 200 --orig-id ${RUN_ID_AF8} -e 10 -c AXT1PRE"
opt[8]="-a 8 -b -l compound_mssim -rfs --n-epochs-orig 200 --orig-id ${RUN_ID_AF8} -e 10 -c AXT1"
opt[9]="-a 8 -b -l compound_mssim -rfs --n-epochs-orig 200 --orig-id ${RUN_ID_AF8} -e 10 -c AXT2"


srun python ./fastmri_recon/training_scripts/updnet_train.py ${opt[$SLURM_ARRAY_TASK_ID]}&

wait  # wait for all ccc_mprun(s) to complete.
