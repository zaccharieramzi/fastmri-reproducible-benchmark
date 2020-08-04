#!/bin/bash
#SBATCH --job-name=updnet_eval     # nom du job
#SBATCH --ntasks=1                   # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --gres=gpu:4                 # nombre de GPU à réserver par nœud
#SBATCH --cpus-per-task=40           # nombre de coeurs à réserver par tâche
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=updnet_eval%A_%a.out # nom du fichier de sortie
#SBATCH --error=updnet_eval%A_%a.out  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-3

set -x
cd $WORK/fastmri-reproducible-benchmark

. ./submission_scripts_jean_zay/env_config.sh

RUN_ID_AF4_CORPDFS_FBK = "updnet_sense_af4_CORPDFS_FBK_i25_prelu_l4_compound_mssim_ca_rf_sm_1590842754"
RUN_ID_AF4_CORPD_FBK = "updnet_sense_af4_CORPD_FBK_i25_prelu_l4_compound_mssim_ca_rf_sm_1590843080"
RUN_ID_AF8_CORPDFS_FBK = "updnet_sense_af8_CORPDFS_FBK_i25_prelu_l4_compound_mssim_ca_rf_sm_1590843080"
RUN_ID_AF8_CORPD_FBK = "updnet_sense_af8_CORPD_FBK_i25_prelu_l4_compound_mssim_ca_rf_sm_1590843080"


# AF4 eval
opt[0]="-a 4 -rfs -r ${RUN_ID_AF4_CORPDFS_FBK} -e 50 -c CORPDFS_FBK -i 25 -nl prelu -ca dense -la 4"
opt[1]="-a 4 -rfs -r ${RUN_ID_AF4_CORPD_FBK} -e 50 -c CORPD_FBK -i 25 -nl prelu -ca dense -la 4"
# AF8 eval
opt[2]="-a 8 -rfs -r ${RUN_ID_AF8_CORPDFS_FBK} -e 50 -c CORPDFS_FBK -i 25 -nl prelu -ca dense -la 4"
opt[3]="-a 8 -rfs -r ${RUN_ID_AF8_CORPD_FBK} -e 50 -c CORPD_FBK -i 25 -nl prelu -ca dense -la 4"

srun python ./fastmri_recon/evaluate/scripts/updnet_sense_eval.py ${opt[$SLURM_ARRAY_TASK_ID]}&

wait  # wait for all ccc_mprun(s) to complete.
