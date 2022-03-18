#!/bin/tcsh
#PBS -N worker
#PBS -l walltime=24:00:00
#PBS -j oe

# init conda within new shell for job
source "/usr/local/anaconda3-2021.05/etc/profile.d/conda.csh"
module load anaconda3/2021.05
unsetenv PYTHONPATH
conda activate dhsrl4

echo $IP_ADDRESS

torchrun --nnodes=$NUMNODES --nproc_per_node=$PPN --rdzv_id=790876 --rdzv_backend=c10d --rdzv_endpoint=$IP_ADDRESS /sciclone/home20/hmbaier/tm/lstm/run_lstm.py $PPN $NUMNODES


