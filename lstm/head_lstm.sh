#!/bin/tcsh
#PBS -N head_node
#PBS -l walltime=24:00:00
#PBS -j oe

hostname -i > /sciclone/home20/hmbaier/tm/lstm/here.txt
date "+%F, %T" > /sciclone/home20/hmbaier/tm/lstm/date.txt

# init conda within new shell for job
source "/usr/local/anaconda3-2021.05/etc/profile.d/conda.csh"
module load anaconda3/2021.05
unsetenv PYTHONPATH
conda activate dhsrl4

torchrun --nnodes=$NUMNODES --nproc_per_node=$PPN --rdzv_id=790876 --rdzv_backend=c10d /sciclone/home20/hmbaier/tm/lstm/run_lstm.py $PPN $NUMNODES