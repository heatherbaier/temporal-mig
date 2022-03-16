#!/bin/tcsh
#PBS -N extract
#PBS -l nodes=1:meltemi:ppn=32
#PBS -l walltime=24:00:00
#PBS -j oe

# init conda within new shell for job
source "/usr/local/anaconda3-2021.05/etc/profile.d/conda.csh"
module load anaconda3/2021.05
unsetenv PYTHONPATH
conda activate gee

python3 /sciclone/home20/hmbaier/tm/extract_features.py $RANK $WS

