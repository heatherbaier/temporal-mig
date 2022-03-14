#!/bin/bash

for ((i = 1; i <= $1; i++))
do
    qsub ncdf_job.sh -v RANK=$i,WS=$1
done
