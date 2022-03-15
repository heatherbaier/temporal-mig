#!/bin/bash

rm /sciclone/home20/hmbaier/tm/date.txt

qsub head_node.sh -l nodes=1:$2:ppn=$3 -v NUMNODES=$1,PPN=$3

# wait until the head node has launched and created the ip address file
while [ ! -f /sciclone/home20/hmbaier/tm/here.txt ]
do
  sleep 1
done

# grab the ip address from the file
value=$(</sciclone/home20/hmbaier/tm/here.txt)

echo $value
echo $2
echo $3

rm /sciclone/home20/hmbaier/tm/here.txt

# then submit the job array with the ip address
for ((i = 1; i <= $1-1; i++))
do
  qsub workers.sh -l nodes=1:$2:ppn=$3 -v IP_ADDRESS=$value,NUMNODES=$1,PPN=$3
done