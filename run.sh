#!/bin/bash

qsub head_node.sh -v NUMNODES=$1

# wait unitl the head node has launched and created the ip address file
while [ ! -f /sciclone/home20/hmbaier/tm/here.txt ]
do
  sleep 2 # or less like 0.2
done

# grab the ip address from the file
value=$(</sciclone/home20/hmbaier/tm/here.txt)
echo $value

rm /sciclone/home20/hmbaier/tm/here.txt

# then submit the job array with the ip address
qsub workers.sh -v IP_ADDRESS=$value,NUMNODES=$1 -t 1-$1-1
