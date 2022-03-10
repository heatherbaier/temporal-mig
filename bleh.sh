#!/bin/bash

head -1 $PBS_NODEFILE > /sciclone/home20/hmbaier/tm/pbs_nf.txt
v=$(</sciclone/home20/hmbaier/tm/pbs_nf.txt)
grep $v /etc/hosts > /sciclone/home20/hmbaier/tm/etc_hosts.txt
tail -1 /sciclone/home20/hmbaier/tm/etc_hosts.txt | xargs | cut -d ' ' -f1 > /sciclone/home20/hmbaier/tm/good_ip.txt
