#!/bin/bash
# SCALEWAY_IP=163.172.181.188
SCALEWAY_IP=51.158.54.206
SCALEWAY=ubuntu@51.158.54.206
scp -i ~/.ssh/id_scaleway *.py *.au setup_scaleway.sh root@$SCALEWAY_IP:~
ssh -i ~/.ssh/id_scaleway root@$SCALEWAY_IP -t 'chmod +x setup_scaleway.sh; ./setup_scaleway.sh'
