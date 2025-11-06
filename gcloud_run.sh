#!/bin/bash
IP="${1:-34.58.135.57}"
GOOGLE_USER=odd_pace5473@$IP
# ssh odd_pace5473@35.238.172.47 -i ~/.ssh/id_gcloud
scp -i ~/.ssh/id_gcloud *.py *.au setup_gcloud.sh $GOOGLE_USER:~
ssh -i ~/.ssh/id_gcloud $GOOGLE_USER -t 'chmod +x setup_gcloud.sh; ./setup_gcloud.sh'
