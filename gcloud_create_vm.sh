gcloud compute instances create instance-20251106-20251106-193040 \
    --project=copper-aloe-477218-n1 \
    --zone=us-central1-b \
    --machine-type=c4a-highcpu-16 \
    --network-interface=network-tier=PREMIUM,nic-type=GVNIC,stack-type=IPV4_ONLY,subnet=default \
    --metadata=ssh-keys=odd_pace5473:ssh-rsa\ \
AAAAB3NzaC1yc2EAAAADAQABAAABgQDAq\+Txi9yh5BPU3Qj0d8GUQeTFKxMQRjCoDP5fdwXsn6Wpnn9lo4XZZ7D44NzsiBe7pPCjBcPSuIEO3WCqLGXxxd8XfHJo2WSTL8VZy4By1KqXQiHkOcZCQS2mNwPj4I4/5mOQflsG13UEUIAzSi\+HvrcFr2hUqh9LZIhIhirbciYtfa1aAh5nuX\+m8iEbi8Uar9UyVygvDGSOz1NngB1p7qUk7k2XuulNEuHXlafRWENBD09t5L0vFiXNP7\+OCEQwGdH\+R9BU1FqzVkWA9r/EinZKUkWAG0A2UCi/4D82AOrL7RV7GimyT1HFEhtOOXeCzIbwCyosyK5Pvg2EpBrqeoqKG/TIHDZ86xtG8xVHoSMc0QEef8\+vs/YrK\+/f15937/WZDadWXPFAhJK6P29ZqUcJteGYbKZIuQfdWDQcnUhlR72l2YcFkmtssC897kfSgkv\+/WQcRG\+fs34dw4JR9Pq14H85/JnFVIpuP1f4JlEG3\+k/8DceIvBY34CGWBk=\ odd_pace5473 \
    --maintenance-policy=MIGRATE \
    --provisioning-model=STANDARD \
    --service-account=1045657370063-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
    --create-disk=auto-delete=yes,boot=yes,device-name=instance-20251106-171631,disk-resource-policy=projects/copper-aloe-477218-n1/regions/us-central1/resourcePolicies/default-schedule-1,image=projects/ubuntu-os-cloud/global/images/ubuntu-minimal-2510-questing-arm64-v20251024,mode=rw,provisioned-iops=3120,provisioned-throughput=170,size=20,type=hyperdisk-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any
