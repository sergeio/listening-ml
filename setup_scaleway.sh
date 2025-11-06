#!/bin/bash
apt-get update
apt install python3-pip
pip3 install numpy librosa matplotlib dill micrograd --break-system-packages
python3 learn_multi_voices.py
