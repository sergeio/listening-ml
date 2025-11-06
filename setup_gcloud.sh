#!/bin/bash
sudo apt-get update
sudo apt install python3-pip -y
pip3 install numpy librosa matplotlib dill micrograd --break-system-packages
time python3 learn_multi_voices.py
