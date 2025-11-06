#!/bin/bash
set -e
sudo apt-get update
sudo apt install python3-pip python3-venv -y
# pip3 install setuptools distutils --break-system-packages
python3 -m venv venv
venv/bin/pip3 install torch numpy matplotlib librosa dill micrograd
time venv/bin/python3 test_reconstruction_deep.py
