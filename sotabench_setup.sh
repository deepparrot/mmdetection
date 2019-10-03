#!/usr/bin/env bash
source /workspace/venv/bin/activate
PYTHON=${PYTHON:-"python"}

pip install torch

apt-get git
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./

cd ..
echo "Installing..."
$PYTHON setup.py develop
