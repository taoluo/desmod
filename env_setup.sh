#!/bin/bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda

conda init

conda create -n des python=3.8
conda activate des
pip install requirement2.txt
