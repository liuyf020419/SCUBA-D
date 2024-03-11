#!/usr/bin/env bash
set -e

pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torch_geometric==2.2.0

pip install torch_sparse==0.6.15 -f https://pytorch-geometric.com/whl/torch-1.11.0%2Bcu113.html
pip install torch_scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.11.0%2Bcu113.html
pip install torch_cluster==1.6.0 -f https://pytorch-geometric.com/whl/torch-1.11.0%2Bcu113.html
pip install torch_spline_conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.11.0%2Bcu113.html

conda install -c ostrokach dssp