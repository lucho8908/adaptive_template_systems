#!/bin/bash

python proteins_data_kernel_ridge.py cder poly
python proteins_data_kernel_ridge.py cder rbf
python proteins_data_kernel_ridge.py cder sigmoid

python proteins_data_kernel_ridge.py gmm poly
python proteins_data_kernel_ridge.py gmm rbf
python proteins_data_kernel_ridge.py gmm sigmoid

python proteins_data_kernel_ridge.py hbdscan poly
python proteins_data_kernel_ridge.py hbdscan rbf
python proteins_data_kernel_ridge.py hbdscan sigmoid