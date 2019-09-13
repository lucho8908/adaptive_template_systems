# Adaptive Template Systems

This repository implements the tools developed in the paper [Approximating Continuous Functions on Persistence Diagrams Using Template Functions](https://arxiv.org/abs/1902.07190). The main goal of this package is to provide a tool that extract features fro persistent diagrams in an adaptive manner.

## Requirments:
* Python/3.3. or higher
* GNU/4.7.1 or higher to run RIPSER
* GDA Toolbox package: https://github.com/geomdata/gda-public/
* Ripser
* Keras
* Theano
* Numba
* hdbscan

## Before running:

- Install Ripser
	~~~
	pip install Cython
	pip install Ripser
	~~~

- Install GDA Toolbox: First you need to clone the following repository https://github.com/geomdata/gda-public/

	~~~	 
	cd ~ #Any folder different to the one where gda-public was clones!
	conda create --name gda_env --file /path_of_cloned_repo/gda-public/requirements.txt python=3
	source activate gda_env
	pip install /path_of_cloned_repo/gda-public
	~~~

	To verify the installation, open a python worksheet and do
	~~~
	import multidim
	~~~

	If the previous steps where previously executed, the next time you only need to run
	~~~
	source activate gda_env
	~~~

- Install Keras+Theano:
	~~~
	pip install theano
	pip install keras
	~~~

- Install numba and hdbscan
	~~~
	conda install -y -c conda-forge hdbscan
	pip install numba
	~~~

## Running and Examples

In the folder `Examples` 