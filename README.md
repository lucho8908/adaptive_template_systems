# Adaptive Template Systems

by [Luis Polanco Contreras](https://www.egr.msu.edu/~polanco2/) and [Jose A. Perea](https://www.joperea.com/)

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

We are gonna create an isolated enviroment to work, to do so you need the file ```environment.yml``` in this repository.

1.  Lets create a new enviroment
	~~~
	conda env create -f environment.yml
	~~~
	
2. Lets activate the enviroment
	~~~
	conda activate ats_env
	~~~
	or 
	~~~
	source activate ats_env
	~~~
	
3. Clone the following repository: [GDA Toolbax](https://github.com/geomdata/gda-public/)

4. cd ~ #Any folder different to the one where gda-public was cloned!

5. Execute 
	~~~
	pip install /path_of_cloned_repo/gda-public
	~~~
	
6. Let us verify our instalation:
	1. Open a python worksheet by typing: ```ipython``` os ```python```
	2. type: ```import multidim, ripser```
	

## Running and Examples

In the folder `Examples` are 3 different examples available: 6-class manifold classification problem, shape classification from the SHREC 2014 data set and protein classification problem from the SCOPe data base.
