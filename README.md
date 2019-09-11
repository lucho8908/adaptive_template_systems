Requires:
	- Python/3.3. or higher
	- GNU/4.7.1 or higher to run RIPSER
	- GDA Toolbox package: https://github.com/geomdata/gda-public/

To do before runing:

- Install Ripser
	> pip install Cython
	> pip install Ripser

- Before executing, you have to run:
	1.	conda create --name gda_env --file gda-public-master/requirements.txt python=3
	2.	source activate gda_env
	3.	pip install gda-public-master

- If the previous steps where previously executed we only need to run: source activate gda_env

- Requires Keras+Theano:
	1.	pip (conda) install theano
	2.	pip (conda) install keras

- Requires hdbscan, ripser and numba:
	1.	conda install -y -c conda-forge hdbscan
	2.	pip install ripser
	2.	pip install numba

