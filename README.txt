To do before runing:

- Requires the Teaspoon packed: http://elizabethmunch.com/code/teaspoon/index.html

- Requires the GDA Toolbox package: https://github.com/geomdata/gda-public/

- This folder contains a copy of GDA Toolbox: gda-public-master
- Before executing, you have to run:
	1) conda create --name gda_env --file gda-public-master/requirements.txt python=3
	2) source activate gda_env
	3) pip install gda-public-master

- If the previous steps where previously executed we only need to run: source activate gda_env

- Requires Keras+Theano:
	- pip (conda) install theano
	- pip (conda) install keras

- Requires hdbscan, ripser and numba:
	- conda install -y -c conda-forge hdbscan
	- pip install ripser
	- pip install numba

Requires:
	- Python/3.3. or higher
	- GNU/4.7.1 or higher to run RIPSER