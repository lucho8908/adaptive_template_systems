Requirments:
	- Python/3.3. or higher
	- GNU/4.7.1 or higher to run RIPSER
	- GDA Toolbox package: https://github.com/geomdata/gda-public/

To do before runing:

- Install Ripser
	~~~
	pip install Cython
	pip install Ripser
	~~~

- Install: First you need to clone the fllowing repository https://github.com/geomdata/gda-public/

	~~~	 
	cd ~ #Any forlder diferent to the one where gda-public was clones!
	conda create --name gda_env --file /path_of_cloned_repo/gda-public/requirements.txt python=3
	source activate gda_env
	pip install /path_of_cloned_repo/gda-public
	~~~

	To verify the installation, open a python worksheet and do
	~~~
	import multidim
	~~~

- If the previous steps where previously executed, the next time you only need to run
	~~~
	source activate gda_env
	~~~

- Isntall Keras+Theano:
	~~~
	pip install theano
	pip install keras
	~~~

- Requires hdbscan, ripser and numba:

	1.	conda install -y -c conda-forge hdbscan
	2.	pip install ripser
	2.	pip install numba

