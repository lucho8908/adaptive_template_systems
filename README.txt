To do before runing:

 - This folder contains gda-public-master
 - Before executing, you have to run:
   1) conda create --name gda_env --file gda-public-master/requirements.txt python=3
   2) source activate gda_env
   3) pip install gda-public-master
 - If the previous steps where previously executed we only need to run: source activate gda_env

 - Requires Keras+Theano:
 - pip (conda) install theano
 - pip (conda) install keras

Requires:
 - Python/3.3. or higher
 - GNU/4.7.1 or higher to run RIPSER

TO CHANGE:
 - Do not include all dimensions in the same diagram (DONE!)
 - Evaluate points in diagram in all necesarry ellipses (DONE!)

To DO:
 - Keras + Theanos (DONE!)
 - Examples: - Henry Adams, Persisitent images
	     - Bauer Reinhaus
 - fOr H_0 consider pair (life_time, death)
 - For time series -> Normalize ( x - mean )/var
