import numpy as np
import multidim
import itertools
import os
import hdbscan
import sys
import time
import pandas as pd

from copy import deepcopy
from matplotlib.patches import Ellipse
from ripser import ripser
from persim import plot_diagrams
from numba import jit, njit, prange
from sklearn import mixture
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import RidgeClassifier

from multidim.covertree import CoverTree
from multidim.models import CDER

import matplotlib.pyplot as plt

np.set_printoptions(precision=2)

sys.path.append('../..')
from ATS import *

# -----------------------------------------------------------------------------
# ------------------------------ IMPORT DATA ----------------------------------
# -----------------------------------------------------------------------------

num_dgms = int(sys.argv[1])

N = num_dgms*6

d = 10

colors = ['red', 'yellow', 'magenta', 'green', 'blue', 'black']

os.system('mkdir cder_images')
os.system('rm -r cder_images/*')

score_train = []
score_test = []
for n in range(10):

	X = testSetManifolds(numDgms = num_dgms, numPts = 200, permute = True)

	F_labels = X.trainingLabel

	labels = X.trainingLabel.unique()

	X_dgm0 = X.Dgm0.tolist()

	# We need to perturbate H_0 to use CDER.
	for h0 in X_dgm0:
		h0[:,0] = h0[:,0] + np.random.uniform(-0.05, 0.05, len(h0))
		h0[:,1][h0[:,1]==np.inf] = 10 # Changge all inf values in H_0 for 10.

	X_dgm1 = X.Dgm1.tolist()

	i=0
	for l in labels:
		F_labels[F_labels == l]=i
		i += 1

	F = F_labels.tolist()

	# -----------------------------------------------------------------------------
	# ------------------------------ H_0 ------------------------------------------
	# -----------------------------------------------------------------------------

	X_train, X_test, F_train, F_test = train_test_split(X_dgm0, F, test_size=0.33, random_state=10)

	# -----------------------------------------------------------------------------
	# ------------------------------ GMM ------------------------------------------
	# -----------------------------------------------------------------------------

	print('Begin GMM...')
	t0 = time.time()
	X_train_temp = np.vstack(X_train)

	X_train_temp = X_train_temp[:,1]

	X_train_temp = X_train_temp.reshape((-1,1))

	gmm_f_train=[]
	for i in range(len(X_train)):
		gmm_f_train.append(F_train[i]*np.ones(len(X_train[i])))
	gmm_f_train = np.concatenate(gmm_f_train)

	gmm = mixture.BayesianGaussianMixture(n_components=d, covariance_type='full', max_iter=int(10e4)).fit(X_train_temp, gmm_f_train)

	ellipses = []
	for i in range(len(gmm.means_)):
		L, v = np.linalg.eig(gmm.covariances_[i])
		temp = {'mean':gmm.means_[i], 'std':np.sqrt(L), 'rotation':v.transpose(), 'radius':max(np.sqrt(L)), 'entropy':gmm.weights_[i]}
		ellipses.append(temp)
	t1 = time.time()
	print('Finish GMM. Time: {}'.format(t1-t0))

	# -----------------------------------------------------------------------------
	# ------------------------------ GMM features ---------------------------------
	# -----------------------------------------------------------------------------
	t0 = time.time()

	X_train_temp = [dgm[:,1] for dgm in X_train]
	X_train_features_0 = get_all_features(X_train_temp, ellipses, f_gaussian)

	X_test_temp = [dgm[:,1] for dgm in X_test]
	X_test_features_0 = get_all_features(X_test_temp, ellipses, f_gaussian)

	t1 = time.time()
	print('Features H_0:{}'.format(t1-t0))

	# -----------------------------------------------------------------------------
	# ------------------------------ H_1 ------------------------------------------
	# -----------------------------------------------------------------------------

	X_train, X_test, F_train, F_test = train_test_split(X_dgm1, F, test_size=0.33, random_state=10)
	
	# -----------------------------------------------------------------------------
	# ------------------------------ CDER -----------------------------------------
	# -----------------------------------------------------------------------------

	F_train_cder = F_train.copy()

	for l in range(6):
		for k, j in enumerate(F_train_cder):
			if j == l:
				F_train_cder[k] = colors[l]

	pc_train = multidim.PointCloud.from_multisample_multilabel(X_train, F_train_cder)
	ct_train = CoverTree(pc_train)

	cder = CDER(parsimonious=True)

	cder.fit(ct_train)

	cder_result = cder.gaussians

	ellipses = []
	for c in cder_result:
		temp = {key:c[key] for key in ['mean', 'std', 'rotation', 'radius', 'entropy']}
		temp['std'] = 3*temp['std']
		ellipses.append(temp)

	for i in range(len(X_train)):
		dgm = np.array(X_train[i])
		plt.scatter(dgm[:,0], dgm[:,1], color='grey')
		
	ellipses_plot_cder = []
	for i in range(len(ellipses)):
		e = ellipses[i]
		ellipses_plot_cder.append(Ellipse(xy=e['mean'], width=e['std'][0], height=e['std'][1], angle=np.arccos(e['rotation'][0,0])))

	for e in ellipses_plot_cder:
		plt.gca().add_artist(e)
		e.set_clip_box(plt.gca().bbox)
		e.set_alpha(0.5)
		e.set_facecolor([1,0,0])
	plt.savefig('cder_images/{}_h1_cder_n_{}.png'.format(n, num_dgms))
	plt.close()

	# -----------------------------------------------------------------------------
	# ------------------------------ CDER features --------------------------------
	# -----------------------------------------------------------------------------

	X_train_features_1 = get_all_features(X_train, ellipses, f_ellipse)

	X_test_features_1 = get_all_features(X_test, ellipses, f_ellipse)

	# -----------------------------------------------------------------------------
	# ------------------------------ Ridge Classification  ------------------------
	# -----------------------------------------------------------------------------

	X_train_features = np.column_stack((X_train_features_0, X_train_features_1))

	X_test_features = np.column_stack((X_test_features_0, X_test_features_1))

	ridge_model = RidgeClassifier().fit(X_train_features, F_train)
	
	score_train.append(ridge_model.score(X_train_features, F_train))
	score_test.append(ridge_model.score(X_test_features, F_test))

	# print('train', score_train[-1])
	# print('test', score_test[-1])

print(np.mean(score_train), np.std(score_train))
print(np.mean(score_test), np.std(score_test))