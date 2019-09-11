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


# ------------------------------ IMPORT DATA ----------------------------------

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

	# ------------------------------ H_0 ------------------------------------------

	X_train, X_test, F_train, F_test = train_test_split(X_dgm0, F, test_size=0.33, random_state=10)

	# ------------------------------ GMM -----------------------------------------

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

	print(gmm.means_)

	print(gmm.covariances_)

	ellipses = []
	for i in range(len(gmm.means_)):
		L, v = np.linalg.eig(gmm.covariances_[i])
		temp = {'mean':gmm.means_[i], 'std':np.sqrt(L), 'rotation':v.transpose(), 'radius':max(np.sqrt(L)), 'entropy':gmm.weights_[i]}
		ellipses.append(temp)
	t1 = time.time()
	print('Finish GMM. Time: {}'.format(t1-t0))

	# for i in range(len(X_train)):
	# 	dgm = np.array(X_train[i])
	# 	plt.scatter(dgm[:,0], dgm[:,1], color='grey')

	# ellipses_plot_cder = []
	# for i in range(len(ellipses)):
	# 	e = ellipses[i]
	# 	ellipses_plot_cder.append(Ellipse(xy=e['mean'], width=e['std'][0], height=e['std'][0], angle=np.arccos(e['rotation'][0,0])))

	# for e in ellipses_plot_cder:
	# 	plt.gca().add_artist(e)
	# 	e.set_clip_box(plt.gca().bbox)
	# 	e.set_alpha(0.5)
	# 	e.set_facecolor([1,0,0])
	# plt.savefig('gmm_images/{}_h0_cder_n_{}.png'.format(n, num_dgms))
	# plt.close()

	# ------------------------------ GMM features --------------------------------
	t0 = time.time()
	X_train_features_0 = np.zeros((len(X_train), len(ellipses)))
	for i in range(len(ellipses)):
		args = {key:ellipses[i][key] for key in ['mean', 'std']}
		X_train_temp = [dgm[:,1] for dgm in X_train]
		X_train_features_0[:,i] = feature(X_train_temp, f_gaussian, **args)
	
	print(X_train_features_0.shape)

	X_test_features_0 = np.zeros((len(X_test), len(ellipses)))
	for i in range(len(ellipses)):
		args = {key:ellipses[i][key] for key in ['mean', 'std']}
		X_test_temp = [dgm[:,1] for dgm in X_test]

		X_test_features_0[:,i] = feature(X_test_temp, f_gaussian, **args)

	t1 = time.time()
	print('Features H_0:{}'.format(t1-t0))


	# ------------------------------ H_1 ------------------------------------------

	X_train, X_test, F_train, F_test = train_test_split(X_dgm1, F, test_size=0.33, random_state=10)
	
	# ------------------------------ CDER -----------------------------------------

	# Recall that CDER has problems with H_0. I need to put a little noise on the birth time.

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
		#temp['std'] = np.array([temp['radius'], temp['radius']])
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

	# ------------------------------ CDER features --------------------------------

	X_train_features_1 = np.zeros((len(X_train), len(ellipses)))
	for i in range(len(ellipses)):
		args = {key:ellipses[i][key] for key in ['mean', 'std', 'rotation']}
		args['center'] = args.pop('mean')
		args['axis'] = args.pop('std')

		X_train_features_1[:,i] = feature(X_train, f_ellipse, **args)

	X_test_features_1 = np.zeros((len(X_test), len(ellipses)))
	for i in range(len(ellipses)):
		args = {key:ellipses[i][key] for key in ['mean', 'std', 'rotation']}
		args['center'] = args.pop('mean')
		args['axis'] = args.pop('std')

		X_test_features_1[:,i] = feature(X_test, f_ellipse, **args)

	# ------------------------------ Ridge Classification  ------------------------

	X_train_features = np.column_stack((X_train_features_0, X_train_features_1))
	print(X_train_features.shape)
	X_test_features = np.column_stack((X_test_features_0, X_test_features_1))

	ridge_model = RidgeClassifier().fit(X_train_features, F_train)
	score_train.append(ridge_model.score(X_train_features, F_train))
	score_test.append(ridge_model.score(X_test_features, F_test))

	print('train', score_train[-1])
	print('test', score_test[-1])

print(np.mean(score_train), np.std(score_train))
print(np.mean(score_test), np.std(score_test))

sys.exit()


# ------------------------------ Gaussian Mixture Model -----------------------



X_train_temp = np.vstack(X_train)

gmm = mixture.BayesianGaussianMixture(n_components=len(ellipses), covariance_type='full', max_iter=int(10e4)).fit(X_train_temp)


gmm_ellipses = []
for i in range(len(gmm.means_)):
	u,s,v = np.linalg.svd(gmm.covariances_[i])
	temp = {'mean':gmm.means_[i], 'std':s, 'rotation':v, 'radius':max(s), 'entropy':gmm.weights_[i]}
	gmm_ellipses.append(temp)


# ------------------------------ HDBSCAN --------------------------------------


clusterer = hdbscan.HDBSCAN()

clusterer.fit(X_train_temp)

num_clusters = clusterer.labels_.max()

hdbscan_ellipses = []
for i in range(num_clusters):
	cluster_i = X_train_temp[clusterer.labels_ == i]

	en = np.mean(clusterer.probabilities_[clusterer.labels_ == i])
	
	mean = np.mean(cluster_i, axis=0)
	cov_matrix = np.cov(cluster_i.transpose())

	u,s,v = np.linalg.svd(cov_matrix, full_matrices=True)

	temp = {'mean':mean, 'std':s, 'rotation':u, 'radius':max(s), 'entropy':en}
	hdbscan_ellipses.append(temp)






ellipses_plot_gmm = []
for i in range(len(gmm_ellipses)):
	e = gmm_ellipses[i]
	ellipses_plot_gmm.append(Ellipse(xy=e['mean'], width=e['std'][0], height=e['std'][1], angle=np.arccos(e['rotation'][0,0])))

fig, ax = plt.subplots()

for i in range(len(X_train)):
	dgm = np.array(X_train[i][:, 0:2])
	plt.scatter(dgm[:,0], dgm[:,1], color='grey')



for e in ellipses_plot_gmm:
	ax.add_artist(e)
	e.set_clip_box(ax.bbox)
	e.set_alpha(0.5)
	e.set_facecolor([0,0,1])

plt.savefig('cder_ellipses.png')

sys.exit()