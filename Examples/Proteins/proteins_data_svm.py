import numpy as np
import multidim
import itertools
import os
import hdbscan
import sys
import time
import pandas as pd
import itertools
import pickle

from copy import deepcopy
from matplotlib.patches import Ellipse
from ripser import ripser
from persim import plot_diagrams
from numba import jit, njit, prange
from sklearn import mixture
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC, SVC

from multidim.covertree import CoverTree
from multidim.models import CDER

import matplotlib.pyplot as plt

np.set_printoptions(precision=2)

sys.path.append('../..')
from ATS import *

# -----------------------------------------------------------------------------
# -------------- ARGUMENTS ----------------------------------------------------
# -----------------------------------------------------------------------------

adptative_feature = str(sys.argv[1])

kernel = str(sys.argv[2])

# -----------------------------------------------------------------------------
# -------------- Classification parameters ------------------------------------
# -----------------------------------------------------------------------------

regularization_constants = range(-4,15,2)

degrees = range(1,11)

# -----------------------------------------------------------------------------
# -------------- Run all classification problems ------------------------------
# -----------------------------------------------------------------------------

all_train_score = np.zeros( (55, len(regularization_constants), len(degrees)) )
all_test_score = np.zeros( (55, len(regularization_constants), len(degrees)) )

for k in range(55):
	# -----------------------------------------------------------------------------
	# -------------- IMPORT DATA --------------------------------------------------
	# -----------------------------------------------------------------------------

	Data = pickle.load( open( "diagrams.pickle", "rb" ) )

	train_index = np.load('./Index/TrainIndex{}.npy'.format(k+1))

	test_index = np.load('./Index/TestIndex{}.npy'.format(k+1))

	train_label = np.load('./Index/TrainLabel{}.npy'.format(k+1))

	test_label = np.load('./Index/TestLabel{}.npy'.format(k+1))

	X_dgm0 = [d['h1'] for d in Data]

	X_dgm1 = [d['h1'] for d in Data]

	# Lets change the birth-death to birth-persistence
	for j in range(len(X_dgm1)):
		temp_dgm = X_dgm1[j]

		temp_dgm[:,1] = temp_dgm[:,1] - temp_dgm[:,0]

		temp_dgm[np.isclose(temp_dgm, 0, rtol=1e-05, atol=1e-05)] = 1e-05

		temp_dgm = np.log(temp_dgm)

		X_dgm1[j] = temp_dgm

	X_train_0 = [X_dgm0[ind] for ind in train_index]
	X_test_0 = [X_dgm0[ind] for ind in test_index]

	X_train_1 = [X_dgm1[ind] for ind in train_index]
	X_test_1 = [X_dgm1[ind] for ind in test_index]

	F_train = train_label
	F_test = test_label

	d = 5

	# -----------------------------------------------------------------------------
	# ------------------------------ H_0 ------------------------------------------
	# -----------------------------------------------------------------------------

	# -----------------------------------------------------------------------------
	# ------------------------------ GMM ------------------------------------------
	# -----------------------------------------------------------------------------

	print('Begin GMM...')
	t0 = time.time()
	X_train_temp = np.log(np.vstack(X_train_0))

	X_train_temp = X_train_temp[:,1]

	X_train_temp = X_train_temp.reshape((-1,1))

	gmm_f_train=[]
	for i in range(len(X_train_0)):
		gmm_f_train.append(F_train[i]*np.ones(len(X_train_0[i])))
	gmm_f_train = np.concatenate(gmm_f_train)

	gmm = mixture.BayesianGaussianMixture(n_components=d, max_iter=int(10e4)).fit(X_train_temp, gmm_f_train)

	ellipses = []
	for i in range(len(gmm.means_)):
		L, v = np.linalg.eig(gmm.covariances_[i])
		temp = {'mean':gmm.means_[i], 'std':np.sqrt(L), 'rotation':v.transpose(), 'radius':max(np.sqrt(L)), 'entropy':gmm.weights_[i]}
		ellipses.append(temp)
	t1 = time.time()
	print('Finish GMM. Time: {}'.format(t1-t0))

	# -----------------------------------------------------------------------------
	# ------------------------------ Features -------------------------------------
	# -----------------------------------------------------------------------------

	t0 = time.time()
	
	X_train_temp = [dgm[:,1] for dgm in X_train_0]
	X_train_features_0 = get_all_features(X_train_temp, ellipses, f_gaussian)

	X_test_temp = [dgm[:,1] for dgm in X_test_0]
	X_test_features_0 = get_all_features(X_test_temp, ellipses, f_gaussian)

	t1 = time.time()
	print('Features H_0:{}'.format(t1-t0))

	# -----------------------------------------------------------------------------
	# ------------------------------ H_1 ------------------------------------------
	# -----------------------------------------------------------------------------

	if adptative_feature=='cder':
		# ----------------------------------------------------------------------------
		# ------------------------------ CDER ----------------------------------------
		# ----------------------------------------------------------------------------

		print('Begin CDER...')
		t0 = time.time()
		pc_train = multidim.PointCloud.from_multisample_multilabel(X_train_1, F_train)
		ct_train = CoverTree(pc_train)

		cder = CDER(parsimonious=True)

		cder.fit(ct_train)

		cder_result = cder.gaussians

		ellipses = []
		for c in cder_result:
			temp = {key:c[key] for key in ['mean', 'std', 'rotation', 'radius', 'entropy']}
			temp['std'] = 2*temp['std']
			#temp['std'] = np.array([temp['radius'], temp['radius']])
			ellipses.append(temp)

		t1 = time.time()
		print('Finish CDER. Time: {}'.format(t1-t0))

	if adptative_feature=='gmm':
		# ----------------------------------------------------------------------------
		# ------------------------------ GMM -----------------------------------------
		# ----------------------------------------------------------------------------
		print('Begin GMM...')
		t0 = time.time()
		X_train_temp = np.vstack(X_train_1)

		gmm_f_train=[]
		for i in range(len(X_train_1)):
			gmm_f_train.append(F_train[i]*np.ones(len(X_train_1[i])))
		gmm_f_train = np.concatenate(gmm_f_train)

		gmm = mixture.BayesianGaussianMixture(n_components=d*d, max_iter=int(10e4)).fit(X_train_temp, gmm_f_train)

		ellipses = []
		for i in range(len(gmm.means_)):
			L, v = np.linalg.eig(gmm.covariances_[i])
			temp = {'mean':gmm.means_[i], 'std':np.sqrt(L), 'rotation':v.transpose(), 'radius':max(np.sqrt(L)), 'entropy':gmm.weights_[i]}
			temp['std'] = 3*temp['std']
			ellipses.append(temp)
		t1 = time.time()
		print('Finish GMM. Time: {}'.format(t1-t0))

	if adptative_feature=='hdbscan':
		# ----------------------------------------------------------------------------
		# ------------------------------ HDBSCAN -------------------------------------
		# ----------------------------------------------------------------------------
		print('Begin HDBSCAN...')
		t0 = time.time()
		X_train_temp = np.vstack(X_train_1)

		clusterer = hdbscan.HDBSCAN(min_samples=d*d, min_cluster_size=d*d)

		clusterer.fit(X_train_temp)

		num_clusters = clusterer.labels_.max()

		ellipses = []
		for i in range(num_clusters):
			cluster_i = X_train_temp[clusterer.labels_ == i]

			en = np.mean(clusterer.probabilities_[clusterer.labels_ == i])
			
			mean = np.mean(cluster_i, axis=0)
			cov_matrix = np.cov(cluster_i.transpose())

			L,v = np.linalg.eig(cov_matrix)

			temp = {'mean':mean, 'std':np.sqrt(L), 'rotation':v.transpose(), 'radius':max(np.sqrt(L)), 'entropy':en}
			temp['std'] = 3*temp['std']
			ellipses.append(temp)

		t1 = time.time()
		print('Finish HDBSCAN. Time: {}'.format(t1-t0))


	# ----------------------------------------------------------------------------
	# ------------------------------ Features ------------------------------------
	# ----------------------------------------------------------------------------
	t0 = time.time()
	
	X_train_features_1 = get_all_features(X_train_1, ellipses, f_ellipse)

	X_test_features_1 = get_all_features(X_test_1, ellipses, f_ellipse)
	
	t1 = time.time()
	print('Features H_1:{}'.format(t1-t0))

	X_train_features = np.column_stack((X_train_features_0, X_train_features_1))

	X_test_features = np.column_stack((X_test_features_0, X_test_features_1))


	# ----------------------------------------------------------------------------
	# ------------------------------ Classification  -----------------------------
	# ----------------------------------------------------------------------------

	score_train = np.zeros(( len(regularization_constants), len(degrees) ))
	score_test = np.zeros(( len(regularization_constants), len(degrees) ))

	position_constant = 0
	for regularization_c in regularization_constants:
		print(regularization_c)
		position_degree = 0 
		for poly_degree in degrees:
			print(poly_degree)
			poly = PolynomialFeatures(1)

			X_train_features_transformed = poly.fit_transform(X_train_features)

			X_test_features_transformed = poly.fit_transform(X_test_features)

			if kernel=='sigmoid':
				ridge_model = SVC(kernel='sigmoid', degree=1, coef0=0.0 ,C=10**(-1*regularization_c), gamma=10**(-1*poly_degree)).fit(X_train_features_transformed, F_train)

			if kernel=='rbf':
				ridge_model = SVC(kernel='rbf', degree=1, coef0=0.0 ,C=10**(-1*regularization_c), gamma=10**(-1*poly_degree)).fit(X_train_features_transformed, F_train)

			if kernel=='poly':
				ridge_model = SVC(kernel='poly', degree=poly_degree, coef0=1.0 ,C=10**(-1*regularization_c), gamma='auto').fit(X_train_features_transformed, F_train)

			F_train_predicted = ridge_model.predict(X_train_features_transformed)

			F_test_predicted = ridge_model.predict(X_test_features_transformed)
			
			score_train[position_constant, position_degree] = sum(F_train_predicted==F_train)/len(F_train_predicted)
			
			score_test[position_constant, position_degree] = sum(F_test_predicted==F_test)/len(F_test_predicted)
			
			position_degree += 1

		position_constant +=1

	all_train_score[k,:,:] = score_train
	all_test_score[k,:,:] = score_test

train_score_mean = np.mean(all_train_score, 0)
train_score_std = np.std(all_train_score, 0)

test_score_mean = np.mean(all_test_score, 0)
test_score_std = np.std(all_test_score, 0)

# ----------------------------------------------------------------------------
# ------------------------------ Save figure ---------------------------------
# ----------------------------------------------------------------------------
os.system('mkdir svm_results')

with open('svm_results/train_score_{}_{}.pickle'.format(kernel, adptative_feature), 'wb') as handle:
    pickle.dump(all_train_score, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('svm_results/test_score_{}_{}.pickle'.format(kernel, adptative_feature), 'wb') as handle:
    pickle.dump(all_test_score, handle, protocol=pickle.HIGHEST_PROTOCOL)

regularization_constants = np.array(regularization_constants)
poly_degree = np.array(poly_degree)

_min, _max = 0, 1

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10*2,10*1), dpi=100)
	
im = ax[0].imshow(train_score_mean, vmin = _min, vmax = _max)
ax[0].set_title('Train score')

# set latels for x-axis on plot 
ax[0].set_yticks(range(0,len(regularization_constants)))
ax[0].set_yticklabels(regularization_constants)
ax[0].set_ylabel('Regilarization')

# set latels for y-axis on plot 
ax[0].set_xticks(range(0,len(degrees)))
ax[0].set_xticklabels(degrees)
ax[0].set_xlabel('Poly kernel degree')

for (k,j),label in np.ndenumerate(train_score_mean):
	annotation  = str(round(label,2)) + ' + ' + str(round(train_score_std[k,j],2))
	ax[0].text(j, k, annotation, ha='center', va='center', rotation=45)


im = ax[1].imshow(test_score_mean, vmin = _min, vmax = _max)
ax[1].set_title('Train score')

# set latels for x-axis on plot 
ax[1].set_yticks(range(0,len(regularization_constants)))
ax[1].set_yticklabels(regularization_constants)
ax[1].set_ylabel('Regilarization')

# set latels for y-axis on plot 
ax[1].set_xticks(range(0,len(degrees)))
ax[1].set_xticklabels(degrees)
ax[1].set_xlabel('Poly kernel degree')

for (k,j),label in np.ndenumerate(test_score_mean):
	annotation  = str(round(label,2)) + ' + ' + str(round(test_score_std[k,j],2))
	ax[1].text(j, k, annotation, ha='center', va='center', rotation=45)

fig.colorbar(im, ax=ax.ravel().tolist())

plt.savefig('svm_results/{}_{}.png'.format(kernel, adptative_feature))
