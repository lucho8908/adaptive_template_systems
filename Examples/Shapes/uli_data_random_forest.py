import numpy as np
import multidim
import itertools
import os
import hdbscan
import sys
import time
import pandas as pd
import itertools

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
from sklearn.ensemble import RandomForestClassifier

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
# -------------- IMPORT DATA --------------------------------------------------
# -----------------------------------------------------------------------------

Data = pd.read_csv('Uli_data/Uli_data.csv')

# Code to reshape the data in the groupby command below
def reshapeVec(g):
    A = np.array([g.dim,g.birth,g.death])
    A = A.T
    return A

DgmsDF = Data.groupby(['freq', 'trial']).apply(reshapeVec)
DgmsDF = DgmsDF.reset_index()
DgmsDF = DgmsDF.rename(columns = {0:'CollectedDgm'})

def getDgm(A, dim = 0):
    if type(dim) != str:
        A = A[np.where(A[:,0] == dim)[0],1:]
    elif dim == 'essential':
        A = A[np.where(A[:,0] <0)[0],:]
    return(A)

DgmsDF['Dgm1'] = DgmsDF.CollectedDgm.apply(lambda x: getDgm(x, dim = 1))
DgmsDF['Dgm0'] = DgmsDF.CollectedDgm.apply(lambda x: getDgm(x, dim = 0))
DgmsDF['DgmInf'] = DgmsDF.CollectedDgm.apply(lambda x: getDgm(x, dim = 'essential'))

def label(index):
    if 0 <= index <= 19:
        return 'male_neutral'
    elif 20<= index <=39:
        return 'male_bodybuilder'
    elif 40<= index <=59:
        return 'male_fat'
    elif 60<= index <=79:
        return 'male_thin'
    elif 80<= index <=99:
        return 'male_average'
    elif 100<= index <=119:
        return 'female_neutral'
    elif 120<= index <=139:
        return 'female_bodybuilder'
    elif 140<= index <=159:
        return 'female_fat'
    elif 160<= index <=179:
        return 'female_thin'
    elif 180<= index <=199:
        return 'female_average'
    elif 200<= index <=219:
        return 'child_neutral'
    elif 220<= index <=239:
        return 'child_bodybuilder'
    elif 240<= index <=259:
        return 'child_fat'
    elif 260<= index <=279:
        return 'child_thin'
    elif 280<= index <=299:
        return 'child_average'
    else:
        print('What are you giving me?')

DgmsDF['TrainingLabel'] = DgmsDF.freq.apply(label)
DgmsDF= DgmsDF.sample(frac=1)

score_train_per_frequency = []
score_test_per_frequency = []

for i in range(1,11):

	freq = i

	SampleDF = DgmsDF[DgmsDF.trial == freq].sample(frac=1)

	X_dgm0 = SampleDF['Dgm0'].tolist()

	X_dgm1 = SampleDF['Dgm1'].tolist()

	# Lets change the birth-death to birth-persistence

	for j in range(len(X_dgm1)):
		temp_dgm = X_dgm1[j]

		temp_dgm[:,1] = temp_dgm[:,1] - temp_dgm[:,0]

		temp_dgm[np.isclose(temp_dgm, 0, rtol=1e-05, atol=1e-05)] = 1e-05

		temp_dgm = np.log(temp_dgm)

		X_dgm1[j] = temp_dgm


	labels = list(set(SampleDF.TrainingLabel))
	labels.sort()

	mapping = {}
	for i in range(len(labels)):
		mapping[labels[i]] = i

	SampleDF = SampleDF.replace({'TrainingLabel': mapping})

	F = SampleDF['TrainingLabel'].tolist()

	d = 10

	score_train_per_iteration = []
	score_test_per_iteration = []

	for rep in range(10):
	
		X_train_0, X_test_0, X_train_1, X_test_1, F_train, F_test = train_test_split(X_dgm0, X_dgm1, F, test_size=0.30)

		# ------------------------------ H_0 ------------------------------------------

		# ------------------------------ GMM ------------------------------------------

		print('Begin GMM...')
		t0 = time.time()
		X_train_temp = np.vstack(X_train_0)

		X_train_temp = X_train_temp[:,1]

		X_train_temp = X_train_temp.reshape((-1,1))

		gmm_f_train=[]
		for i in range(len(X_train_0)):
			gmm_f_train.append(F_train[i]*np.ones(len(X_train_0[i])))
		gmm_f_train = np.concatenate(gmm_f_train)

		gmm = mixture.BayesianGaussianMixture(n_components=d, covariance_type='full', max_iter=int(10e4)).fit(X_train_temp, gmm_f_train)

		ellipses = []
		for i in range(len(gmm.means_)):
			L, v = np.linalg.eig(gmm.covariances_[i])
			temp = {'mean':gmm.means_[i], 'std':np.sqrt(L), 'rotation':v.transpose(), 'radius':max(np.sqrt(L)), 'entropy':gmm.weights_[i]}
			ellipses.append(temp)
		t1 = time.time()
		print('Finish GMM. Time: {}'.format(t1-t0))

		# ------------------------------ Features -------------------------------------
		t0 = time.time()
		X_train_features_0 = np.zeros((len(X_train_0), len(ellipses)))
		for i in range(len(ellipses)):
			args = {key:ellipses[i][key] for key in ['mean', 'std']}
			X_train_temp = [dgm[:,1] for dgm in X_train_0]
			X_train_features_0[:,i] = feature(X_train_temp, f_gaussian, **args)


		X_test_features_0 = np.zeros((len(X_test_0), len(ellipses)))
		for i in range(len(ellipses)):
			args = {key:ellipses[i][key] for key in ['mean', 'std']}
			X_test_temp = [dgm[:,1] for dgm in X_test_0]

			X_test_features_0[:,i] = feature(X_test_temp, f_gaussian, **args)

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
		X_train_features_1 = np.zeros((len(X_train_1), len(ellipses)))
		for i in range(len(ellipses)):
			args = {key:ellipses[i][key] for key in ['mean', 'std', 'rotation']}
			args['center'] = args.pop('mean')
			args['axis'] = args.pop('std')

			X_train_features_1[:,i] = feature(X_train_1, f_ellipse, **args)

		X_test_features_1 = np.zeros((len(X_test_1), len(ellipses)))
		for i in range(len(ellipses)):
			args = {key:ellipses[i][key] for key in ['mean', 'std', 'rotation']}
			args['center'] = args.pop('mean')
			args['axis'] = args.pop('std')

			X_test_features_1[:,i] = feature(X_test_1, f_ellipse, **args)
		t1 = time.time()
		print('Features H_1:{}'.format(t1-t0))

		X_train_features = np.column_stack((X_train_features_0, X_train_features_1))

		X_test_features = np.column_stack((X_test_features_0, X_test_features_1))

		# ----------------------------------------------------------------------------
		# ------------------------------ Classification  -----------------------------
		# ----------------------------------------------------------------------------

		regularization_constants = range(0,20)
		
		degrees = range(1,202,10)

		score_train = np.zeros(( len(regularization_constants), len(degrees) ))
		score_test = np.zeros(( len(regularization_constants), len(degrees) ))

		position_constant = 0
		for regularization_c in regularization_constants:
			print(regularization_c)
			position_degree = 0 
			for poly_degree in degrees:
				print(poly_degree)
				
				X_train_features_transformed = X_train_features

				X_test_features_transformed = X_test_features

				ridge_model = RandomForestClassifier(criterion='entropy', min_impurity_decrease=10**(-1*regularization_c), n_estimators=poly_degree , n_jobs=16).fit(X_train_features_transformed, F_train)

				score_train[position_constant, position_degree] = ridge_model.score(X_train_features_transformed, F_train)
				
				score_test[position_constant, position_degree] = ridge_model.score(X_test_features_transformed, F_test)

				position_degree += 1

			position_constant +=1

		score_train_per_iteration.append(score_train)
		score_test_per_iteration.append(score_test)
		
	score_train_per_frequency.append(score_train_per_iteration)
	score_test_per_frequency.append(score_test_per_iteration)

score_train_per_frequency = np.array(score_train_per_frequency)
score_test_per_frequency = np.array(score_test_per_frequency)

train_means = np.mean(score_train_per_frequency, 1)
test_means = np.mean(score_test_per_frequency, 1)

train_stds = np.std(score_train_per_frequency, 1)
test_stds = np.std(score_test_per_frequency, 1)

# -----------------------------------------------------------------------------
# ----------- Save Figures ----------------------------------------------------
# -----------------------------------------------------------------------------
os.system('mkdir rforest_results')

for i in range(train_means.shape[0]):
	train_m = train_means[i,:,:]
	train_s = train_stds[i,:,:]

	test_m = test_means[i,:,:]
	test_s = test_stds[i,:,:]

	# Combine all data
	combined_data = np.array([train_m, test_m])
	# Get the min and max of all your data
	_min, _max = np.amin(combined_data), np.amax(combined_data)

	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10*2,10*1), dpi=100)
	
	im = ax[0].imshow(train_m, vmin = _min, vmax = _max)
	ax[0].set_title('Train score')

	# set latels for x-axis on plot 
	ax[0].set_yticks(range(0,len(regularization_constants)))
	ax[0].set_yticklabels(regularization_constants)
	ax[0].set_ylabel('Entropy split 10^(-i)')

	# set latels for y-axis on plot 
	ax[0].set_xticks(range(0,len(degrees)))
	ax[0].set_xticklabels(degrees)
	ax[0].set_xlabel('Num. estimators')

	# set color bar in each plot
	# fig.colorbar(im, ax=ax[0])

	# include the value of the matrix in each entry of the plot
	for (k,j),label in np.ndenumerate(train_m):
		annotation  = str(round(label,2)) + ' + ' + str(round(train_s[k,j],2))
		ax[0].text(j, k, annotation, ha='center', va='center', rotation=45)


	im = ax[1].imshow(test_m, vmin = _min, vmax = _max)
	ax[1].set_title('Test score')

	# set latels for x-axis on plot 
	ax[1].set_yticks(range(0,len(regularization_constants)))
	ax[1].set_yticklabels(regularization_constants)
	ax[1].set_ylabel('Regilarization')

	# set latels for y-axis on plot 
	ax[1].set_xticks(range(0,len(degrees)))
	ax[1].set_xticklabels(degrees)
	ax[1].set_xlabel('Poly kernel degree')

	# set color bar in each plot
	# fig.colorbar(im, ax=ax[1])

	fig.colorbar(im, ax=ax.ravel().tolist())

	# include the value of the matrix in each entry of the plot
	for (k,j),label in np.ndenumerate(test_m):
		annotation  = str(round(label,2)) + ' + ' + str(round(test_s[k,j],2))
		ax[1].text(j, k, annotation, ha='center', va='center', rotation=45)


	best = np.unravel_index(np.argmin(np.abs(train_m - test_m), axis=None), train_m.shape)

	# add title to the figure
	plt.suptitle('Frequency {} \n {} // {} \n {}'.format(i, train_m[best], test_m[best], best) )
	
	plt.savefig('rforest_results/f_{}_gmm.png'.format(i))
