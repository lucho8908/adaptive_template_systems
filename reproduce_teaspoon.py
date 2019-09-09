import numpy as np
import multidim
import itertools
import os
import hdbscan
import teaspoon
import sys
import pandas as pd

from teaspoon.MakeData import PointCloud
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

from approximation import *

# ------------------------------ IMPORT DATA ----------------------------------

num_dgms = int(sys.argv[1])

N = num_dgms*6

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
	print(F_test)
	# ------------------------------ BOX ------------------------------------------

	centers, delta = box_centers(X_train, 10, 0.05)

	if n == 9:
		for i in range(len(X_dgm0)):
			plot_diagrams(X_dgm0[i], lifetime=True)
			plt.gca().get_legend().remove()
		
		for c in centers:
			plt.axvline(x=c[0]-delta, color='r', linestyle='-')
			plt.axvline(x=c[0]+delta, color='r', linestyle='-')
			plt.axhline(y=c[1]-delta, color='r', linestyle='-')
			plt.axhline(y=c[1]+delta, color='r', linestyle='-')
		plt.savefig('h0_box.png')
		plt.close()

	# ------------------------------ BOX features ---------------------------------

	X_train_features_0 = np.zeros((len(X_train), len(centers)))
	for i in range(len(centers)):
		args = {'center':centers[i], 'delta':delta}
		X_train_features_0[:,i] = feature(X_train, f_box, **args)

	X_test_features_0 = np.zeros((len(X_test), len(centers)))
	for i in range(len(centers)):
		args = {'center':centers[i], 'delta':delta}
		X_test_features_0[:,i] = feature(X_test, f_box, **args)

	# ------------------------------ H_1 ------------------------------------------

	X_train, X_test, F_train, F_test = train_test_split(X_dgm1, F, test_size=0.33, random_state=10)
	print(F_test)
	# ------------------------------ BOX ------------------------------------------

	centers, delta = box_centers(X_train, 10, 0.05)

	if n == 9:
		for i in range(len(X_dgm1)):
			plot_diagrams(X_dgm1[i], lifetime=True)
			plt.gca().get_legend().remove()
		
		for c in centers:
			plt.axvline(x=c[0]-delta, color='r', linestyle='-')
			plt.axvline(x=c[0]+delta, color='r', linestyle='-')
			plt.axhline(y=c[1]-delta, color='r', linestyle='-')
			plt.axhline(y=c[1]+delta, color='r', linestyle='-')
		plt.savefig('h1_box.png')
		plt.close()

	# ------------------------------ BOX features ---------------------------------

	X_train_features_1 = np.zeros((len(X_train), len(centers)))
	for i in range(len(centers)):
		args = {'center':centers[i], 'delta':delta}
		X_train_features_1[:,i] = feature(X_train, f_box, **args)

	X_test_features_1 = np.zeros((len(X_test), len(centers)))
	for i in range(len(centers)):
		args = {'center':centers[i], 'delta':delta}
		X_test_features_1[:,i] = feature(X_test, f_box, **args)

	# ------------------------------ Ridge Classification  ------------------------

	X_train_features = np.column_stack((X_train_features_0, X_train_features_1))
	print(X_train_features.shape)
	X_test_features = np.column_stack((X_test_features_0, X_test_features_1))

	ridge_model = RidgeClassifier().fit(X_train_features, F_train)
	score_train.append(ridge_model.score(X_train_features, F_train))
	score_test.append(ridge_model.score(X_test_features, F_test))

print(np.mean(score_train), np.std(score_train))
print(np.mean(score_test), np.std(score_test))

sys.exit()

