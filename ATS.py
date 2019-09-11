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


def Circle(N = 100, r=1, gamma=None, seed = None):
	'''
	Generate N points in R^2 from the circle centered
	at the origin with radius r.

	If gamma != None, then we add normal noise in the normal direction with std dev gamma.

	:param N: Number of points to generate
	:type N: int
	
	:param r: Radius of the circle
	:type r: float

	:param gamma: Stan dard deviation of the normally distributed noise. 
	:type gamma: float or 'None'

	:param seed: Fixes the seed.  Good if we want to replicate results.
	:type seed: float or 'None'

	:return: numpy 2D array -- A Nx2 numpy array with the points drawn as the rows.

	'''
	np.random.seed(seed)
	theta = np.random.rand(N,1)
	theta = theta.reshape((N,))
	P = np.zeros([N,2])

	P[:,0] = r*np.cos(2*np.pi*theta)
	P[:,1] = r*np.sin(2*np.pi*theta)

	if gamma is not None:
		noise = np.random.normal(0, gamma, size=(N,2))
		P += noise

	return P


def Sphere(N = 100, r = 1, noise = 0, seed = None):
	'''
	Generate N points in R^3 from the sphere centered
	at the origin with radius r.
	If noise is set to a positive number, the points
	can be at distance r +/- noise from the origin.

	:param N: Number of points to generate
	:type N: int

	:param r: Radius of the sphere
	:type r: float

	:param seed: Fixes the seed.  Good if we want to replicate results.
	:type seed: float

	:return: Numpy 2D array -- A Nx3 numpy array with the points drawn as the rows.

	'''
	np.random.seed(seed)

	R = 2*noise*np.random.random(N) + r
	theta =   np.pi * np.random.random(N)
	phi = 2 * np.pi * np.random.random(N)

	P = np.zeros((N,3))
	P[:,0] = R * np.sin(theta) * np.cos(phi)
	P[:,1] = R * np.sin(theta) * np.sin(phi)
	P[:,2] = R * np.cos(theta)

	return P

def Annulus(N=200,r=1,R=2, seed = None):
	'''
	Returns point cloud sampled from uniform distribution on
	annulus in R^2 of inner radius r and outer radius R

	:param N: Number of points to generate
	:type N: int

	:param r: Inner radius of the annulus
	:type r: float

	:param R: Outer radius of the annulus
	:type R: float

	:param seed: Fixes the seed.  Good if we want to replicate results.
	:type seed: float

	:return: Numpy 2D array -- A Nx2 numpy array with the points drawn as the rows.

	'''
	np.random.seed(seed)
	P = np.random.uniform(-R,R,[2*N,2])
	S = P[:,0]**2 + P[:,1]**2
	P = P[np.logical_and(S>= r**2, S<= R**2)]
	#print np.shape(P)

	while P.shape[0]<N:
		Q = np.random.uniform(-R,R,[2*N,2])
		S = Q[:,0]**2 + Q[:,1]**2
		Q = Q[np.logical_and(S>= r**2, S<= R**2)]
		P = np.append(P,Q,0)
		#print np.shape(P)

	return P[:N,:]


def Torus(N = 100, r = 1,R = 2,  seed = None):
	'''
	Generates torus with points
	x = ( R + r*cos(theta) ) * cos(psi),
	y = ( R + r*cos(theta) ) * sin(psi),
	z = r * sin(theta)

	:param N: Number of points to generate
	:type N: int

	:param r: Inner radius of the torus
	:type r: float

	:param R: Outer radius of the torus
	:type R: float

	:param seed: Fixes the seed.  Good if we want to replicate results.
	:type seed: float

	:return: numpy 2D array -- A Nx3 numpy array with the points drawn as the rows.

	'''

	np.random.seed(seed)
	psi = np.random.rand(N,1)
	psi = 2*np.pi*psi

	outputTheta = []
	while len(outputTheta)<N:
		theta = np.random.rand(2*N,1)
		theta = 2*np.pi*theta

		eta = np.random.rand(2*N,1)
		eta = eta / np.pi

		fx = (1+ r/float(R)*np.cos(theta)) / (2*np.pi)

		outputTheta = theta[eta<fx]


	theta = outputTheta[:N]
	theta = theta.reshape(N,1)


	x = ( R + r*np.cos(theta) ) * np.cos(psi)
	y = ( R + r*np.cos(theta) ) * np.sin(psi)
	z = r * np.sin(theta)
	x = x.reshape((N,))
	y = y.reshape((N,))
	z = z.reshape((N,))

	P = np.zeros([N,3])
	P[:,0] = x
	P[:,1] = y
	P[:,2] = z

	return P

def Cube(N = 100, diam = 1, dim = 2, seed = None):
	'''
	Generate N points in the box [0,diam]x[0,diam]x...x[0,diam]

	:param N: Number of points to generate
	:type N: int

	:param diam: lenght of one side of the box
	:type diam: float

	:param dim: Dimension of the box; point are embbeded in R^dim
	:type dim: int

	:return: numpy array -- A Nxdim numpy array with the points drawn as the rows.

	'''
	np.random.seed(seed)

	P = diam*np.random.random((N,dim))

	return P


def Clusters(N = 100, centers = np.array(((0,0),(3,3))), sd = 1, seed = None):
	'''
	Generate k clusters of points, `N` points in total. k is the number of centers.

	:param N: Number of points to be generated
	:type N: int 

	:param centers: k x d numpy array, where centers[i,:] is the center of the ith cluster in R^d.
	:type centers: numpy array

	:param sd: standard deviation of clusters.
	:type sd: numpy array

	:param seed: Fixes the seed.
	:type seed: float

	:return: numpy array -- A Nxd numpy array with the points drawn as the rows.

	'''

	np.random.seed(seed)


	# Dimension for embedding
	d = np.shape(centers)[1]

	# Identity matrix for covariance
	I = sd * np.eye(d)


	# Number of clusters
	k = np.shape(centers)[0]

	ptsPerCluster = N//k
	ptsForLastCluster = N//k + N%k

	for i in range(k):
		if i == k-1:
			newPts = np.random.multivariate_normal(centers[i,:], I, ptsForLastCluster)
		else:
			newPts = np.random.multivariate_normal(centers[i,:], I, ptsPerCluster)

		if i == 0:
			P = newPts[:]
		else:
			P = np.concatenate([P,newPts])

	return P


def testSetManifolds(numDgms = 50, numPts = 300, permute = True, seed = None):
	'''
	This function generates persisten diaglams from point clouds generated from the following collection of manifolds

		- Torus
		- Annulus
		- Cube
		- 3 clusters
		- 3 clusters of 3 clusters
		- Sphere

	The diagrmas are obtained by computing persistent homology (using Ripser) of sampled point clouds from the described manifolds.

	:param numDgms: Number of diagrmas per manifold
	:type numDgms: int

	:param numPts: Number of point per sampled point cloud
	:type numPts: int

	:param permute: If True it will permute the final result, so that diagrams of point clouds sampled from the samw manifold are not contiguous.
	:type permute: bool

	:param seed: Fixes the random seed.
	:type seed: float

	:return: pandas data frame -- Each row corersponds to the 0- and 1-dimensional persistent homology of a point cloud sampled from one of the 6 manifolds.
	'''
	
	columns = ['Dgm0', 'Dgm1', 'trainingLabel']
	index = range(6*numDgms)
	DgmsDF = pd.DataFrame(columns = columns, index = index)

	counter = 0

	if type(seed) == int:
		fixSeed = True
	else:
		fixSeed = False

	#-
	print('Generating torus clouds...')
	for i in range(numDgms):
		if fixSeed:
			seed += 1
		dgmOut = ripser(Torus(N=numPts,seed = seed))['dgms']
		DgmsDF.loc[counter] = [dgmOut[0],dgmOut[1], 'Torus']
		counter +=1

	#-
	print('Generating annuli clouds...')
	for i in range(numDgms):
		if fixSeed:
			seed += 1
		dgmOut = ripser(Annulus(N=numPts,seed = seed))['dgms']
		DgmsDF.loc[counter] = [dgmOut[0],dgmOut[1], 'Annulus']
		counter +=1

	#-
	print('Generating cube clouds...')
	for i in range(numDgms):
		if fixSeed:
			seed += 1
		dgmOut = ripser(Cube(N=numPts,seed = seed))['dgms']
		DgmsDF.loc[counter] = [dgmOut[0],dgmOut[1], 'Cube']
		counter +=1

	#-
	print('Generating three cluster clouds...')
	centers = np.array( [ [0,0], [0,2], [2,0]  ])
	for i in range(numDgms):
		if fixSeed:
			seed += 1
		dgmOut = ripser(Clusters(centers=centers, N = numPts, seed = seed, sd = .05))['dgms']
		DgmsDF.loc[counter] = [dgmOut[0],dgmOut[1], '3Cluster']
		counter +=1

	#-
	print('Generating three clusters of three clusters clouds...')

	centers = np.array( [ [0,0], [0,1.5], [1.5,0]  ])
	theta = np.pi/4
	centersUp = np.dot(centers,np.array([(np.sin(theta),np.cos(theta)),(np.cos(theta),-np.sin(theta))])) + [0,5]
	centersUpRight = centers + [3,5]
	centers = np.concatenate( (centers,  centersUp, centersUpRight))
	for i in range(numDgms):
		if fixSeed:
			seed += 1
		dgmOut = ripser(Clusters(centers=centers,
										N = numPts,
										sd = .05,
										seed = seed))['dgms']
		DgmsDF.loc[counter] = [dgmOut[0],dgmOut[1], '3Clusters of 3Clusters']
		counter +=1

	#-
	print('Generating sphere clouds...')

	for i in range(numDgms):
		if fixSeed:
			seed += 1
		dgmOut = ripser(Sphere(N = numPts, noise = .05,seed = seed))['dgms']
		DgmsDF.loc[counter] = [dgmOut[0],dgmOut[1], 'Sphere']
		counter +=1

	print('Finished generating clouds and computing persistence.\n')

	# Permute the diagrams if necessary.
	if permute:
		DgmsDF = DgmsDF.reindex(np.random.permutation(DgmsDF.index))

	return DgmsDF

# ------------------------------ My feature functions -------------------------

def limits_box(list_dgms):
	'''
	This function computes the min and max of a collection of pers. dgms. in the birth-lifespan space.
	
	:param list_dgms: List of persistent diagrmas
	:type list_dgms: list

	:return: list -- mins = [min_birth, min_lifespan] and maxs = [max_birth, max_lifespan]
	'''

	list_dgms_temp = deepcopy(list_dgms)
	mins = np.inf*np.ones(2)
	maxs = -np.inf*np.ones(2)

	for dgm in list_dgms_temp:
		dgm[:,1] = dgm[:,1]-dgm[:,0] # Turns the birth-death into birth-lifespan
		mins = np.minimum(np.amin(dgm, axis=0), mins)
		maxs = np.maximum(np.amax(dgm, axis=0), maxs)

	return mins, maxs

def box_centers(list_dgms, d, padding):
	'''
	This function computes the collection of centers use to define tent functions, as well as the size the tent.

	:param list_dgms: List of persistent diagrmas
	:type list_dgms: list

	:param d: number of a bins in each axis.
	:type d: int

	:param padding: this increases the size of the box aournd the collection of persistent diagrmas
	:type padding: float

	:return: numpy array, float -- d x 2 array of centers, size of the tent domain
	'''

	minimums, maximums = limits_box(list_dgms)
	birth_min = minimums[0] - padding
	lifespan_min = minimums[1] - padding
	birth_max = maximums[0] + padding
	lifespan_max = maximums[1] + padding

	birth_step = (birth_max - birth_min)/(d+1)
	lifespan_step = (lifespan_max - lifespan_min)/(d+1)

	birth_coord = []
	lifespan_coord = []
	for i in range(1,d):
		birth_coord.append(birth_min + birth_step*i)
		lifespan_coord.append(lifespan_min + lifespan_step*i)

	x, y = np.meshgrid(birth_coord, lifespan_coord)

	x = x.flatten() # center of the box, birth coordinate
	y = y.flatten() # center of the box, lifespan coordiante

	return np.column_stack((x,y)), max(birth_step, lifespan_step)

def f_box(x, center, delta):
	'''
	Computes the function

	.. math::
		g_{(a,b), \delta}(x) = \max \\left\{ 0,1 - \\frac{1}{\delta} \max\\left\{ | x-a | , | y-b | \\right\} \\right\}

	:param x: point to evaluate the function :math:`g_{(a,b), \delta}`.
	:type x: numpy array

	:param center: Center of the tenf function.
	:type center: numpy array

	:param delta: size fo the tent function domain.
	:type delta: float

	:return: float -- tent function :math:`g_{(a,b), \delta}(x)` evaluated at `x`.

	'''
	x = deepcopy(x)
	x[1] = x[1] - x[0] # Change death to lifespan.
	return max(0, 1 - (1/delta)*max(abs(x[0] - center[0]), abs(x[1] - center[1])))

def f_ellipse (x, center=np.array([0,0]), axis=np.array([1,1]), rotation=np.array([[1,0],[0,1]])):
	'''
	Computes a bump function centered with an ellipsoidal domain centered ac `c`, rotaded by 'rotation' and with axis given by 'axis'. The bump function is computed using the gollowing formula 

	.. math::
		f_{A,c} (x) = \max \\left\{ 0, 1 - (x - c)^T A (x - c)\\right\}

	:param x: point to avelatuate the function :math:`f_{A,c}`
	:type z: Numpy array

	:param center: center of the ellipse
	:type center: Numpy array

	:param axis: Size f themjor an minor axis of the ellipse
	:type axis: Numpy array

	:param rotation: Rotation matrix for the ellipse
	:type rotation: Numpy array

	:return: float -- value of :math:`f_{A,c} (x)`.
	'''
	sigma = np.diag(np.power(axis, -2))
	x_centered = np.subtract(x, center)
	temp = x_centered@rotation@sigma@np.transpose(rotation)@np.transpose(x_centered)
	temp = np.diag(temp)

	return np.maximum(0, 1-temp)

def f_gaussian(x, mean, std):
	'''
	Computes

	.. math::
		g(x) = \\frac{1}{\sigma \sqrt{2\pi}} e^{- \\left( \\frac{x-\mu}{\sqrt{2}\sigma} \\right)^2}
	
	:param x: point to evaluate :math:`g` at
	:type x: numpy array

	:param mean: center of the gaussian
	:type mean: float

	:param std: standard deviation
	:type std: float

	:return: numpy array -- :math:`g(x)`.
	'''

	return np.exp((-0.5)*np.power((x-mean)/std, 2)) / (std*np.sqrt(2*np.pi))

#@jit(parallel=True)
def f_dgm(dgm, function, **keyargs):
	'''
	Given a persistend diagram :math:`D = (S,\mu)` and a compactly supported function in :math:`\mathbb{R}^2`, this function computes

	.. math::
		\\nu_{D}(f) = \sum_{x\in S} f(x)\mu(x)

	:param dgm: persistent diagram, array of points in :math:`\mathbb{R}^2`.
	:type dgm: Numpy array

	:param function: Compactly supported function in :math:`\mathbb{R}^2`.
	:type function: function

	:param keyargs: Additional arguments required by `funciton`
	:type keyargs: Dicctionary

	:return: float -- value of :math:`\\nu_{D}(f)`.
	'''

	temp = function(dgm, **keyargs)

	return sum(temp)

#@jit(parallel=True)
def feature(list_dgms, function, **keyargs):
	'''
	Given a collection of persistent diagrams and a compactly supported function in :math:`\mathbb{R}^2`, computes :math:`\\nu_{D}(f)` for each diagram :math:`D` in the collection.

	:param list_dgms: list of persistent diagrams
	:type list_dgms: list

	:param function: Compactly supported function in :math:`\mathbb{R}^2`.
	:type function: function

	:param keyargs: Additional arguments required by `funciton`
	:type keyargs: Dicctionary

	:return: Numpy array -- Array of values :math:`\\nu_{D}(f)` for each diagram :math:`D` in the collection `list_dgms`.
	'''
	num_diagrams = len(list_dgms)

	feat = np.zeros(num_diagrams)
	for i in range(num_diagrams):
		feat[i] = f_dgm(list_dgms[i], function, **keyargs)

	return feat