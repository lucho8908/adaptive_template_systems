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

# ------------------------------ Teaspoon data generation  --------------------
#-------------Circles and Annuli---------------------------------------#
def Circle(N = 100, r=1, gamma=None, seed = None):
	"""
	Generate N points in R^2 from the circle centered
	at the origin with radius r.

	If `gamma` is not `None`, then we add noise
	using a normal distribution.  Note that this means the resulting
	distribution is not bounded, so your favorite stability theorem doesn't
	immediately apply.

	Parameters
	----------
	N -
		Number of points to generate
	r -
		Radius of the circle
	gamma -
		Standard deviation of the normally distributed noise. 
	seed -
		Fixes the seed.  Good if we want to replicate results.


	Returns
	-------
	P -
		A Nx2 numpy array with the points drawn as the rows.

	"""
	np.random.seed(seed)
	theta = np.random.rand(N,1)
	theta = theta.reshape((N,))
	P = np.zeros([N,2])

	P[:,0] = r*np.cos(2*np.pi*theta)
	P[:,1] = r*np.sin(2*np.pi*theta)

	if gamma is not None:
		# better be a number of some type!
		noise = np.random.normal(0, gamma, size=(N,2))
		P += noise

	return P


def Sphere(N = 100, r = 1, noise = 0, seed = None):
	"""
	Generate N points in R^3 from the sphere centered
	at the origin with radius r.
	If noise is set to a positive number, the points
	can be at distance r +/- noise from the origin.

	Parameters
	----------
	N -
		Number of points to generate
	r -
		Radius of the sphere
	seed -
		Fixes the seed.  Good if we want to replicate results.


	Returns
	-------
	P -
		A Nx3 numpy array with the points drawn as the rows.

	"""
	np.random.seed(seed)

	Rvect = 2*noise*np.random.random(N) + r
	thetaVect =   np.pi * np.random.random(N)
	phiVect = 2 * np.pi * np.random.random(N)

	P = np.zeros((N,3))
	P[:,0] = Rvect * np.sin(thetaVect) * np.cos(phiVect)
	P[:,1] = Rvect * np.sin(thetaVect) * np.sin(phiVect)
	P[:,2] = Rvect * np.cos(thetaVect)

	return P


def Annulus(N=200,r=1,R=2, seed = None):
	'''
	Returns point cloud sampled from uniform distribution on
	annulus in R^2 of inner radius r and outer radius R

	Parameters
	----------
	N -
		Number of points to generate
	r -
		Inner radius of the annulus
	R -
		Outer radius of the annulus
	seed -
		Fixes the seed.  Good if we want to replicate results.


	Returns
	-------
	P -
		A Nx2 numpy array with the points drawn as the rows.

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


#-------------Torus a la Diaconis paper--------------------------------#

def Torus(N = 100, r = 1,R = 2,  seed = None):
	'''
	Sampling method taken from Sampling from a Manifold by Diaconis,
	Holmes and Shahshahani, arXiv:1206.6913

	Generates torus with points
	x = ( R + r*cos(theta) ) * cos(psi),
	y = ( R + r*cos(theta) ) * sin(psi),
	z = r * sin(theta)

	Need to draw theta with distribution

	g(theta) = (1+ r*cos(theta)/R ) / (2pi) on 0 <= theta < 2pi

	and psi with uniform density on [0,2pi).

	For theta, draw theta uniformly from [0,2pi) and
	eta from [1-r/R,1+r/R].  If eta< 1 + (r/R) cos(theta), return theta.

	Parameters
	----------
	N -
		Number of points to generate
	r -
		Inner radius of the torus
	R -
		Outer radius of the torus
	seed -
		Fixes the seed.  Good if we want to replicate results.


	Returns
	-------
	P -
		A Nx3 numpy array with the points drawn as the rows.

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




#----------------------------------------------------------------------#

def Cube(N = 100, diam = 1, dim = 2, seed = None):
	"""
	Generate N points in R^dim from the box
	[0,diam]x[0,diam]x...x[0,diam]

	Parameters
	----------
	N -
		Number of points to generate
	diam -
		Points are pulled from the box
		[0,diam]x[0,diam]x...x[0,diam]
	dim -
		Points are embedded in R^dim

	Returns
	-------
	P -
		A Nxdim numpy array with the points drawn as the rows.

	"""
	np.random.seed(seed)

	P = diam*np.random.random((N,dim))

	return P



#----------------------------------------------------------------------#

def Clusters(N = 100,
			centers = np.array(((0,0),(3,3))),
			sd = 1,
			seed = None):
	"""
	Generate k clusters of points, N points in total (evenly divided?)
	centers is a k x d numpy array, where centers[i,:] is the center of
	the ith cluster in R^d.
	Points are drawn from a normal distribution with std dev = sd

	Parameters
	----------
	N -
		Number of points to be generated
	centers -
		k x d numpy array, where centers[i,:] is the center of
		the ith cluster in R^d.

	sd -
		standard deviation of clusters.
		TODO: Make this enterable as a vector so each cluster can have
		a different sd?
	seed -
		Fixes the seed.  Good if we want to replicate results.

	Returns
	-------
	P -
		A Nxd numpy array with the points drawn as the rows.

	"""

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

# ---------------------------------------------------------------------------------

def testSetManifolds(numDgms = 50,
						numPts = 300,
						permute = True,
						seed = None
						):



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
		# Dgms.append([dgmOut[0],dgmOut[1]])
		DgmsDF.loc[counter] = [dgmOut[0],dgmOut[1], 'Cube']
		counter +=1

	#-
	print('Generating three cluster clouds...')
	# Centered at (0,0), (0,5), and (5,0) with sd =1
	# Then scaled by .3 to make birth/death times closer to the other examples
	centers = np.array( [ [0,0], [0,2], [2,0]  ])
	# centers = np.array( [ [0,0], [0,2], [2,0]  ])
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
		# Dgms.append([dgmOut[0],dgmOut[1]])
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
	list_dgms_temp = deepcopy(list_dgms)
	mins = np.inf*np.ones(2)
	maxs = -np.inf*np.ones(2)

	for dgm in list_dgms_temp:
		dgm[:,1] = dgm[:,1]-dgm[:,0] # Turns the birth-death into birth-lifespan
		mins = np.minimum(np.amin(dgm, axis=0), mins)
		maxs = np.maximum(np.amax(dgm, axis=0), maxs)

	return mins, maxs

def box_centers(list_dgms, d, padding):
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
		# birth_coord.append(birth_min + birth_step*i + birth_step/2)
		# lifespan_coord.append(lifespan_min + lifespan_step*i + lifespan_step/2)

		birth_coord.append(birth_min + birth_step*i)
		lifespan_coord.append(lifespan_min + lifespan_step*i)

	x, y = np.meshgrid(birth_coord, lifespan_coord)

	x = x.flatten() # center of the box, birth coordinate
	y = y.flatten() # center of the box, lifespan coordiante

	return np.column_stack((x,y)), max(birth_step, lifespan_step)

def f_box(x, center, delta):
	x = deepcopy(x)
	x[1] = x[1] - x[0] # Change death to lifespan.
	return max(0, 1 - (1/delta)*max(abs(x[0] - center[0]), abs(x[1] - center[1])))

def f_ellipse (x, center=np.array([0,0]), axis=np.array([1,1]), rotation=np.array([[1,0],[0,1]])):
	# point_centered = np.reshape(x - center, (-1,1))
	# point_rotated = np.transpose(rotation)@point_centered
	sigma = np.diag(np.power(axis, -2))
	# temp = point_rotated.transpose()@sigma@point_rotated

	x_centered = np.subtract(x, center)

	temp = x_centered@rotation@sigma@np.transpose(rotation)@np.transpose(x_centered)
	temp = np.diag(temp)

	return np.maximum(0, 1-temp)

def f_gaussian(x, mean, std):

	return np.exp((-0.5)*np.power((x-mean)/std, 2)) / (std*np.sqrt(2*np.pi))

#@jit(parallel=True)
def f_dgm(dgm, function, **keyargs):
	# total = 0
	# for i in range(len(dgm)):
	# 	total += function(dgm[i,:], **keyargs)

	temp = function(dgm, **keyargs)

	return sum(temp)

#@jit(parallel=True)
def feature(list_dgms, function, **keyargs):
	num_diagrams = len(list_dgms)

	feat = np.zeros(num_diagrams)
	for i in range(num_diagrams):
		feat[i] = f_dgm(list_dgms[i], function, **keyargs)

	return feat