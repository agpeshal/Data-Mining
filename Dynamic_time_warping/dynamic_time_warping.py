from __future__ import division, print_function
import numpy as np
from math import inf

def dist(v1, v2):
	return np.sum(abs(v1-v2))


def constrained_dtw(t1, t2, w):
	m = len(t1)
	n = len(t2)
	C = np.zeros((m+1, n+1))
	C[0, 0] = 0
	for i in range(1,m):
		C[i, 0] = inf
	for j in range(1,n):
		C[0, j] = inf

	for i in range(m):
		for j in range(n):
			if abs(i - j) <= w:
				C[i+1,j+1] = dist(t1[i], t2[j]) + min( C[i, j], C[i, j+1], C[i+1, j] )
			else:
				C[i+1,j+1] = inf

	return(C[-1, -1])