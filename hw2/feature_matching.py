import numpy as np
import scipy as sp
import sys
def feature_matching(feature_set1, feature_set2):
  pairs = []
  for i in range(0,len(feature_set1)):
    delta = feature_set2 - feature_set1[i]
    dists = sp.linalg.norm(delta, axis = 1)
    j = np.argmin(a = dists)
    if(dists[j] < 0.4):
      pairs.append((i,j))
  return pairs
