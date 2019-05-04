from scipy.spatial import KDTree
import sys
sys.setrecursionlimit(10000)
def feature_matching(features):
  indices = []
  for i in range(0,len(features)):
    indices += [i] * len(features[i])
  coordinates = []
  for i in range(0,len(features)):
    coordinates += [feature[0] for feature in features[i]]
  orientations = []
  for i in range(0,len(features)):
    orientations+= [feature[1] for feature in features[i]]
  descriptions = []
  for i in range(0,len(features)):
    descriptions+= [feature[2] for feature in features[i]]
  print(len(descriptions))
  features_list = [[indices[i],coordinates[i],orientations[i],descriptions[i]] for i in range(len(orientations))]
  tree = KDTree(descriptions)
  pair_list = list(tree.query_pairs(r = 3))
  feature_pair_list = [(features_list[i1],features_list[i2]) for (i1,i2) in pair_list]
  return feature_pair_list 
