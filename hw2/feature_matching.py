from scipy.spatial import KDTree
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
  features_list = [[indices[i],coordinates[i],orientations[i],descriptions[i]] for i in range(len(orientations))]
  tree = KDTree(descriptions)
  pair_list = list(tree.query_pairs(r = 3))
  feature_pair_list = [(features_list[i1],features_list[i2]) for (i1,i2) in pair_list]
  
  return feature_pair_list 
features = [
  [[[1,2],[3,5],[2,3,4]],[[1,5],[3,1],[3,3,2]]],
  [[[2,1],[3,4],[10,3,4]],[[1,5],[3,2],[5,3,2]]],
  [[[2,-1],[3,5],[10,3,1]],[[3,5],[3,0],[7,3,5]]],
]
for pair in (feature_matching(features)):
    print(pair)
