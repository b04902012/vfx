import random
import numpy as np
import cv2
def transform(transform_matrix, coordinate):
    new_coordinate = np.matmul(transform_matrix,np.transpose(np.array([coordinate[0],coordinate[1],1])))
#    return np.array([new_coordinate[0]/new_coordinate[2],new_coordinate[1]/new_coordinate[2]])
    return np.array([new_coordinate[0],new_coordinate[1]])
def image_matching(coordinate_set1,coordinate_set2):
  min_outlier = len(coordinate_set1)
  print(len(coordinate_set1))
  min_transform_matrix = np.identity(3)
  transform_matrix = np.identity(3)
  for t in range(3000):
    indices = random.sample(list(range(len(coordinate_set1))), 3)
    c1 = np.float32(np.array(coordinate_set1))[indices]
    c2 = np.float32(np.array(coordinate_set2))[indices]
    transform_matrix = cv2.getAffineTransform(c1, c2)
    transform_matrix = np.append(transform_matrix, np.float32(np.array([[0,0,1]])),0)
    outlier = 0
    for i in range(len(coordinate_set1)):
      original_coordinate = np.array(coordinate_set1[i])
      transformed_coordinate = transform(transform_matrix, coordinate_set2[i])
      if(np.linalg.norm(transformed_coordinate-original_coordinate)>2):
        outlier += 1
    if(outlier<min_outlier):
      min_outlier = outlier
      min_transform_matrix = transform_matrix
  print(min_transform_matrix)
  print(min_outlier)
  return min_transform_matrix
