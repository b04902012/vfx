import random
import numpy as np
import cv2
def transform(transform_matrix, coordinate):
    new_coordinate = np.matmul(transform_matrix,np.transpose(np.array([coordinate[0],coordinate[1],1])))
    return np.array([new_coordinate[0]/new_coordinate[2],new_coordinate[1]/new_coordinate[2]])
def image_matching(coordinate_set1,coordinate_set2):
  min_outlier = len(coordinate_set1)
  print(len(coordinate_set1))
  min_transform_matrix = np.identity(3)
  for t in range(6000):
    transform_matrix = np.identity(3)
    index = random.sample(range(0,len(coordinate_set1)),1)[0]
    transform_matrix[0][2] = coordinate_set1[index][0]-coordinate_set2[index][0]
    transform_matrix[1][2] = coordinate_set1[index][1]-coordinate_set2[index][1]
    outlier = 0
    for i in range(len(coordinate_set1)):
      original_coordinate = np.array(coordinate_set1[i])
      transformed_coordinate = transform(transform_matrix, coordinate_set2[i])
      if(np.linalg.norm(transformed_coordinate-original_coordinate)>1):
        outlier += 1
    if(outlier<min_outlier):
      min_outlier = outlier
      min_transform_matrix = transform_matrix
  print(min_outlier)
  return min_transform_matrix
