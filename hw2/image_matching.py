import random
import numpy as np
import cv2
def transform(transform_matrix, coordinate):
    return np.matmul(transform_matrix,np.transpose(np.array([coordinate[0],coordinate[1],1])))
def image_matching(coordinate_set1,coordinate_set2):
  min_outlier = len(coordinate_set1)
  min_transform_matrix = np.zeros((2,3))
  for t in range(1000):
    indices = random.sample(range(0,len(coordinate_set1)),3)
    transform_matrix = cv2.getAffineTransform(np.float32([coordinate_set1[index] for index in indices]),np.float32([coordinate_set2[index] for index in indices]))
    outlier = 0
    for i in range(len(coordinate_set1)):
      original_coordinate = np.array(coordinate_set2[i])
      transformed_coordinate = transform(transform_matrix, coordinate_set1[i])
      if(np.linalg.norm(transformed_coordinate-original_coordinate)>5):
        outlier += 1
    if(outlier<min_outlier):
      min_outlier = outlier
      min_transform_matrix = transform_matrix
  print(min_outlier)
  return min_transform_matrix
