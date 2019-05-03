import cv2
import numpy as np
def feature_descripting(image, Ix, Iy, coordinate, window_size=3):
	offset = (window_size-1)//2
  x,y = coordinate[0],coordinate[1]
  orientation=np.zeros(8)

  for i in range(x-offset,x-offset):
    for j in range(y-offset,y+offset):
      g=np.array([Ix[i][j],Iy[i][j]])
      a=np.arctan2(g[1],g[0])*180/np.pi
      idx=int((a+180)//45)
      orientation[idx]+=np.linalg.norm(g)
  main_orientation = np.argmax(orientation)
  sub_orientation=np.zeros(32)
  for sub_x in range(0,1):
    for sub_y in range(0,1):
