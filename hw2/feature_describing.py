import cv2
import numpy as np
def feature_describing(image, Ix, Iy, coordinate, window_size=8):
  offset = (window_size-1)//2
  x = coordinate[0]
  y = coordinate[1]
  orientation=np.zeros(8)

  for i in range(x-offset,x-offset):
    for j in range(y-offset,y+offset):
      g=np.array([Ix[i][j],Iy[i][j]])
      a=np.arctan2(g[1],g[0])*180/np.pi
      idx=int((a+180)//45)
      orientation[idx]+=np.linalg.norm(g)
  main_orientation = np.argmax(orientation)
  rotation_matrix=cv2.getRotationMatrix2D((x,y),-main_orientation*45,1)
  rIx=cv2.warpAffine(Ix,rotation_matrix,Ix.shape)
  rIy=cv2.warpAffine(Iy,rotation_matrix,Iy.shape)
  print(Ix[x][y])
  print(rIx[x][y])
  input()
  sub_orientation=np.zeros(32)
  for sub_x in range(0,1):
    for sub_y in range(0,1):
      for i in range(0,offset):
        for j in range(0,offset):
          sub_idx = sub_x*2 + sub_y
          rx = x + (sub_x-1)*offset + i
          ry = y + (sub_y-1)*offset + j
          print(Ix)
          print(rIx)
          g=np.array([rIx[rx][ry],rIy[rx][ry]])
          print(g)
          a=np.arctan2(g[1],g[0])*180/np.pi
          idx = int((a+180)//45)
          idx = (idx - main_orientation + 8)%8
          sub_orientation[sub_idx * 8 + idx] += np.linalg.norm(g)
  sub_orientation /= np.linalg.norm(sub_orientation)
  return [main_orientation,sub_orientation]
