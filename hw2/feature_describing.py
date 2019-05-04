import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
def rotate_image(img, pivot, angle):
  padX = [img.shape[1] - pivot[0]+2000, pivot[0]+2000]
  padY = [img.shape[0] - pivot[1]+2000, pivot[1]+2000]
  imgP = np.pad(img, [padY, padX], 'constant')
  imgR = ndimage.rotate(imgP, angle, reshape=False)
  imgC = imgR[padY[0] : -padY[1], padX[0] : -padX[1]]
  return imgC

def feature_describing(image, Ix, Iy, coordinate, window_size=8):
  offset = (window_size-1)//2
  x = coordinate[0]
  y = coordinate[1]
  orientation=np.zeros(8)

  for i in range(x-offset,x+offset):
    for j in range(y-offset,y+offset):
      g=np.array([Ix[i][j],Iy[i][j]])
      a=np.arctan2(g[1],g[0])*180//np.pi
      idx=int((a+180)//45)%8
      orientation[idx]+=np.linalg.norm(g)
  main_orientation = np.argmax(orientation)
  '''rotation_matrix=cv2.getRotationMatrix2D(center = (y,x), angle = -main_orientation*45, scale = 1)
  rIx=cv2.warpAffine(Ix,rotation_matrix,Ix.shape)
  rIy=cv2.warpAffine(Iy,rotation_matrix,Iy.shape)'''
  rIx=Ix
  rIy=Iy
  
  cv2.imwrite('rIx.png',rIx)
  sub_orientation=np.zeros(32)
  for sub_x in range(0,2):
    for sub_y in range(0,2):
      for i in range(0,offset):
        for j in range(0,offset):
          sub_idx = sub_x*2 + sub_y
          rx = x + (sub_x-1)*offset + i
          ry = y + (sub_y-1)*offset + j
          g=np.array([rIx[rx][ry],rIy[rx][ry]])
          a=np.arctan2(g[1],g[0])*180/np.pi
          idx = int((a+180)//45)
          idx = (idx - main_orientation + 8)%8
          sub_orientation[sub_idx * 8 + idx] += np.linalg.norm(g)
  if(np.linalg.norm(sub_orientation) > 0.0):
    sub_orientation /= np.linalg.norm(sub_orientation)
  return [coordinate,main_orientation,sub_orientation]
