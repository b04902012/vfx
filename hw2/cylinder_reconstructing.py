import numpy as np
def cylinder_inverse_transforming(c,s):
  x,y = c[0],c[1]
  x1 = s * np.tan(x/s)
  y1 = ((x**2 + s**2)**0.5) * y / s
  return (x1, y1)
def cylinder_reconstructing(img, s):
  new_img = np.zeros(img.shape,dtype = np.uint8)
  for x in range(img.shape[1]):
    for y in range(img.shape[0]):
      (x1, y1) = cylinder_inverse_transforming((x-img.shape[1]/2,y-img.shape[0]/2), s)
      x1 += img.shape[1]/2
      y1 += img.shape[0]/2
      x1 = int(x1)
      y1 = int(y1)
      if (0<=x1) and (x1<img.shape[1]) and (0<=y1) and (y1<img.shape[0]):
        new_img[y][x] = img[y1][x1]
  return new_img
