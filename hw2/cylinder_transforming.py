import numpy as np
def cylinder_inverse_transforming(c,s):
  x,y = c[0],c[1]
  x1 = np.arctan(x/s) * s
  y1 = s * y / sqrt(x**2 + s**2)
  return (x1, y1)
def cylinder_reconstructing(img, s):
  new_img = np.zeros(img.shape)
  for x in range(img.shape[1]):
    for y in range(img.shape[0]):
      x1, y1 = cylinder_inverse_transforming((x,y), s)
      x1 = round(x1)
      y1 = round(y1)
      if (0<=x) and (x<img.shape[1]) and (0<=y) and (y<img.shape[0]):
        new_img[y][x] = img[y1][x1]
