import numpy as np
def cylinder_transforming(c,s):
  x,y = c[0],c[1]
  x1 = s * np.arctan(x/s)
  y1 = s * y / ((x**2 + s**2)**0.5) 
  return (x1, y1)
def cylinder_inverse_transforming(c,s):
  x,y = c[0],c[1]
  x1 = s * np.tan(x/s)
  y1 = ((x**2 + s**2)**0.5) * y / s
  return (x1, y1)
def cylinder_reconstructing(img, s):
  (new_img_w,new_img_h) = cylinder_transforming((img.shape[1]/2,img.shape[0]/2),s)
  new_img_w *= 2
  new_img_h *= 2
  new_img_w = int(new_img_w)
  new_img_h = int(new_img_h)
  new_img = np.zeros((new_img_h,new_img_w,3),dtype = np.uint8)
  for x in range(new_img_w):
    for y in range(new_img_h):
      (x1, y1) = cylinder_inverse_transforming((x-new_img_w/2,y-new_img_h/2), s)
      x1 += img.shape[1]/2
      y1 += img.shape[0]/2
      x1 = int(x1)
      y1 = int(y1)
      if (0<=x1) and (x1<img.shape[1]) and (0<=y1) and (y1<img.shape[0]):
        new_img[y][x] = img[y1][x1]
  return new_img
