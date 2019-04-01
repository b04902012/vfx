import numpy as np
import cv2
import random
import pickle
import os
import sys
from matplotlib import pyplot as plt
### convert into grey scale ###
def toGrayScale(img):
    return 0.27 * img[:,:,2] + 0.67 * img[:,:,1] + 0.06 * img[:,:,0]

### calculate the high-50% bool matrix ###
def toThresholdImage(img):
    m = np.median(img)
    return img>m

### discard the cell close to median ###
def toMaskImage(img):
    m = np.median(img)
    return np.any([img>m*1.05, img<m*0.95],0)

### save bool matrix as black-white png
def saveBoolImage(img, name):
    cv2.imwrite(name, (img*255).astype(int))

### image aligning ###
def alignImage(imgs, num_shift):
    print("aligning...")
    grey_imgs = [toGrayScale(img) for img in imgs]
    thres_imgs = [toThresholdImage(img) for img in grey_imgs]
    mask_imgs = [toMaskImage(img) for img in grey_imgs]
    n = num_shift
    img0 = thres_imgs[len(imgs)//2][2**(n+1):-(2**(n+1)),2**(n+1):-(2**(n+1))]
    print(img0)
    mask0 = mask_imgs[len(imgs)//2][2**(n+1):-(2**(n+1)),2**(n+1):-(2**(n+1))]
    aligned_images=[]
    for i in range(0, len(imgs)):
        shift_x = 0
        shift_y = 0
        num_shift = n
        print("aligning image ", i)
        while(num_shift >= 0):
            min_x = 0
            min_y = 0
            min_dif = 1e9
            for x in range(-1,2) :
                for y in range(-1,2) :
                    del_x = shift_x + x*(2**num_shift)
                    del_y = shift_y + y*(2**num_shift)
                    img1 = np.roll(thres_imgs[i],(del_x, del_y), (0,1))[2**(n+1):-(2**(n+1)),2**(n+1):-(2**(n+1))]
                    mask1 = np.roll(mask_imgs[i], (del_x, del_y), (0,1))[2**(n+1):-(2**(n+1)),2**(n+1):-(2**(n+1))]
                    mask = np.logical_and(mask0,mask1)
                    masked_dif = np.logical_and(mask,np.logical_xor(img0,img1))
                    dif = np.sum(masked_dif)
                    if(dif < min_dif):
                        min_dif = dif
                        min_x = x
                        min_y = y
            shift_x = shift_x + min_x*(2**num_shift)
            shift_y = shift_y + min_y*(2**num_shift)
            num_shift=num_shift-1
        print("Shift: ",(shift_x, shift_y))
        aligned_images.append(np.roll(imgs[i], (shift_x,shift_y), (0,1))[2**(n+1):-(2**(n+1)),2**(n+1):-(2**(n+1))])
    return aligned_images
