import numpy as np
import cv2
import random
import pickle
import os
import sys
from matplotlib import pyplot as plt


### I/O ###
def readImages(dir_name, txt_name):
    print("Reading images...")
    with open(dir_name + txt_name, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            num_img = int(line)
            break

        contents = []
        for line in f:
            if line[0] == '#':
                continue
            contents.append(line.split())

        filenames = [content[0] for content in contents]
        log_t = list(map(float, [content[1] for content in contents]))
        log_t = -np.log2(log_t)
        [print("-", f) for f in filenames]

    # cv2.imread flag: grayscale -> =0, BGR -> >0
    imgs            = [cv2.imread(dir_name+f.replace("ppm", "png"), 1) for f in filenames]
    resized_imgs    = [cv2.resize(img, (0, 0), fx=1/round(max(img.shape)/1000), fy=1/round(max(img.shape)/1000), interpolation=cv2.INTER_AREA) for img in imgs]

    print("done.\n")
    return resized_imgs, log_t

### convert into grey scale ###
def toGrayScale(img):
#    return (img[:,:,0] * 19 + 183 * img[:,:,1] + 54 * img[:,:,2]) / 256
    return 0.27 * img[:,:,2] + 0.67 * img[:,:,1] + 0.06 * img[:,:,0]

def toThresholdImage(img):
    m = np.median(img)
    return img>m

def toMaskImage(img):
    m = np.median(img)
    return np.any([img>m*1.05, img<m*0.95],0)

def saveBoolImage(img, name):
    cv2.imwrite(name, (img*255).astype(int))
### tone mapping ###
def alignImage(imgs, num_shift, dirs, names):
    print("aligning...")
    grey_imgs = [toGrayScale(img) for img in imgs]
    for i in range(len(names)):
        saveBoolImage(grey_imgs[i],dirs+'grey_'+names[i])
    thres_imgs = [toThresholdImage(img) for img in grey_imgs]
    mask_imgs = [toMaskImage(img) for img in grey_imgs]
    for i in range(len(names)):
        saveBoolImage(mask_imgs[i],dirs+'mask_'+names[i])
    n = num_shift
    img0 = thres_imgs[len(imgs)//2][2**(n+1):-(2**(n+1)),2**(n+1):-(2**(n+1))]
    print(img0)
    mask0 = mask_imgs[len(imgs)//2][2**(n+1):-(2**(n+1)),2**(n+1):-(2**(n+1))]
    print('mask:',np.sum(mask0))
    for i in range(0, len(imgs)):
        shift_x = 0
        shift_y = 0
        num_shift = n
        print("aligning image ", i)
        while(num_shift >= 0):
#            print("layer ",num_shift)
            min_x = 0
            min_y = 0
            min_dif = 1e9
            for x in range(-1,2) :
                for y in range(-1,2) :
#                    print("testing ",(x,y))
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
        print("offset: ",(shift_x, shift_y))

if __name__ == '__main__':
    dir_name = sys.argv[1]
    txt_name = '.hdr_image_list.txt'
    print("Reading images...")
    with open(dir_name + txt_name, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            num_img = int(line)
            break

        contents = []
        for line in f:
            if line[0] == '#':
                continue
            contents.append(line.split())

        filenames = [content[0] for content in contents]
        log_t = list(map(float, [content[1] for content in contents]))
        log_t = -np.log2(log_t)
        [print("-", f) for f in filenames]

    # cv2.imread flag: grayscale -> =0, BGR -> >0
    imgs            = [cv2.imread(dir_name+f.replace("ppm", "png"), 1) for f in filenames]
    resized_imgs    = [cv2.resize(img, (0, 0), fx=1/round(max(img.shape)/1000), fy=1/round(max(img.shape)/1000), interpolation=cv2.INTER_AREA) for img in imgs]
    print("done.\n")
    alignImage(resized_imgs, 4, dir_name, filenames)
