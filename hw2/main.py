#!/usr/bin/python3

import numpy as np
from numpy import linalg as LA
import cv2

import getopt
from tqdm import tqdm
import sys
import os
import feature_describing
feature_describing=feature_describing.feature_describing

def readImages(dir_name):
    """
    Read <dir_name>/pano.txt to find all the images in <dir_name>, return the images
    
    Args:
        dir_name (str)
    
    Returns:
        list of cv2 color images
    """
    print('* Reading images...')
    
    imgs = []
    with open(os.path.join(dir_name, "pano.txt")) as f:
        for image_name in f.readlines()[::13]:
            full_name = os.path.join(dir_name, image_name.strip())
            print('  -', full_name)
            imgs.append(cv2.imread(full_name))

    return imgs

def featureDetection(color_imgs, imgs, window_size=3, k=0.05, threshold=None):
    """
    Detect features and return (x, y) coordinates of keypoints.
    Saving new images with red dots highlighting the keypoints.

    Args:
        color_imgs
        imgs: gray images for detecting features
        window_size (int): window size of the gaussian filter
        k (float): recommended value 0.04 ~ 0.06
        threshold: set threshold for keypoints, if None then output first 3000 keypoints
    
    Returns:
        list of list of tuples (coordinates of the detected keypoints in each image)
    """
    offset = (window_size-1)//2
    sigma = (window_size+1)/3

    cornerlist = [[] for img in imgs]

    x, y = np.mgrid[-offset:offset+1, -offset:offset+1]
    gaussian = np.exp(-(x**2+y**2)/2/sigma**2)

    g = lambda window: (window*gaussian).sum()

    with open("log", "w") as f:
        for i, (color_img, img) in enumerate(tqdm(zip(color_imgs, imgs))):

            h, w = img.shape

            #Ix, Iy = np.gradient(img)      # buggy gradient
            Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0)
            Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1)

            Ixx = Ix**2
            Iyy = Iy**2
            Ixy = Ix*Iy

            for x in range(2*offset, h-2*offset):
                for y in range(2*offset, w-2*offset):
                    M = np.array(   ((g(Ixx[x-offset:x+offset+1, y-offset:y+offset+1]),
                                     g(Ixy[x-offset:x+offset+1, y-offset:y+offset+1])),
                                    (g(Ixy[x-offset:x+offset+1, y-offset:y+offset+1]),
                                     g(Iyy[x-offset:x+offset+1, y-offset:y+offset+1]))))
                    
                    eigs = LA.eigvals(M)

                    R = eigs[0]*eigs[1] - k*((eigs[0]+eigs[1])**2)

                    #f.write(str(R)+'\n')
                    
                    if threshold == None:
                        print(Ix[x][y])
                        print(Iy[x][y])
                        print(feature_describing(img,Ix,Iy,[x,y]))
                        cornerlist[i].append((R, (x, y)))

                    elif R > threshold:
                        cornerlist[i].append((x, y))
                        
            if threshold == None:
                cornerlist[i].sort()
                #for j in range(3000):
                #   print(cornerlist[-i-1][0])
                cornerlist[i] = [(x, y) for r, (x, y) in cornerlist[i][-3000:]]


            for x, y in cornerlist[i]:
                color_img.itemset((x, y, 0), 0)
                color_img.itemset((x, y, 1), 0)
                color_img.itemset((x, y, 2), 255)
            
            cv2.imwrite(os.path.join(dir_name, f"feature{i}.png"), color_img)

    return cornerlist



if __name__ == "__main__":
    args, ignore = getopt.getopt(sys.argv[1:], "f:w:", ["file=", "window="])
    args = dict(args)

    dir_name = args.get('-f') or args.get('--file')
    if not dir_name:
        sys.exit('Please provide directory name with -f or --file.')

    color_imgs = readImages(dir_name)
    gray_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in color_imgs]

    keypoints = featureDetection(color_imgs, gray_imgs)
    #featureMatching(keypoints)


