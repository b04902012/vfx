#!/usr/bin/python3

import numpy as np
from numpy import linalg as LA
import cv2
from scipy.ndimage import gaussian_filter

import getopt
from tqdm import tqdm

import sys
import os

import feature_describing
import feature_matching
import image_matching
import cylinder_reconstructing
feature_describing=feature_describing.feature_describing
feature_matching=feature_matching.feature_matching
image_matching=image_matching.image_matching
cylinder_reconstructing = cylinder_reconstructing.cylinder_reconstructing

def parseArgs():
    args, ignore = getopt.getopt(sys.argv[1:], "f:w:t:l", ["file=", "window=", "local", "threshold"])
    args = dict(args)

    dir_name = args.get('-f') or args.get('--file')
    threshold = args.get('-t') or args.get('--threshold')
    local = ('-l' in args or '--local' in args)

    if not dir_name:
        sys.exit('Please provide directory name with -f or --file.')

    if not threshold:
        sys.exit('Please provide threshold value with -t or --threshold.')

    threshold = int(threshold)

    return dir_name, threshold, local


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

def featureDetection(color_imgs, imgs, window_size=5, k=0.05, threshold=None, local=False):
    """
    Detect features and return (x, y) coordinates of keypoints.
    Saving new images with red dots highlighting the keypoints.

    Args:
        color_imgs
        imgs: gray images for detecting features
        window_size (int): window size of the gaussian filter
        k (float): recommended value 0.04 ~ 0.06
        threshold: set threshold for keypoints, if None then output first 5000 keypoints
        local: output local keypoints instead of global keypoints if it is set to True
    
    Returns:
        list of list of tuples (coordinates of the detected keypoints in each image)
    """
    offset = (window_size-1)//2
    sigma = (window_size+1)/3

    cornerlist = [[] for img in imgs]
    descriptionlist = [[] for img in imgs]

    x, y = np.mgrid[-offset:offset+1, -offset:offset+1]
    gaussian = np.exp(-(x**2+y**2)/2/sigma**2)

    g = lambda window: (window*gaussian).sum()

    with open("log", "w") as f:
        for i, (color_img, img) in enumerate(tqdm(zip(color_imgs, imgs))):

            h, w = img.shape

            #Ix, Iy = np.gradient(img)      # buggy gradient
            print(img)
            print(i)
            Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0)
            Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1)

            Ixx = Ix**2
            Iyy = Iy**2
            Ixy = Ix*Iy

            R = np.zeros(img.shape)

            for x in range(offset, h-offset):
                for y in range(offset, w-offset):
                    M = np.array(   ((g(Ixx[x-offset:x+offset+1, y-offset:y+offset+1]),
                                     g(Ixy[x-offset:x+offset+1, y-offset:y+offset+1])),
                                    (g(Ixy[x-offset:x+offset+1, y-offset:y+offset+1]),
                                     g(Iyy[x-offset:x+offset+1, y-offset:y+offset+1]))))
                    
                    eigs = LA.eigvals(M)

                    R[x][y] = eigs[0]*eigs[1] - k*((eigs[0]+eigs[1])**2)

            cornerlist[i] = [(R[x, y], (x, y)) for x in range(offset, h-offset) for y in range(offset, w-offset)]
            
            if local:
                cornerlist[i] = [(r, (x, y)) for (r, (x, y)) in cornerlist[i] if r == np.amax(R[x-offset:x+offset, y-offset:y+offset]) and r - np.amin(R[x-offset:x+offset, y-offset:y+offset]) >= threshold]

            
            cornerlist[i] = [(x, y) for r, (x, y) in cornerlist[i] if r >= threshold]
            
            descriptionlist[i] = [feature_describing(img, Ix, Iy, (x, y)) for (x, y) in cornerlist[i]]

            for x, y in cornerlist[i]:
                color_img.itemset((x, y, 0), 0)
                color_img.itemset((x, y, 1), 0)
                color_img.itemset((x, y, 2), 255)

            print(len(cornerlist[i]))
            
            cv2.imwrite(os.path.join(dir_name, f"feature{i}.png"), color_img)

    return cornerlist, descriptionlist



if __name__ == "__main__":
    dir_name, threshold, local = parseArgs()

    color_imgs = readImages(dir_name)
    color_imgs = color_imgs[:5]
    color_imgs = color_imgs[::-1]
#    for i in range(len(color_imgs)):
#      color_imgs[i] = cylinder_reconstructing(color_imgs[i], 705)
    gray_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in color_imgs]

    cornerlist, descriptionlist = featureDetection(color_imgs, gray_imgs, threshold=threshold, local=local)
    print("matching......")
    cur_transform = np.identity(3)
    for i in range(0,len(gray_imgs)-1):
      index_pairs = feature_matching(descriptionlist[i],descriptionlist[i+1])
      index_set1 = [pair[0] for pair in index_pairs]
      index_set2 = [pair[1] for pair in index_pairs]
      transform_matrix = image_matching([cornerlist[i][index] for index in index_set1],[cornerlist[i+1][index] for index in index_set2])
      cur_transform = np.matmul(cur_transform, transform_matrix)
      print(gray_imgs[i].shape)
      img1 = np.transpose(cv2.warpPerspective(src = np.transpose(gray_imgs[i+1]), M = cur_transform, dsize = (gray_imgs[i].shape[0],5*gray_imgs[i].shape[1])))
      cv2.imwrite(f"parrington/test{i+1}.png", img1)
