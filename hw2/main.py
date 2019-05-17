#!/usr/bin/python3

import numpy as np
from numpy import linalg as LA
import cv2
from scipy.ndimage import gaussian_filter

import getopt
from tqdm import tqdm

import sys
import os
import pickle
from multiprocessing import Pool

from feature_describing import feature_describing
from feature_matching import feature_matching
from image_matching import image_matching
from cylinder_reconstructing import cylinder_reconstructing
from image_blending import image_blending

focal_mul = 5
resize_rate = 10
#resize_rate = 1
gaussain = None

def usage():
    print()
    print(f"Usage: python {sys.argv[0]} -f <directory_name> -t <threshold> [-l] [-w window_size] [--skip]")
    print()
    print("positional arguments:")
    print("    -f, --file       <directory_name>", "specifies the image folder you want to process")
    print("    -l, --threshold  <threshold>     ", "threshold on the level of feature points")
    print()
    print("optional arguments:")
    print("    -h, --help                       ", "show this message")
    print("    -w, --window     <window_size>   ", "assign window size for gaussian filter used in feature detection")
    print("    -l, --local                      ", "use local feature points instead of global feature points")
    print()

def g(window):
    return (window*gaussian).sum()

def parseArgs():
    try:
        args, ignore = getopt.getopt(sys.argv[1:], "f:w:t:lh", ["file=", "window=", "local", "threshold", "skip", "help"])
    except:
        usage()
        sys.exit()
    args = dict(args)

    dir_name = args.get('-f') or args.get('--file')
    threshold = args.get('-t') or args.get('--threshold')
    local = ('-l' in args or '--local' in args)
    skip = '--skip' in args

    if '-h' in args or '--help' in args or not dir_name or not threshold:
        usage()
        sys.exit()

    threshold = int(threshold)

    return dir_name, threshold, local, skip


def readImages(dir_name):
    """
    Read <dir_name>/pano.txt to find all the images in <dir_name>, return the images
    
    Args:
        dir_name (str)
    
    Returns:
        list of cv2 color images
    """
    
    imgs = []
    fls = []
    with open(os.path.join(dir_name, "pano.txt")) as f:
        for image_name in f.readlines()[::13]:
            image_name = image_name.split('\\')[-1]
            full_name = os.path.join(dir_name, image_name.strip())
            print('  -', full_name)
#            imgs.append(cv2.imread(full_name))
            img = cv2.imread(full_name)
            imgs.append(img)
        f.seek(0)
        for focal_length in f.readlines()[11::13]:
            fls.append(float(focal_length))

    return imgs, fls

def feature_detection_for_one(color_img, img, offset, k, threshold, local):
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

            R[x, y] = eigs[0]*eigs[1] - k*((eigs[0]+eigs[1])**2)

    corners = [(R[x, y], (x, y)) for x in range(offset, h-offset) for y in range(offset, w-offset)]
    
    if local:
        corners = [(r, (x, y)) for (r, (x, y)) in cornerlist[i] if r == np.amax(R[x-offset:x+offset, y-offset:y+offset]) and r - np.amin(R[x-offset:x+offset, y-offset:y+offset]) >= threshold]

    
    corners = [(x, y) for r, (x, y) in corners if r >= threshold]
    
    descriptions = [feature_describing(img, Ix, Iy, (x, y)) for (x, y) in corners]
    
    for x, y in corners:
        color_img.itemset((x, y, 0), 0)
        color_img.itemset((x, y, 1), 0)
        color_img.itemset((x, y, 2), 255)

    print(len(corners))

    return corners, descriptions

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
    n = len(color_imgs)

    #cornerlist = [[] for img in imgs]
    #descriptionlist = [[] for img in imgs]

    x, y = np.mgrid[-offset:offset+1, -offset:offset+1]
    
    global gaussian
    gaussian= np.exp(-(x**2+y**2)/2/sigma**2)


    with Pool(n) as pool:
        featurelist = pool.starmap(feature_detection_for_one, zip(color_imgs, imgs, [offset]*n, [k]*n, [threshold]*n, [local]*n))

    cornerlist = [feature[0] for feature in featurelist]
    descriptionlist = [feature[1] for feature in featurelist]

    for i, color_img in enumerate(tqdm(color_imgs)):
        cv2.imwrite(os.path.join(dir_name, f"feature{i}.png"), color_img)
            
    return cornerlist, descriptionlist



if __name__ == "__main__":
    dir_name, threshold, local, skip = parseArgs()

    print("\n[*] Reading images...")
    imgs, focal_lengths = readImages(dir_name)
    
    # focal length
    focal_lengths = [focal_mul*focal for focal in focal_lengths]

    print("\n[*] Cylinder reconstructing...")
    with Pool(len(imgs)) as pool:
        color_imgs = pool.starmap(cylinder_reconstructing, zip(imgs, focal_lengths))

    for i in range(len(color_imgs)):
        cv2.imwrite(os.path.join(dir_name, f"cylinder{i+1}.png"), color_imgs[i])
    #color_imgs = color_imgs[:8]
#    color_imgs = color_imgs[::-1]
    if not skip:
        
        resized_imgs = [cv2.resize(img, (img.shape[1]//resize_rate, img.shape[0]//resize_rate)) for img in color_imgs]
        gray_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in resized_imgs]

        cornerlist, descriptionlist = featureDetection(resized_imgs, gray_imgs, threshold=threshold, local=local)
        print("\n[*] Matching......")

        transforms = []
        cur_transform = np.identity(3)
        transforms.append(cur_transform.copy())

        for i in range(0,len(gray_imgs)-1):
          index_pairs = feature_matching(descriptionlist[i],descriptionlist[i+1])
          index_set1 = [pair[0] for pair in index_pairs]
          index_set2 = [pair[1] for pair in index_pairs]
          transform_matrix = image_matching([cornerlist[i][index] for index in index_set1],[cornerlist[i+1][index] for index in index_set2])
          cur_transform = np.matmul(cur_transform, transform_matrix)
          transforms.append(cur_transform.copy())
          print(gray_imgs[i].shape)
          img1 = np.transpose(cv2.warpPerspective(src = np.transpose(gray_imgs[i+1]), M = cur_transform, dsize = (gray_imgs[i].shape[0],5*gray_imgs[i].shape[1])))
          cv2.imwrite(os.path.join(dir_name, f"test{i+1}.png"), img1)


        
        with open(os.path.join(dir_name, f"transform-{local}-{threshold}"), "wb") as f:
            pickle.dump(transforms, f)

    else:
        with open(os.path.join(dir_name, f"transform-{local}-{threshold}"), "rb") as f:
            transforms = pickle.load(f)
        #print(transforms)
        for transform in transforms:
            transform[0:2, 2] *= resize_rate
            
    
    pano = image_blending(color_imgs, transforms)
    cv2.imwrite(os.path.join(dir_name, f"mypano-{local}-{threshold}.png"), pano)
