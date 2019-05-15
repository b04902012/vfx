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

from feature_describing import feature_describing
from feature_matching import feature_matching
from image_matching import image_matching
from cylinder_reconstructing import cylinder_reconstructing
from image_blending import image_blending

def parseArgs():
    args, ignore = getopt.getopt(sys.argv[1:], "f:", ["file="])
    args = dict(args)

    dir_name = args.get('-f') or args.get('--file')

    if not dir_name:
        sys.exit('Please provide directory name with -f or --file.')

    return dir_name


def resizeImages(dir_name):
    """
    Read <dir_name>/pano.txt to find all the images in <dir_name>, return the images
    
    Args:
        dir_name (str)
    
    Returns:
        list of cv2 color images
    """
    print('* Reading images...')
    
    imgs = []
    fls =[]
    with open(os.path.join(dir_name, "pano_original.txt")) as f:
        for image_name in f.readlines()[::13]:
            image_name = image_name.split('\\')[-1]
            full_name = os.path.join(dir_name, image_name.strip())
            print('  -', full_name)
#            imgs.append(cv2.imread(full_name))
            img = cv2.imread(full_name)
            img = cv2.resize(img, (img.shape[1]//10, img.shape[0]//10))
            cv2.imwrite(os.path.join(dir_name, 'resized_'+image_name.strip()), img)

if __name__ == "__main__":
    dir_name = parseArgs()

    resizeImages(dir_name)
