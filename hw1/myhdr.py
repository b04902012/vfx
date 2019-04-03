import numpy as np
import cv2
import random
import pickle
import os
import math
from scipy import signal
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt

try:
    from tqdm import tqdm
    tqdm_exist = True
except:
    tqdm_exist = False


### HDR ###

def loadHDR(filename):
    """ Load existent radiance map (in pickle format)
    
    Args:
        filename (str)
    
    Returns:
        numpy.array((height, width, nChannels), dtype=float): A radiance map with shape (h, w, 3)
    """
    with open(filename, 'rb') as f:
        rad_img = pickle.load(f)
    return rad_img

def writeHDRFile(image, filename):
    """ Save image to .hdr format
    
    Args:
        image (numpy.array((height, width, nChannels), dtype=float)): The computed radiance map in B, G, R order
        filename (str): Output filename (without the .hdr extention)
    """
    with open(filename+".hdr", "wb") as f:
        f.write("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n".encode())
        f.write(f"-Y {image.shape[0]} +X {image.shape[1]}\n".encode())

        brightest = np.maximum(np.maximum(image[...,2], image[...,1]), image[...,0])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 256.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[...,0:3] = np.around(image[...,::-1] * scaled_mantissa[...,None])
        rgbe[...,3] = np.around(exponent + 128)

        rgbe.ravel().tofile(f)
    return

def printFigure(img):
    """ For debugging purpose
    
    Args:
        img (numpy.array)
    """
    for r in range(0, img.shape[0], 10):
        for c in range(0, img.shape[1], 20):
            print(img[r, c:c+5])

    print("max =", np.amax(img))
    print("min =", np.amin(img))

    cv2.imwrite('image.png',img)
    cv2.imshow('image',img)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

def weight(pixel):
    """ Assign weight for every pixel
    
    Args:
        pixel (int): 0~255
    
    Returns:
        int: weight
    """
    z_min, z_max = 0, 255
    if pixel <= (z_min+z_max) / 2:
        return pixel - z_min
    return z_max - pixel


def getSamples(images, z_min = 0, z_max = 255):
    """ According to the middle image, choose one pixel for each pixel possible value 
        and return the pixels from all series of images.
    Args:
        images (numpy.array((nImages, height, width), dtype=uint8)): 
            images with different exposures
        z_min (int, optional): 
            minimum of possible pixel value
        z_max (int, optional): 
            maximum of possible pixel value
    
    Returns:
        numpy.array((nSamples, nImages), dtype=uint8): sampled pixels
    """

    z_range = z_max - z_min + 1
    nImages = len(images)
    nSamples = z_range
    intensity_values = np.zeros((nSamples, nImages), dtype=np.uint8)

    mid_img = images[nImages // 2]

    for i in range(nSamples):
        height, width = np.where(mid_img == i)
        if len(height) != 0:
            idx = random.randrange(len(height))
            for j in range(nImages):
                intensity_values[i, j] = images[j][height[idx], width[idx]]
    return intensity_values
    


def getResponseCurve(Z, log_t, smooth_lambda, weighting_func, z_min = 0, z_max = 255):
    """ Recover response curve for a single channel
    
    Args:
        Z (numpy.array((nSamples, nImages), dtype=uint8)): 
            sampled pixels
        log_t (float): 
            log of shutter speed
        smooth_lambda (float): 
            scalar that weights the smoothness term relative to the data fitting term
        weighting_func (function): 
            decides the weight for each pixel value from different images
        z_min (int, optional): 
            minimum of possible pixel value
        z_max (int, optional): 
            maximum of possible pixel value
    
    Returns:
        numpy.array((z_max-z_min+1), dtype=float): response curve that maps z_min ~ z_max to logE
    """
    n = z_max - z_min + 1

    nSamples = Z.shape[0]
    nImages = len(log_t)

    A = np.zeros((nSamples * nImages + n + 1, n + nSamples))
    B = np.zeros((A.shape[0], 1))

    k = 0
    for i in range(nSamples):
        for j in range(nImages):
            w_ij = weighting_func(Z[i, j]+1)
            A[k, Z[i, j]] = w_ij
            A[k, n+i] = -w_ij
            B[k, 0] = w_ij * log_t[j]
            k += 1

    A[k, n//2] = 1
    k += 1

    for i in range(n-2):
        tmp = smooth_lambda * weighting_func(i+1)
        A[k, i]     = tmp
        A[k, i+1]   = -2 * tmp
        A[k, i+2]   = tmp
        k += 1

    x = np.linalg.lstsq(A, B)[0]

    g = x[:n]

    return g[:, 0]


def getRadianceMap(images, log_t, response_curve, weighting_func):
    """ Get radiance map for a single channel
    
    Args:
        images (numpy.array((nImages, height, width), dtype=uint8)): 
            images with different exposures
        log_t (float): 
            log of shutter speed
        response_curve (numpy.array((z_max-z_min+1), dtype=float)): 
            response curve for this channel that maps z_min ~ z_max to logE
        weighting_func (function): 
            decides the weight for each pixel value from different images
    
    Returns:
        numpy.array((height, width), dtype=float): radiance map for a single channel
    """
    print("Computing radiance map...", end="         ")
    height, width = images[0].shape
    rad_img = np.zeros((height, width))
    nImages = len(images)

    for r in range(height):
        for c in range(width):
            weight = np.array([weighting_func(images[i][r, c]) for i in range(nImages)])
            weight_sum = np.sum(weight)
            log_e = np.array([response_curve[images[i][r, c]] - log_t[i] for i in range(nImages)])
            weighted_e = np.dot(weight, log_e)

            if weight_sum != 0:
                weighted_e /= weight_sum
            else:
                weighted_e = response_curve[images[nImages//2][r, c]] - log_t[nImages//2]
            rad_img[r, c] = 2 ** weighted_e

    print("done.")
    return rad_img

def computeHDR(dir_name, imgs, log_t):
    """ Main function for HDR imaging. 
        Performs HDR rendering, saves response curve figures, saves .hdr file, 
        saves pickle file for radiance map and returns it.
    
    Args:
        dir_name (str): 
            directory name that files will be saved into
        imgs (numpy.array((nImages, height, width, nChannels), dtype=uint8)): 
            images used for HDR imaging
        log_t (float): 
            log of shutter speed
    
    Returns:
        numpy.array((height, width, nChannels), dtype=float): radiance map
    """

    # resize images
    imgs    = [cv2.resize(img, (0, 0), fx=1/round(max(img.shape)/1000), fy=1/round(max(img.shape)/1000), interpolation=cv2.INTER_AREA) for img in imgs]
    # split into BGR
    channels = [[img[:, :, i] for img in imgs] for i in range(3)]
    rad_img = np.zeros(imgs[0].shape)    
    print("image shape:", imgs[0].shape)

    color = ['blue', 'green', 'red']
    responseCurve = []
    for i in range(3):
        print()
        print("=====", color[i], "=====")
        samples = getSamples(channels[i])

        responseCurve.append(getResponseCurve(samples, log_t, 100, weight))

        plt.plot(responseCurve[i], np.arange(256), c=color[i])
        plt.savefig(f'{dir_name}/curve_{i+1}.png')
        plt.clf()

        rad_img[..., i] = getRadianceMap(channels[i], log_t, responseCurve[i], weight)
        print(rad_img[:, :, i])
        print("max =", np.amax(rad_img[:, :, i]))
        print("min =", np.amin(rad_img[:, :, i]))

    for i in range(3):
        plt.plot(responseCurve[i], np.arange(256), c=color[i])
    plt.savefig(f'{dir_name}/curve_{4}.png')
    plt.clf()
    
    with open(os.path.join(dir_name, dir_name+'_rad'), 'wb') as f:
        pickle.dump(rad_img, f)

    writeHDRFile(rad_img, os.path.join(dir_name, dir_name))
    
    cv2.imwrite(os.path.join(dir_name, dir_name+'.png'), rad_img)

    return rad_img
    


### TONE MAPPING ###

def toGrayScale(color_img):
    """ Convert color image into greyscale
    
    Args:
        color_img (numpy.array((height, width, 3)))
    
    Returns:
        numpy.array((height, width)): greyscale image
    """
    return 0.27 * color_img[:, :, 2] + 0.67 * color_img[:, :, 1] + 0.06 * color_img[:, :, 0]

def toLuminance(img, key_value):
    """ Compute Lw (luminance) and Lm (scaled luminance)
    
    Args:
        img (numpy.array(height, width, 3)): color image
        key_value (float): user specified scalar
    
    Returns:
        (numpy.array(height, width), numpy.array(height, width)): Lw, Lm
    """
    delta   = 0.001
    Lw      = toGrayScale(img)
    L_w     = np.log2(delta + Lw)
    L_w     = np.exp2(np.sum(L_w)/img.shape[0]/img.shape[1], out=L_w)
    L_m     = key_value / L_w * Lw

    print("Lw:")
    print(np.amax(Lw))
    print(np.amin(Lw))
    print()

    return Lw, L_m

def gaussianBlur(img, sigma):
    """ Generate a gaussian blurred image
    
    Args:
        img (numpy.array(height, width)): greyscale image
        sigma (float): standard deviation of the gaussian
    
    Returns:
        numpy.array(height, width): a blurred image
    """
    blurred = gaussian_filter(img, sigma=sigma)

    return blurred

def photographicGlobal(rad_img, key_value = 0.18, multi_value = 1):
    """ Performs global operator on the radiance map
    
    Args:
        rad_img (numpy.array((height, width, 3)): radiance map
        key_value (float, optional): a
        multi_value (int, optional): adjust global luminance
    
    Returns:
        numpy.array((height, width, 3)): a tone-mapped image
    """
    Lw, L_m = toLuminance(rad_img, key_value)
    L_white = np.amax(L_m)
    L_d     = L_m * (1 + L_m/(L_white ** 2)) / (1 + L_m)
    print(L_d)
    print(np.amax(L_d))
    print(np.amin(L_d))

    channels = [np.expand_dims(rad_img[:, :, i]/Lw*L_d, 2) for i in range(3)]

    tone_mapped_img = np.concatenate(channels, 2)

    print()
    print('After tone mapping:')
    print('max:', np.amax(tone_mapped_img))
    print('min:', np.amin(tone_mapped_img))

    # adjust intensity
    tone_mapped_img *= 100 / np.average(tone_mapped_img) * multi_value


    print(Lw[200][200], L_d[200][200])
    print(Lw[300][300], L_d[300][300])
    print(Lw[400][400], L_d[400][400])

    return tone_mapped_img

def computeV(x, y, gaussian, phi, key_value):
    """ Compute local luminance and local contrast
    
    Args:
        x, y (int): position of the pixel
        gaussian (tuple): (s, [blurred_img_alpha1, blurred_img_alpha2])
        phi (float): user-specified scalar
        key_value (float): a
    
    Returns:
        (float, float): local luminance and local contrast
    """
    V_i = lambda x, y, i: gaussian[1][i][y][x]

    V1 = V_i(x, y, 0)
    V2 = V_i(x, y, 1)

    return (V1 - V2)/(2**phi * key_value / gaussian[0]**2 + V1), V1


def photographicLocal(rad_img, key_value, multi_value, eps, s, phi):
    """ Performs lobal operator on the radiance map
    
    Args:
        rad_img (numpy.array((height, width, 3)): radiance map
        key_value (float, optional): a
        multi_value (int, optional): adjust global luminance
        eps (float): the threshold for local contrast, user-specified
        s (float): the initial value for local range
        phi (float): user-specified scalar
    
    Returns:
        numpy.array((height, width, 3)): a tone-mapped image
    """
    Lw, L_m = toLuminance(rad_img, key_value)
    maxV1   = np.zeros(Lw.shape)

    sqrt2 = math.sqrt(2)
    alpha = [1 / 2 / sqrt2, 1.6 / 2 / sqrt2]
    gaussians = []       # (s, [img_alpha1, img_alpha2])
 
    for _ in range(8):
        gaussians.append((s, [gaussianBlur(Lw, alpha[i]*s/sqrt2) for i in range(2)]))
        s *= 1.6


    height, width, *rest = rad_img.shape
    for y in (tqdm(range(height)) if tqdm_exist else range(height)):
        for x in range(width):
            maxV = Lw[y][x]
            for gaussian in gaussians:
                delta, V1 = computeV(x, y, gaussian, phi, key_value)
                if abs(delta) < eps:
                    maxV = V1
                else:
                    break
            maxV1[y][x] = maxV

    L_d     = L_m / (1 + maxV1)
    print("L_d:")
    print(L_d)
    print(np.amax(L_d))
    print(np.amin(L_d))
    print()

    channels = [np.expand_dims(rad_img[:, :, i]/Lw*L_d, 2) for i in range(3)]
    tone_mapped_img = np.concatenate(channels, 2)
    tone_mapped_img *= 100 / np.average(tone_mapped_img) * multi_value

    print(Lw[200][200], L_d[200][200])
    print(Lw[300][300], L_d[300][300])
    print(Lw[400][400], L_d[400][400])

    return tone_mapped_img
