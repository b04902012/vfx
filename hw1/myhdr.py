import numpy as np
import cv2
import random
import pickle
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
        log_t = np.log2(log_t)
        [print("-", f) for f in filenames]

    # cv2.imread flag: grayscale -> =0, BGR -> >0
    imgs            = [cv2.imread(dir_name+f.replace("ppm", "png"), 1) for f in filenames]
    resized_imgs    = [cv2.resize(img, (0, 0), fx=1/round(max(img.shape)/1000), fy=1/round(max(img.shape)/1000), interpolation=cv2.INTER_AREA) for img in imgs]

    print("done.\n")
    return resized_imgs, log_t

def loadHDR(filename):
    with open(filename, 'rb') as f:
        rad_img = pickle.load(f)
    return rad_img

def writeHDRFile(image, filename):
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
    return rgbe

def printFigure(img):
    for r in range(0, img.shape[0], 10):
        for c in range(0, img.shape[1], 20):
            print(img[r, c:c+5])

    print("max =", np.amax(img))
    print("min =", np.amin(img))

    cv2.imwrite('image.png',img)
    cv2.imshow('image',img)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

### HDR ###
def weight(pixel):
    z_min, z_max = 0, 255
    if pixel <= (z_min+z_max) / 2:
        return pixel - z_min
    return z_max - pixel


def getSamples(images, z_min = 0, z_max = 255):
    '''
    return np.array of shape(nSamples, nImages)
    '''

    z_range = z_max - z_min + 1
    nImages = len(images)
    nSamples = z_range
    intensity_values = np.zeros((nSamples, nImages), dtype=np.uint8)

    '''
    rows = np.random.randint(images[0].shape[0], size=(nSamples))
    cols = np.random.randint(images[0].shape[1], size=(nSamples))
    
    for i in range(len(rows)):
        for j in range(nImages):
            intensity_values[i, j] = images[j][rows[i], cols[i]]

    return intensity_values
    
    '''
    mid_img = images[nImages // 2]

    for i in range(nSamples):
        rows, cols = np.where(mid_img == i)
        if len(rows) != 0:
            idx = random.randrange(len(rows))
            for j in range(nImages):
                intensity_values[i, j] = images[j][rows[idx], cols[idx]]
    return intensity_values
    


def getResponseCurve(Z, log_t, smooth_lambda, weighting_func, z_min = 0, z_max = 255):
    n = z_max - z_min + 1

    nSamples = Z.shape[0]
    nImages = len(log_t)

    A = np.zeros((nSamples * nImages + n + 1, n + nSamples))
    B = np.zeros((A.shape[0], 1))


    # Include the data-fitting equations

#    print(Z)
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
    print(x[n//2])
    return g[:, 0]


def getRadianceMap(images, log_t, response_curve, weighting_func):
    print("Computing radiance map......", end="         ")
    rows, cols = images[0].shape
    rad_img = np.zeros((rows, cols))
    nImages = len(images)

    for r in range(rows):
        for c in range(cols):
            log_e = 0
            weight = np.array([weighting_func(images[i][r, c]) for i in range(nImages)])
            weight_sum = np.sum(weight)
            log_e = np.array([response_curve[images[i][r, c]] - log_t[i] for i in range(nImages)])
            weighted_e = np.dot(weight, log_e)

            if weight_sum != 0:
                weighted_e /= weight_sum
            else:
                weighted_e = response_curve[images[nImages//2][r, c]] - log_t[nImages//2]
            rad_img[r, c] = 2 ** weighted_e
#            print(log_e)
#            rad_img[r, c] = log_e
    print("done.")
    return rad_img

def computeHDR(dir_name):
    imgs, log_t = readImages(dir_name+'/', '.hdr_image_list.txt')
    channels = [[img[:, :, i] for img in imgs] for i in range(3)]
    rad_img = np.zeros(imgs[0].shape)    
    print("image shape:", imgs[0].shape)

    color = ['blue', 'green', 'red']
    responseCurve = []
    for i in range(3):
        print()
        print("=====", color[i], "=====")
        samples = getSamples(channels[i])
#       print(log_t)
        responseCurve.append(getResponseCurve(samples, log_t, 100, weight))
#       print(responseCurve[i][128])
        plt.plot(responseCurve[i], np.arange(256), c=color[i])
        plt.savefig(f'{dir_name}/curve_{i+1}.png')
        plt.clf()
#       plt.show(block='false')
        rad_img[..., i] = getRadianceMap(channels[i], log_t, responseCurve[i], weight)
        print(rad_img[:, :, i])
        print("max =", np.amax(rad_img[:, :, i]))
        print("min =", np.amin(rad_img[:, :, i]))
    '''
    print(rad_img.shape)
    '''
    for i in range(3):
        plt.plot(responseCurve[i], np.arange(256), c=color[i])
    plt.savefig(f'{dir_name}/curve_{4}.png')
    plt.clf()
    
    with open(dir_name+'/'+dir_name+'_rad', 'wb') as f:
        pickle.dump(rad_img, f)

    writeHDRFile(rad_img, dir_name+'/'+dir_name)
    

    '''
    for r in range(0, 768, 10):
        for c in range(0, 502, 20):
            print(rad_img[r, c:c+5])

    for r in range(0, 768, 10):
        for c in range(0, 502, 20):
            print(rad_img[r, c:c+5])
    '''
    
    cv2.imwrite(dir_name+'/'+dir_name+'.png',rad_img)
#    cv2.imshow('image',rad_img)
#    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

    return rad_img
    


### tone mapping ###
def photographic(rad_img, key_value = 0.18, multi_value = 1):
    '''
    rad_img: (rowls, cols, 3)
    '''
    delta   = 0.001

    Lw      = 0.27*rad_img[:, :, 0] + 0.67*rad_img[:, :, 1] + 0.06*rad_img[:, :, 2]
    #Lw = 0.3*rad_img[:, :, 0] + 0.59*rad_img[:, :, 1] + 0.11*rad_img[:, :, 2]
    L_w  = np.log2(delta + Lw)

    L_w     = np.exp2(np.sum(L_w)/rad_img.shape[0]/rad_img.shape[1], out=L_w)
    print(L_w)
    L_m     = key_value / L_w * Lw
    L_white = np.amax(L_m)
    L_d     = L_m * (1 + L_m/(L_white ** 2)) / (1 + L_m)
    print(L_d)
    print(np.amax(L_d))
    print(np.amin(L_d))

    tone_mapped_img = rad_img * L_w
    channels = [np.expand_dims(tone_mapped_img[:, :, i]/L_d, 2) for i in range(3)]

    tone_mapped_img = np.concatenate(channels, 2)

    # adjust intensity
    tone_mapped_img *= 128 / np.average(tone_mapped_img) * multi_value

#    printFigure(tone_mapped_img)

    return tone_mapped_img
