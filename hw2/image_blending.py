import numpy as np
import cv2

def transform(transform_matrix, coordinate):
    new_coordinate = np.matmul(transform_matrix,np.transpose(np.array([coordinate[0],coordinate[1],1])))
    return np.array([int(new_coordinate[0]/new_coordinate[2]),int(new_coordinate[1]/new_coordinate[2])])

def image_blending(images, transforms):
    print(images[0])
    print(len(images))
    print(images[1])
    h, w, channels = images[0].shape
    all_h = h+200
    all_w = w*len(images)+50
    outputImage = np.zeros((all_h, all_w, 3))
    xi = 100
    yi = 10
    
    last_y = 0

    for i in range(len(images)):
        img = np.zeros((all_h, all_w, 3))
        for channel in range(3):
            img[:, :, channel] = np.transpose(cv2.warpPerspective(src = np.transpose(images[i][:, :, channel]), M = transforms[i], dsize = (all_h, all_w)))
        (x1, y1), (x2, y2) = transform(transforms[i], [0, 0]), transform(transforms[i], [h, 0])
        (x3, y3), (x4, y4) = transform(transforms[i], [0, w]), transform(transforms[i], [h, w])

        this_y = y1
        linear_width = last_y - this_y
        linear_width = min(linear_width, h//5)
        print(last_y, this_y)

        for y in range(last_y-linear_width, last_y):
            alpha = (y-last_y+linear_width)/linear_width
            outputImage[xi:xi+h, y+yi] = outputImage[xi:xi+h, y+yi]*(1-alpha) + img[:h, y]*alpha

        #print(xi+max(x1, x3), xi+min(x2, x4), yi+last_y, yi+y3)
        #print(max(x1, x3), min(x2, x4), last_y, y3)
        outputImage[xi+max(x1, x3, 0):xi+min(x2, x4, h), yi+last_y:yi+y3] = img[max(x1, x3, 0):min(x2, x4, h), last_y:y3]
        last_y = y3

    return outputImage[:, :last_y]
