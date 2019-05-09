def transform(transform_matrix, coordinate):
    new_coordinate = np.matmul(transform_matrix,np.transpose(np.array([coordinate[0],coordinate[1],1])))
    return np.array([new_coordinate[0]/new_coordinate[2],new_coordinate[1]/new_coordinate[2]])

def image_blending(images, transforms):
    h, w, channels = images[0].shape
    all_h = h+200
    all_w = w*len(images)+50
    outputImage = np.zeros((all_h, all_w, 3))
    xi = 100
    yi = 10
    
    cur_transform = transforms[0]
    outputImage[xi:xi+h, yi:yi+w] = images[0]
    last_y = yi+w

    for i in range(1, len(images)):
        cur_transform = np.matmul(cur_transform, transforms[i])
        img = np.transpose(cv2.warpPerspective(src = np.transpose(images[i]), M = cur_transform, dsize = (all_h, all_w)))
        (x1, y1), (x2, y2) = transform(cur_transform, [0, 0]), transform(cur_transform, [h, 0])
        (x3, y3), (x4, y4) = transform(cur_transform, [0, w]), transform(cur_transform, [h, w])

        this_y = max(y1, y2)
        linear_width = last_y - this_y
        
        for y in range(this_y, last_y):
            alpha = (y-this_y)/linear_width
            for x in range(all_h):
                if np.any(img[x, y]) and np.any(outputImage[x, y]):
                    outputImage[x, y] = outputImage[x, y]*(1-alpha) + img[x, y]*alpha
        outputImage[xi+max(x1, x3):xi+min(x2, x4), yi+last:yi+max(y3, y4)] = img[max(x1, x3):min(x2, x4), last:max(y3, y4)]
        last_y = min(y3, y4)

    return outputImage
