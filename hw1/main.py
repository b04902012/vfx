import numpy as np
import cv2
import myhdr as hdr
import myalign as align
import os
import pickle
import sys

### I/O ###
def readImages(dir_name, txt_name):
    print("Reading images...")
    with open(os.path.join(dir_name,txt_name), 'r') as f:
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
    #resized_imgs    = [cv2.resize(img, (0, 0), fx=1/round(max(img.shape)/1000), fy=1/round(max(img.shape)/1000), interpolation=cv2.INTER_AREA) for img in imgs]

    print("done.\n")
    return imgs, log_t

if __name__ == '__main__':
    if len(sys.argv) < 2:
        dir_name = input('Please input image name: ')
    else:
        dir_name = sys.argv[1]

    if input('Already has radiance map? [y/n] ') in 'Yy':
        rad_img = hdr.loadHDR(os.path.join(dir_name, dir_name+'_rad'))
    else:
        imgs, log_t = readImages(dir_name+'/', '.hdr_image_list.txt')
        imgs = align.alignImages(imgs, 6)
        rad_img = hdr.computeHDR(dir_name, imgs, log_t)

    if input('Do tone mapping? [y/n] ') in 'Yy':
        print('Before tone mapping:')
        print('max:', np.amax(rad_img))
        print('min:', np.amin(rad_img))
        print()

        inputs = input('Please input key value and multi value: ').split()

        key_value = 0.18
        multi_value = 1

        if len(inputs) > 0:
            key_value = float(inputs[0])
        if len(inputs) > 1:
            multi_value = float(inputs[1])

        #tone_mapped = hdr.photographicGlobal(rad_img, key_value, multi_value)
        tone_mapped = hdr.photographicLocal(rad_img, phi=8, key_value=1.8)
        blurred = hdr.gaussianBlur(tone_mapped, 4)

        print()
        print('After tone mapping:')
        print('max:', np.amax(tone_mapped))
        print('min:', np.amin(tone_mapped))

        cv2.imwrite(os.path.join(dir_name, dir_name+f'_toned_{key_value}_{multi_value}.png'), tone_mapped)
        cv2.imwrite(os.path.join(dir_name, dir_name+f'_blurred.png'), blurred)
        
        print(f'Save image to', os.path.join(dir_name, dir_name+f'_toned_{key_value}_{multi_value}.png'))
        
