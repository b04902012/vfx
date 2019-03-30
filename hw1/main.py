import numpy as np
import cv2
import myhdr as hdr
import os
import pickle
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        dir_name = input('Please input image name: ')
    else:
        dir_name = sys.argv[1]

    if input('Already has radiance map? [y/n] ') in ('y', 'Y'):
        rad_img = hdr.loadHDR(os.path.join(dir_name, dir_name+'_rad'))
    else:
        rad_img = hdr.computeHDR(dir_name)

    if input('Do tone mapping? [y/n] ') in ('y', 'Y'):
        
        print(np.amax(rad_img))
        print(np.amin(rad_img))

        inputs = input('Please input key value and multi value: ').split()

        key_value = 0.18
        multi_value = 1

        if len(inputs) > 0:
            key_value = float(inputs[0])
        if len(inputs) > 1:
            multi_value = float(inputs[1])
        tone_mapped = hdr.photographic(rad_img, key_value, multi_value)

        print(np.amax(tone_mapped))
        print(np.amin(tone_mapped))

        cv2.imwrite(os.path.join(dir_name, dir_name+f'_toned_{key_value}_{multi_value}.png'), tone_mapped)
        print(f'Save image to {dir_name}/{dir_name}_toned_{key_value}_{multi_value}.png')
        
