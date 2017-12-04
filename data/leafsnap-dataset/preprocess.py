import glob
import os
from math import ceil, floor

import cv2
import numpy as np

MAX_P = 0.95
MIN_P = 0.7
B_Y_THRESH = 0.3
B_X_THRESH = 0.1
W_THRESH = 0.5
K_SIZE = 10


def cut_edges(show=False):
    for species_path in sorted(glob.glob('dataset/images/lab/*')):
        if not os.path.exists(species_path):
            os.makedirs(species_path)
        for image_path in sorted(glob.glob(species_path + '/*')):
            print(image_path)
            new_path = 'dataset/images/lab_p/' + '/'.join(image_path.split('/')[3:])
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            _, sat, vib = cv2.split(hsv)
            _, sat = cv2.threshold(sat, 30, 255, cv2.THRESH_BINARY)
            _, thresh = cv2.threshold(grey, 165, 255, cv2.THRESH_BINARY_INV)
            thresh = cv2.add(thresh, sat)
            thresh = cv2.dilate(thresh, None, iterations=1)
            y_cut = h
            last_y = 0
            for y in range(h, int(MIN_P * h), -1):
                v = np.mean(thresh[y - K_SIZE:y]) / 255
                if v < B_Y_THRESH and last_y > W_THRESH:
                    y_cut = y - int(K_SIZE / 2)
                    break
                else:
                    last_y = max(v, last_y)
            thresh = thresh[:y_cut]
            x_cut = w
            last_x = 0
            for x in range(w, int(MIN_P * w), -1):
                v = np.mean(thresh[..., x - K_SIZE:x]) / 255
                if v < B_X_THRESH and last_x > W_THRESH:
                    x_cut = x - int(K_SIZE / 2)
                    break
                else:
                    last_x = max(v, last_x)
            if show:
                cv2.line(image, (0, y_cut), (w, y_cut), (0, 0, 255), 2)
                cv2.line(image, (x_cut, 0), (x_cut, h), (0, 0, 255), 2)
                cv2.destroyAllWindows()
                cv2.moveWindow(image_path + 't', -100, -500)
                cv2.imshow(image_path, image)
                cv2.moveWindow(image_path, -100 + int(w * 1.1), -500)
                cv2.waitKey(0)
                break
            else:
                cropped = image[0:y_cut, 0:x_cut]
                cv2.imwrite(new_path, cropped)


def delete_dupes():
    for image_path in glob.glob('dataset/images/field/*/*'):
        if ' (1)' in image_path:
            print(image_path)
            os.remove(image_path)


def print_count(t):
    print('\n', t.upper())
    print('-' * 45)
    count = 0
    for species_path in sorted(glob.glob('dataset/images/' + t + '/*')):
        species = species_path.split('/')[-1]
        n = len(glob.glob(species_path + '/*'))
        count += n
        print((species + ': ').ljust(40), str(n).rjust(4))
    print('-' * 45)
    print('Total: '.ljust(35), str(count).rjust(9))


def resize(show=False):
    for species_path in sorted(glob.glob('dataset/images/field/*')):
        new_path = 'dataset/images/field_r/' + '/'.join(species_path.split('/')[3:])
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        for image_path in sorted(glob.glob(species_path + '/*')):
            new_img_path = new_path + '/' + image_path.split('/')[-1]
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            sqs = max(h, w)
            hd = (sqs - h) / 2
            wd = (sqs - w) / 2
            squared = cv2.copyMakeBorder(image, top=floor(hd), bottom=ceil(hd), left=floor(wd), right=ceil(wd), borderType=cv2.BORDER_REPLICATE)
            resized = cv2.resize(squared, (512, 512))
            if show:
                cv2.destroyAllWindows()
                cv2.imshow(image_path, resized)
                cv2.waitKey(0)
                break
            else:
                cv2.imwrite(new_img_path, resized)

# print_count('field')
# print_count('lab_p')
resize()

cv2.destroyAllWindows()
