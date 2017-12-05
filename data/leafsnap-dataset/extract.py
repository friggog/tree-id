import glob
import os
from math import ceil, floor, sqrt

import cv2
import numpy as np

import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')

def calculate_curvature(line):
    # TODO tune h
    out = []
    for h in range(10, 50, 5): # 10, 50, 5
        curv = []
        for i in range(h, len(line) - h):
            m = line[i - h][0]
            c = line[i][0]
            p = line[i + h][0]
            f = (p - c) / pow(h, 2)
            ff = (p - 2 * c + m) / pow(h, 2)
            k = (f[0] * ff[1] - f[1] * ff[0]) / pow(pow(f[0], 2) + pow(f[1], 2), 1.5)
            curv.append(k)
        # TODO tune bins
        bins = 64
        hist = np.histogram(curv, bins=bins, range=[-64, 64], density=True)
        hist = np.nan_to_num(hist[0])
        out.extend(hist)
    return out


def get_features(path, show=False, env='lab'):
    image = cv2.imread(path)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # if 'lab' in env:
    #     _, thresh = cv2.threshold(grey, np.median(grey) * 0.6, 255, cv2.THRESH_BINARY_INV)
    #     # maybe do this to remove stalks??
    #     # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    # else:
    _, thresh = cv2.threshold(grey, np.median(grey) * 0.55, 255, cv2.THRESH_BINARY_INV)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, sat, vib = cv2.split(hsv)
    _, sat = cv2.threshold(sat, 77, 255, cv2.THRESH_BINARY) # 0.6*(np.min(sat)+np.max(sat))/2
    if np.mean(sat) < 240:
        thresh = np.add(sat, thresh)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    curr = (None, 0)
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a == 0:
            continue
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        dc = sqrt(pow(abs(256-cx),2) + pow(abs(256-cy),2))
        if a > curr[1] and dc < 150:
            curr = (cnt, a)
    if curr[0] is None:
        if len(contours) > 0:
            curr = (contours[0], -1)
        else:
            return []
    if show:
        cv2.drawContours(image, contours, -1, (0, 0, 255))
        cv2.drawContours(image, [curr[0]], -1, (255, 255, 0))
        # plt.close()
        # plt.bar(range(bins), hist[0])
        # plt.show()
        cv2.imshow(path, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return calculate_curvature(curr[0])

def extract(env, show=False, limit=-1):
    print('EXTRACTING')
    images = {}
    count = 0
    for species_path in sorted(glob.glob('dataset/images/' + env + '/*')):
        species = species_path.split('/')[-1]
        new_path = 'dataset/features/' + env + '/' + species
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        for i, image_path in enumerate(sorted(glob.glob(species_path + '/*'))):
            f_path = new_path + '/' + image_path.split('/')[-1].split('.')[0]
            features = get_features(image_path, show, env)
            feature_path = np.save(f_path, features)
            count += 1
            print('Done:', count, end="\r")
            if show or (limit > 0 and i > limit):
                break
