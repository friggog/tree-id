import glob
import os
from math import ceil, floor, sqrt

import cv2
import numpy as np
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')

C_BINS = 32
C_MIN = 10
C_MAX = 40
C_STEP = 5

def normalise_contour(contour):
    # resize the contour so the long side is 300 long
    # position doesnt matter as curvature is then based on differences
    # rotation invariance is built into curvature
    rect = cv2.minAreaRect(contour)
    h, w = rect[1]
    s = 300 / max(h, w)
    contour = contour * s
    return contour


def get_curvature_map(line, segment, scale=25):
    curv = []
    # for i in range(len(line)):
    #     if i - h < - len(line) + 1:
    #         m = line[i - h + len(line)][0]
    #     else:
    #         m = line[i - h][0]
    #     c = line[i][0]
    #     if i > len(line) - h - 1:
    #         p = line[i + h - len(line)][0]
    #     else:
    #         p = line[i + h][0]
    #     f = (p - c) / pow(h, 2)
    #     ff = (p - 2 * c + m) / pow(h, 2)
    #     k = (f[0] * ff[1] - f[1] * ff[0]) / pow(pow(f[0], 2) + pow(f[1], 2), 1.5)
    #     curv.append(k)
    # curv = np.nan_to_num(curv)
    # curv = np.clip(curv, -200, 200)
    # return curv
    r = scale
    ha = np.pi * pow(r,2) / 2
    # line = line.reshape((-1,1,2))
    for i in range(0, len(line), max(int(len(line)/100),1)):
        mask = np.zeros(segment.shape[:2], np.uint8)
        c = line[i][0]
        cv2.circle(mask, (int(c[0]),int(c[1])), r, (255,255,255), -1)
        res = cv2.bitwise_and(mask, segment)
        o = (np.sum(res/255) - ha) / ha
        curv.append(o)
    #     cv2.imshow('asd',mask)
    #     cv2.waitKey(0)
    # plt.plot(curv)
    # plt.show()
    return curv

# FEATURES #

def f_curvature_stat(curvature_map):
    out = []
    # BENDING ENERGY
    s = 0
    for k in curvature_map:
        s += pow(k, 2)
    s /= len(curvature_map)
    out.append(s)
    # MEAN CURVATURE
    out.append(np.mean(curvature_map))
    # STD CURVATURE
    out.append(np.std(curvature_map))
    return out


def f_curvature_hist(contour, segment):
    n_bins = C_BINS
    hists = []
    for h in range(C_MIN, C_MAX, C_STEP):
        curvature_map = get_curvature_map(contour, segment, scale=h)
        hist = np.histogram(curvature_map, bins=n_bins, density=True)
        hists.extend(hist[0])
    return hists


def f_bending_bins(curvature_map):
    n_bins = 7
    bin_size = int(len(curvature_map) / n_bins)
    vs = []
    for i in range(n_bins):
        s = 0
        for j in range(bin_size):
            s += pow(curvature_map[i * bin_size + j], 2)
        vs.append(s)
    vs /= np.max(vs)
    return vs


def f_basic_shape(cnt, a, l):
    # SOLIDITY
    hull = cv2.convexHull(cnt)
    ha = cv2.contourArea(hull)
    solidity = 1
    if ha != 0:
        solidity = a / ha
    # CONVEXITY
    convexity = cv2.arcLength(hull, False) / l
    # SQUARENESS
    rect = cv2.minAreaRect(cnt)
    w, h = rect[1]
    ratio = min(w, h) / max(w, h)
    return [solidity, convexity, ratio]

# EXTRACTION #

def isolate_leaf(image):
    h, w = image.shape[:2]
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, np.median(grey) * 0.6, 255, cv2.THRESH_BINARY_INV)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, sat, vib = cv2.split(hsv)
    _, sat = cv2.threshold(sat, 77, 255, cv2.THRESH_BINARY)
    if np.mean(sat) < 200:
        thresh = np.add(sat, thresh)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    curr = (None, 0, 0)
    for cnt in contours:
        a = cv2.contourArea(cnt)
        l = cv2.arcLength(cnt, False)
        if a < 300 or l < 300:
            continue
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        dc = sqrt(pow(abs(256 - cx), 2) + pow(abs(256 - cy), 2))
        if a > curr[1] and dc < (h * w / 1500):
            curr = (cnt, a, l)
    if curr[0] is None:
        return None, None, None, None, None
    segment = np.zeros((h,w), np.uint8)
    cv2.drawContours(segment, [curr[0]], -1, color=(255,255,255), thickness=-1)
    return grey, segment, curr[0], curr[1], curr[2]

def get_features(path, show=False):
    image = cv2.imread(path)
    h,w = image.shape[:2]
    grey, segmentation, contour, area, length = isolate_leaf(image)
    if contour is None:
        return []
    contour = contour.reshape((-1,2))
    x1, y1 = np.min(contour, axis=0)
    x2, y2 = np.max(contour, axis=0)
    o = 5
    ymin = max(y1-o,0)
    ymax = min(y2+o,h)
    xmin = max(x1-o,0)
    xmax = min(x2+o,w)
    image = image[ymin:ymax,xmin:xmax]
    grey = grey[ymin:ymax,xmin:xmax]
    segmentation = segmentation[ymin:ymax,xmin:xmax]
    grey = cv2.bitwise_and(grey, grey, mask=segmentation)
    contour[:,0] -= xmin
    contour[:,1] -= ymin
    contour = contour.reshape((-1, 1, 2))
    if show:
        # [vx,vy,x,y] = cv2.fitLine(curr[0], cv2.DIST_L2,0,0.01,0.01)
        # lefty = int((-x*vy/vx) + y)
        # righty = int(((w-x)*vy/vx)+y)
        # cv2.line(image,(w-1,righty),(0,lefty),(0,255,0),2)
        # cv2.drawContours(image, contours, -1, (0, 0, 255))
        # cv2.drawContours(image, [contour], -1, (255, 255, 0))
        cv2.imshow(path, segmentation)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    curvature_map = get_curvature_map(contour, segmentation)
    f = []
    f.extend(f_basic_shape(contour, area, length))
    f.extend(f_curvature_stat(curvature_map))
    # f.extend(f_bending_bins(curvature_map))
    f.extend(f_curvature_hist(contour, segmentation))
    return f


def extract(env, show=False, limit=-1, setup=None):
    print('EXTRACTING')
    images = {}
    count = 0
    global C_BINS, C_MIN, C_MAX, C_STEP
    if setup is not None:
        C_BINS, C_MIN, C_MAX, C_STEP = setup
    for species_path in sorted(glob.glob('dataset/images/' + env + '/*')):
        species = species_path.split('/')[-1]
        new_path = 'dataset/features/' + env + '/' + species
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        for i, image_path in enumerate(sorted(glob.glob(species_path + '/*'))):
            f_path = new_path + '/' + image_path.split('/')[-1].split('.')[0]
            features = get_features(image_path, show)
            np.save(f_path, features)
            count += 1
            print('Done:', count, end="\r")
            if show or (limit > 0 and i >= limit):
                break
