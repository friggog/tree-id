import glob
import os
import sys
from math import ceil, floor, sqrt

import cv2
import numpy as np
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')


def get_curvature_map(line, segment, scale=0.1):
    curv = []
    l = 192 # 128 > 65, 256 > 66
    r = max(int(len(line) * scale), 2)
    a = np.pi * pow(r, 2)
    for i in range(0, len(line), max(floor(len(line) / l), 1)):
        mask = np.zeros(segment.shape[:2], np.uint8)
        c = line[i][0]
        cv2.circle(mask, (int(c[0]), int(c[1])), r, (255, 255, 255), -1)
        res = cv2.bitwise_and(mask, segment)
        o = np.sum(res) / a
        curv.append(o)
    c = np.array(curv, np.uint8)
    if len(c) != l:
        c = cv2.resize(c, (1, l)).reshape(l)
    return c


def write_curvature_image(path, curve, segment):
    maps = []
    for h in range(10, 106, 6):
        c = get_curvature_map(curve, segment, scale=h / 300)
        maps.append(c)
    maps = np.array(maps, np.uint8).reshape((16, 128))
    # cv2.imshow('o', maps)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(path, maps)

# FEATURES #
def f_glcm(image):
    gl = np.zeros((256, 256), np.uint8)
    h, w = image.shape[:2]
    for i in range(h - 1):
        for j in range(w - 1):
            if image[i, j] != 0 and image[i, j + 1] != 0:
                gl[image[i, j], image[i, j + 1]] += 1
                gl[image[i, j], image[i + 1, j]] += 1
    gl = gl / gl.sum() / 2
    energy = 0
    contrast = 0
    homogenity = 0
    IDM = 0
    entropy = 0
    for i in range(255):
        for j in range(255):
            energy += pow(gl[i, j], 2)
            contrast += (i - j) * (i - j) * gl[i, j]
            homogenity += gl[i, j] / (1 + abs(i - j))
            if i != j:
                IDM += gl[i, j] / pow(i - j, 2)
            if gl[i, j] != 0:
                entropy -= gl[i, j] * np.log10(gl[i, j])
            mean = + 0.5 * (i * gl[i, j] + j * gl[i, j])
    # print( 100*energy, contrast/200, homogenity, IDM, entropy/5)
    return 100 * energy, contrast / 200, homogenity, IDM, entropy / 5


def f_curvature_stat(curvature_map):
    out = []
    # BENDING ENERGY
    s = 0
    for k in curvature_map:
        s += pow(k, 2)
    s /= len(curvature_map)
    out.append(s)
    # MEAN
    out.append(np.mean(curvature_map))
    # STD
    out.append(np.std(curvature_map))
    # FIRST DIFF
    # mean of basic and absolute
    fd = np.diff(curvature_map, n=1)
    out.append(np.mean(fd))
    out.append(np.mean(np.abs(fd)))
    # SECOND DIFF
    # mean of basic and absolute
    sd = np.diff(curvature_map, n=2)
    out.append(np.mean(sd))
    out.append(np.mean(np.abs(sd)))
    return out


def f_curvature_hist(curvature_map):
    n_bins = 32
    hist = np.histogram(curvature_map, bins=n_bins, density=True)
    return hist[0]


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


def f_fft(cnt):
    f = np.abs(np.fft.rfft(cnt, n=len(cnt)))
    return f/f.max()

# EXTRACTION #
def isolate_leaf(image):
    h, w = image.shape[:2]
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, np.median(grey) * 0.6, 255, cv2.THRESH_BINARY_INV)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, sat, vib = cv2.split(hsv)
    _, sat = cv2.threshold(sat, 77, 255, cv2.THRESH_BINARY)
    # if saturation is valid then add it to the threshold
    if np.mean(sat) < 200:
        thresh = np.add(sat, thresh)
    # top hat it to remove stem, unless the leaf is very small
    if np.mean(thresh) > 2:
        thresh = cv2.subtract(thresh, cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, np.ones((3, 3), np.uint8)))
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    curr = (None, 0, 0)
    for cnt in contours:
        a = cv2.contourArea(cnt)
        # if a < (h * w) * 0.001:
        # print('A')
        # continue
        l = cv2.arcLength(cnt, False)
        if l < 30:
            continue
        M = cv2.moments(cnt)
        try:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        except:
            continue
        dc = sqrt(pow(abs(w / 2 - cx), 2) + pow(abs(h / 2 - cy), 2))
        if dc < (h * w / 750) and a > curr[1]:
            curr = (cnt, a, l)
    if curr[0] is None:
        return None, None, None, None, None
    # cv2.drawContours(image, [curr[0]], -1, color=(255,255,0))
    # cv2.imshow('d',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    segment = np.zeros((h, w), np.uint8)
    cv2.drawContours(segment, [curr[0]], -1, color=(255, 255, 255), thickness=-1)
    return grey, curr[0], curr[1], curr[2], segment


def get_features(path, show=False):
    image = cv2.imread(path)
    # c_map = cv2.imread(path.replace('images', 'curvature_maps'), 0)
    grey, contour, area, length, segmentation = isolate_leaf(image)
    if contour is None or segmentation is None:
        return None
    f = []
    f.extend(f_basic_shape(contour, area, length))
    for h in range(0, 15, 5):
        c = get_curvature_map(contour, segmentation, scale=(h*3+10)/300)
        # c = c_map[h,:]
        f.extend(f_fft(c))
        f.extend(f_curvature_stat(c/255))
    f = np.nan_to_num(f)
    return f


def get_curvature_maps(path):
    image = cv2.imread(path)
    c_path = path.replace('images', 'curvature_maps')
    if not os.path.exists(c_path):
        grey, contour, area, length, segmentation = isolate_leaf(image)
        if contour is not None and segmentation is not None:
            write_curvature_image(c_path, contour, segmentation)

def curve_map_for_mlp(path):
    image = cv2.imread(path)
    c_path = path.replace('images', 'mlp_features')
    if not os.path.exists(c_path):
        grey, contour, area, length, segmentation = isolate_leaf(image)
        if contour is not None and segmentation is not None:
            c = get_curvature_map(contour, segmentation, scale=0.05)
            np.save(c_path, c)
    

def extract(env, limit=-1, step=1, base=0, mode=0, show=False):
    images = {}
    count = 0
    for species_path in sorted(glob.glob('dataset/images/' + env + '/*')):
        if mode == 0:
            new_path = species_path.replace('images', 'features')
        elif mode == 1:
            new_path = species_path.replace('images', 'curvature_maps')
        else:
            new_path = species_path.replace('images', 'mlp_features')
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        for i, image_path in enumerate(sorted(glob.glob(species_path + '/*'))):
            if step == 1 or (step > 1 and (i - base) % step == 0):
                if mode == 0:
                    f_path = new_path + '/' + image_path.split('/')[-1].split('.')[0]
                    features = get_features(image_path, show)
                    if features is not None:
                        np.save(f_path, features)
                elif mode == 1:
                    get_curvature_maps(image_path)
                else:
                    curve_map_for_mlp(image_path)
                count += 1
                print('Done:', str(count).rjust(6), end="\r")
            if show or (limit > 0 and i >= limit):
                break
    print('Done:', str(count).rjust(6))
        

def main(argv):
    if len(argv) == 1:
        extract(argv[0], mode=2)
    else:
        extract(argv[0], limit=int(argv[1]), step=int(argv[2]), base=int(argv[3]), mode=0)


if __name__ == "__main__":
    main(sys.argv[1:])
