#! /usr/bin/env python

import glob
import os
import sys
from math import floor, sqrt
import functools
import cv2
import numpy as np
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')


def circle_kernel(rad):
    k = np.zeros((rad * 2 + 1, rad * 2 + 1), np.uint8)
    cv2.circle(k, (rad, rad), rad, (1, 1, 1), -1)
    return k


def square_kernel(rad):
    return np.ones((rad, rad), np.uint8)


def get_curvature_map(line, segment, scale=0.1, length=128, figure=False, plot=False):
    curv = []
    r = max(int(len(line) * scale), 2)
    a = np.pi * pow(r, 2)
    for i in range(0, len(line), max(floor(len(line) / length), 1)):
        mask = np.zeros(segment.shape[:2], np.uint8)
        c = line[i][0]
        cv2.circle(mask, (int(c[0]), int(c[1])), r, (255, 255, 255), -1)
        res = cv2.bitwise_and(mask, segment)
        o = np.sum(res) / a
        curv.append(o)
        if figure:
            show = np.zeros((segment.shape[0], segment.shape[1], 3))
            show[:, :, 0] += segment
            show[:, :, 1] += segment
            show[:, :, 2] += segment
            show[:, :, 2] += mask
            show[:, :, 2] -= res
            show[:, :, 2] -= res
            show[:, :, 0] -= res
            cv2.imshow('o', show)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    c = np.array(curv, np.uint8)
    if len(c) != length:
        c = cv2.resize(c, (1, length)).reshape(length)
    if plot:
        for i, a in enumerate(c):
            print('(' + str(i) + ',' +str(a) + ') ', end='')
        print('')
        plt.plot(c)
        plt.show()
    return c


def write_curvature_image(path, curve, segment):
    maps = []
    for h in range(10, 106, 6):
        c = get_curvature_map(curve, segment, scale=h / 300)
        maps.append(c)
    maps = np.array(maps, np.uint8).reshape((16, 128))
    cv2.imwrite(path, maps)

# FEATURES #


def entropy(seq, w=1):
    seq = seq / np.sum(seq)
    entropy = 0
    for q in seq:
        entropy -= q * np.nan_to_num(np.log2(q))
    entropy += np.log(w)
    return entropy


def f_curvature_stat(curvature_map):
    out = []

    # MEAN
    out.append(np.mean(curvature_map))
    # STD
    out.append(np.std(curvature_map))

    # FIRST DIFF
    # mean of basic and absolute
    fd = np.diff(curvature_map, n=1)
    out.append(np.mean(fd))
    out.append(np.std(fd))
    out.append(np.mean(np.abs(fd)))
    out.append(np.std(np.abs(fd)))

    # SECOND DIFF
    # mean of basic and absolute
    sd = np.diff(curvature_map, n=2)
    out.append(np.mean(sd))
    out.append(np.std(sd))
    out.append(np.mean(np.abs(sd)))
    out.append(np.std(np.abs(sd)))

    # BENDING ENERGY
    s = 0
    for k in curvature_map:
        s += pow(k, 2)
    s /= len(curvature_map)
    out.append(s)

    # ENTROPY
    b = 128
    c_hist = np.histogram(curvature_map, bins=b, range=(0, 1))[0]
    e = entropy(c_hist, 1 / b)
    out.append(e / 12)

    # CURVE AREA
    ac = np.abs(curvature_map - np.mean(curvature_map))
    out.append(np.mean(ac))

    return out


def f_basic_shape(cnt, a, l):
    out = []
    hull = cv2.convexHull(cnt)
    rect = cv2.minAreaRect(cnt)
    w, h = rect[1]
    # SOLIDITY
    ha = cv2.contourArea(hull)
    solidity = 1
    if ha != 0:
        solidity = a / ha
    out.append(solidity)
    # CONVEXITY
    # convexity = cv2.arcLength(hull, False) / l
    # out.append(convexity)
    # CIRCULARITY
    circularity = (4 * np.pi * a) / (l**2)
    out.append(circularity)
    # RECTANGULARITY
    rectangularity = a / (w * h)
    out.append(rectangularity)
    # COMPACTNESS
    compactness = l / a
    out.append(compactness)
    return out


def f_fft(cnt):
    # REAL FFT
    x = np.abs(np.fft.rfft(cnt, len(cnt)))
    # SPECTRAL CENTROID
    f = np.fft.rfftfreq(len(cnt))
    c = 0
    for i in range(len(x)):
        c += f[i] * x[i]
    c /= np.sum(x)
    # SPECTRAL ENTROPY
    x /= x.max()
    out = x.tolist()
    # out.append(np.mean(x))
    # out.append(np.std(x))
    out.append(c)
    return out

# EXTRACTION #


def remove_stem(image, r=7):
    # tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, square_kernel(r))
    # _, contours, _ = cv2.findContours(tophat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # c = None
    # cl = 0
    # for cnt in contours:
    #     l = cv2.arcLength(cnt, False)
    #     if l > cl:
    #         cl = l
    #         c = cnt
    # if c is not None:
    #     sub = cv2.drawContours(np.zeros(image.shape[:2], np.uint8), [c], -1, (255, 255, 255), thickness=-1)
    #     # cv2.imshow('a', sub)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     image = cv2.subtract(image, sub)
    image = cv2.subtract(image, cv2.morphologyEx(image, cv2.MORPH_TOPHAT, square_kernel(r)))
    return image


def get_largest_contour(contours, h, w):
    curr = (None, 0, 0)
    for cnt in contours:
        a = cv2.contourArea(cnt)
        l = cv2.arcLength(cnt, False)
        if l < 30:
            continue
        M = cv2.moments(cnt)
        try:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        except Exception:
            continue
        dc = sqrt(pow(abs(w / 2 - cx), 2) + pow(abs(h / 2 - cy), 2))
        if dc < sqrt(h**2 + w**2) / 4 and a > curr[1]:
            curr = (cnt, a, l)
    if curr[0] is None:
        return None, None, None
    return curr


def isolate_folio_leaf(path):
    image = cv2.imread(path)
    h, w = image.shape[:2]
    s = 1000 / max(h, w)
    image = cv2.resize(image, (int(s * w), int(s * h)))
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gt = max(np.min([np.mean(grey[:, 0]), np.mean(grey[:, -1]), np.mean(grey[0, :]), np.mean(grey[-1, :])]), 90)
    _, thresh = cv2.threshold(grey, gt * 0.6, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, circle_kernel(9))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, sat, _ = cv2.split(hsv)
    st = np.max([np.mean(sat[:, 0]), np.mean(sat[:, -1]), np.mean(sat[0, :]), np.mean(sat[-1, :])])
    _, sat = cv2.threshold(sat, st + 35, 255, cv2.THRESH_BINARY)
    sat = cv2.morphologyEx(sat, cv2.MORPH_CLOSE, circle_kernel(7))
    # if saturation is valid then add it to the threshold
    if np.mean(sat) < 200:
        thresh = np.add(sat, thresh)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    curr = (None, 0, 0)
    for cnt in contours:
        a = cv2.contourArea(cnt)
        l = cv2.arcLength(cnt, False)
        if l < 30:
            continue
        if a > curr[1]:
            curr = (cnt, a, l)
    if curr[0] is None:
        # print('SKIPPED', path)
        return None, None, None, None, None
    return grey, curr[0], curr[1], curr[2], thresh


def isolate_leafsnap_leaf(t, path):
    image = cv2.imread(path)
    h, w = image.shape[:2]
    if t == 'field':
        # resize field images to approx match lab
        if h > w:
            image= cv2.resize(image, (375, 500))
        else:
            image= cv2.resize(image, (500, 375))
        h, w = image.shape[:2]
    # threshold and contour
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gt = max(np.min([np.mean(grey[:, 0]), np.mean(grey[:, -1]), np.mean(grey[0, :]), np.mean(grey[-1, :])]), 90)
    _, thresh_raw = cv2.threshold(grey, gt * 0.6, 255, cv2.THRESH_BINARY_INV)
    thresh_raw = cv2.morphologyEx(thresh_raw, cv2.MORPH_CLOSE, circle_kernel(5))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, sat, _ = cv2.split(hsv)
    st = np.max([np.mean(sat[:, 0]), np.mean(sat[:, -1]), np.mean(sat[0, :]), np.mean(sat[-1, :])])
    _, sat = cv2.threshold(sat, st + 45, 255, cv2.THRESH_BINARY)  # 115
    sat = cv2.morphologyEx(sat, cv2.MORPH_CLOSE, circle_kernel(5))
    # if saturation is valid then add it to the threshold
    if np.mean(sat) < 200:
        thresh_raw = np.add(sat, thresh_raw)
    # top hat it to remove stem, unless the leaf is very small
    a = np.sum(thresh_raw)
    thresh_th = remove_stem(thresh_raw, 13 if t == 'field' else 7)
    b = np.sum(thresh_th)
    if b > 0.9 * a:
        thresh = thresh_th
    else:
        thresh = thresh_raw
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    curr = get_largest_contour(contours, h, w)
    if curr[0] is None:
        thresh = cv2.dilate(cv2.Canny(image, 100, 200, 15), circle_kernel(1), iterations=1)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        curr = get_largest_contour(contours, h, w)
    if curr[0] is None:
        return None, None, None, None, None
    return grey, curr[0], curr[1], curr[2], thresh


def isolate_foliage_leaf(path):
    image = cv2.imread(path)
    h, w = image.shape[:2]
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, 220, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    curr = (None, 0, 0)
    for cnt in contours:
        a = cv2.contourArea(cnt)
        l = cv2.arcLength(cnt, False)
        if l < 30:
            continue
        if a > curr[1]:
            curr = (cnt, a, l)
    if curr[0] is None:
        return None, None, None, None, None
    return grey, curr[0], curr[1], curr[2], thresh


def isolate_swedish_leaf(path):
    image = cv2.imread(path)
    h, w = image.shape[:2]
    s = 1000 / max(h, w)
    image = cv2.resize(image, (int(s * w), int(s * h)))
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_raw = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.subtract(thresh_raw, cv2.morphologyEx(thresh_raw, cv2.MORPH_TOPHAT, square_kernel(15)))
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    curr = (None, 0, 0)
    for cnt in contours:
        a = cv2.contourArea(cnt)
        l = cv2.arcLength(cnt, False)
        if l < 30:
            continue
        if a > curr[1]:
            curr = (cnt, a, l)
    if curr[0] is None:
        return None, None, None, None, None
    return grey, curr[0], curr[1], curr[2], thresh_raw


def isolate_flavia_leaf(path):
    image = cv2.imread(path)
    h, w = image.shape[:2]
    s = 1000 / max(h, w)
    image = cv2.resize(image, (int(s * w), int(s * h)))
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    curr = (None, 0, 0)
    for cnt in contours:
        a = cv2.contourArea(cnt)
        l = cv2.arcLength(cnt, False)
        if l < 30:
            continue
        if a > curr[1]:
            curr = (cnt, a, l)
    if curr[0] is None:
        return None, None, None, None, None
    return grey, curr[0], curr[1], curr[2], thresh


def isolate_leaves_leaf(path):
    grey = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(grey, 240, 255, cv2.THRESH_BINARY)
    if 'fish' not in path and 'mpeg' not in path and 'animals' not in path:
        thresh = cv2.subtract(thresh, cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, square_kernel(11)))
    else:
        h, w = grey.shape[:2]
        s = 640 / max(h, w)
        grey = cv2.resize(grey, (int(s * w), int(s * h)))
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    curr = (None, 0, 0)
    for cnt in contours:
        a = cv2.contourArea(cnt)
        l = cv2.arcLength(cnt, False)
        if l < 30:
            continue
        if a > curr[1]:
            curr = (cnt, a, l)
    if curr[0] is None:
        return None, None, None, None, None
    segment = cv2.imread(path, 0)
    return grey, curr[0], curr[1], curr[2], segment


def isolate_shapecn_leaf(path):
    grey = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(grey, 240, 255, cv2.THRESH_BINARY)
    if thresh.shape[0] != 256:
        thresh = cv2.dilate(thresh, circle_kernel(2), iterations=1)
        thresh = cv2.resize(thresh, (256, 256), interpolation=cv2.INTER_NEAREST)
    else:
        thresh = cv2.dilate(thresh, circle_kernel(1), iterations=1)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    curr = (None, 0, 0)
    for cnt in contours:
        a = cv2.contourArea(cnt)
        l = cv2.arcLength(cnt, False)
        if l < 5:
            continue
        if a > curr[1]:
            curr = (cnt, a, l)
    if curr[0] is None:
        return None, None, None, None, None
    segment = cv2.drawContours(np.zeros(thresh.shape[:2], np.uint8), [curr[0]], -1, (255, 255, 255), thickness=-1)
    return grey, curr[0], curr[1], curr[2], segment


def get_features(dataset, path, use_cmap=False):
    if dataset == 'foliage':
        isolate_leaf = isolate_foliage_leaf
    elif dataset in ['leaves', 'leafsnap-s', 'fish', 'mpeg', 'animals']:
        isolate_leaf = isolate_leaves_leaf
    elif 'shapecn' in dataset:
        isolate_leaf= isolate_shapecn_leaf
    elif dataset == 'swedish':
        isolate_leaf = isolate_swedish_leaf
    elif dataset == 'leafsnap-l' or dataset == 'figure':
        isolate_leaf = functools.partial(isolate_leafsnap_leaf, 'lab')
    elif dataset == 'leafsnap-f':
        isolate_leaf = functools.partial(isolate_leafsnap_leaf, 'field')
    elif dataset == 'flavia':
        isolate_leaf = isolate_flavia_leaf
    elif dataset == 'folio':
        isolate_leaf = isolate_folio_leaf
    else:
        raise Exception('invalid dataset')
    grey, contour, area, length, segmentation = isolate_leaf(path)
    if contour is None or segmentation is None:
        return None
    f = []
    f.extend(f_basic_shape(contour, area, length))
    cmap_path = '_'.join(path.replace('images', 'cmaps').split('.')[:-1])
    if use_cmap:
        curvatures = np.load(cmap_path + '.npy')
        for c in curvatures:
            f.extend(f_fft(c))
            f.extend(f_curvature_stat(c / 255))
    else:
        curvatures = []
        scales = [0.01, 0.025, 0.05, 0.1, 0.15]
        for h in scales:
            c = get_curvature_map(contour, segmentation, scale=h, length=256)
            curvatures.append(c)
            f.extend(f_fft(c))
            f.extend(f_curvature_stat(c / 255))
        cmap_folder = '/'.join(cmap_path.split('/')[:-1])
        if not os.path.exists(cmap_folder):
            os.makedirs(cmap_folder)
        np.save(cmap_path, curvatures)
    f = np.nan_to_num(f)  # to be safe
    return f


def extract(dataset, t, limit=-1, step=1, base=0, show=False, use_cmap=False):
    count = 0
    skipped = 0
    for species_path in sorted(glob.glob(dataset + '/images/' + t + '/*')):
        new_path = species_path.replace('images', 'features')
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        for i, image_path in enumerate(sorted(glob.glob(species_path + '/*'))):
            if step == 1 or (step > 1 and (i - base) % step == 0):
                f_path = new_path + '/' + '_'.join(image_path.split('/')[-1].split('.')[:-1])
                features = get_features(dataset, image_path, use_cmap)
                if features is not None:
                    np.save(f_path, features)
                else:
                    skipped += 1
                count += 1
                print('Done ' + t + ':', str(count).rjust(6), '(' + str(skipped) + ')', end='\r')
            if (show and i > 3) or (limit > 0 and i >= limit - 1):
                break
    print('Done ' + t + ':', str(count).rjust(6), '(' + str(skipped) + ')')


def help():
    print('Basic usage:')
    print('extract.py dataset_name subset_name [-cm]')
    print()
    print('dataset_name must be one of foliage, leaves, shapecn, swedish, leafsnap-l, leafsnap-f, flavia, folio')
    print('datasets should be located in the same directoy in folder named as above')
    print('inside each folder should be a folder called images, inside this a number of folders, one for each subset')
    print('inside these there should be a folder for each class which contains the appropriate images')
    print('subsets are usually "train" and "test" for example, the leaves folder has an example layout')
    print('curvature maps are cached when features are extracted, cached versions can be used with the -cm option')
    print()
    print('Advanced usage:')
    print('extract.py dataset_name subset_name [-cm] [-l k] [-s n -b m]')
    print('-l k limits the number of images per class for which features are extracted to n')
    print('For splitting the task across multiple processes -s and -b can be used')
    print('-s n defines the step i.e. the number of processes being used as n')
    print('-b m defines the base for this process as m i.e. which of the n processes it is')
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '-h' or sys.argv[1] == '--help':
            help()
        else:
            lim = -1
            step = 1
            base = 0
            if '-l' in sys.argv:
                lim = int(sys.argv[sys.argv.index('-l') + 1])
            if '-s' in sys.argv or '-b' in sys.argv:
                if not ('-s' in sys.argv and '-b' in sys.argv):
                    raise Exception('both step and base required')
                step = int(sys.argv[sys.argv.index('-s') + 1])
                base = int(sys.argv[sys.argv.index('-b') + 1])
            extract(sys.argv[1],
                    sys.argv[2],
                    use_cmap=('-cm' in sys.argv),
                    limit=lim,
                    step=step,
                    base=base)
    else:
        help()
