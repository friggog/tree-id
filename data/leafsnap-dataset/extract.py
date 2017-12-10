import glob
import os
from math import ceil, floor, sqrt

import cv2
import numpy as np
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
import sys

np.seterr(divide='ignore', invalid='ignore')

def normalise_contour(contour):
    # resize the contour so the long side is 300 long
    # position doesnt matter as curvature is then based on differences
    # rotation invariance is built into curvature
    rect = cv2.minAreaRect(contour)
    h, w = rect[1]
    s = 300 / max(h, w)
    contour = contour * s
    return contour


def get_curvature_map(line, segment, scale=10, method=0):
    curv = []
    # if method == 0:
    l = 64
    r = scale
    ha = np.pi * pow(r, 2) / 2
    for i in range(0, len(line), max(floor(len(line) / l), 1)):
        mask = np.zeros(segment.shape[:2], np.uint8)
        c = line[i][0]
        cv2.circle(mask, (int(c[0]), int(c[1])), r, (255, 255, 255), -1)
        res = cv2.bitwise_and(mask, segment)
        o = (np.sum(res / 255) - ha) / ha
        curv.append(o)
    if len(curv) < l:
        for i in range(l - len(curv)):
            curv.append(0) # TODO
    else:
        curv = curv[:l]
    # plt.plot(curv)
    # plt.show()
    # curv = []
    # # else:
    # h = int(len(line)/100)
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
    # plt.plot(curv)
    # plt.show()
    return curv

# FEATURES #

def f_glcm(image):
    gl = np.zeros((256,256), np.uint8)
    h,w = image.shape[:2]
    for i in range(h-1):
        for j in range(w-1):
            if image[i,j] != 0 and image[i,j+1] != 0:
                gl[image[i,j],image[i,j+1]] += 1
                gl[image[i,j],image[i+1,j]] += 1
    gl = gl / gl.sum() / 2
    energy = 0
    contrast = 0
    homogenity = 0
    IDM = 0
    entropy = 0
    for i in range(255):
        for j in range(255):
            energy += pow(gl[i,j], 2)
            contrast += (i-j)*(i-j)*gl[i,j]
            homogenity += gl[i,j]/(1+abs(i-j))
            if i !=j:
                IDM += gl[i,j]/pow(i-j, 2)
            if gl[i,j] != 0:
                entropy -= gl[i,j]*np.log10(gl[i,j])
            mean =+ 0.5*(i*gl[i,j] + j*gl[i,j])
    # print( 100*energy, contrast/200, homogenity, IDM, entropy/5)
    return 100*energy, contrast/200, homogenity, IDM, entropy/5

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


def f_curvature_hist(curvature_map):
    n_bins = 16
    hist = np.histogram(curvature_map, bins=n_bins, density=True)
    return hist[0]


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
    # thresh = cv2.subtract(thresh, cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, np.ones((3,3), np.uint8)))
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    curr = (None, 0, 0)
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a < (h * w) * 0.01:
            continue
        l = cv2.arcLength(cnt, False)
        if l < max(w,h) * 0.05:
            continue
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        dc = sqrt(pow(abs(w/2 - cx), 2) + pow(abs(h/2 - cy), 2))
        if dc < (h * w / 750):
            curr = (cnt, a, l)
    if curr[0] is None:
        return None, None, None, None, None
    segment = np.zeros((h,w), np.uint8)
    cv2.drawContours(image, [curr[0]], -1, color=(255,0,255))
    cv2.drawContours(segment, [curr[0]], -1, color=(255,255,255), thickness=-1)
    return grey, curr[0], curr[1], curr[2], segment

def get_features(path, show=False):
    image = cv2.imread(path)
    h,w = image.shape[:2]
    size = 256
    scale = size/max(w,h)
    image = cv2.resize(image, (int(w*scale), int(h*scale)))
    grey, contour, area, length, segmentation = isolate_leaf(image)
    if contour is None or segmentation is None:
        return []
    # contour = contour.reshape((-1,2))
    # x1, y1 = np.min(contour, axis=0)
    # x2, y2 = np.max(contour, axis=0)
    # o = 5
    # ymin = max(y1-o,0)
    # ymax = min(y2+o,h)
    # xmin = max(x1-o,0)
    # xmax = min(x2+o,w)
    # image = image[ymin:ymax,xmin:xmax]
    # grey = grey[ymin:ymax,xmin:xmax]
    # segmentation = segmentation[ymin:ymax,xmin:xmax]
    grey = cv2.bitwise_and(grey, grey, mask=segmentation)
    # contour[:,0] -= xmin
    # contour[:,1] -= ymin
    # contour = contour.reshape((-1, 1, 2))
    if show:
        # [vx,vy,x,y] = cv2.fitLine(curr[0], cv2.DIST_L2,0,0.01,0.01)
        # lefty = int((-x*vy/vx) + y)
        # righty = int(((w-x)*vy/vx)+y)
        # cv2.line(image,(w-1,righty),(0,lefty),(0,255,0),2)
        # cv2.drawContours(image, contours, -1, (0, 0, 255))
        # cv2.drawContours(image, [contour], -1, (255, 255, 0))
        cv2.imshow(path, segmentation)
        cv2.imshow(path+'i', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # curvature_map = get_curvature_map(contour, segmentation)
    f = []
    f.extend(f_basic_shape(contour, area, length))
    # f.extend(f_glcm(grey))
    for h in range(10,50,5):
        c = get_curvature_map(contour, segmentation, scale=h)
        f.extend(f_curvature_stat(c))
        # f.extend(f_curvature_hist(c))
    return f


def extract(env, limit=-1, step=1, base=0, show=False,):
    print('EXTRACTING')
    images = {}
    count = 0
    for species_path in sorted(glob.glob('dataset/images/' + env + '/*')):
        species = species_path.split('/')[-1]
        new_path = 'dataset/features/' + env + '/' + species
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        for i, image_path in enumerate(sorted(glob.glob(species_path + '/*'))):
            if step == 1 or (step > 1 and (i - base) % step == 0):
                f_path = new_path + '/' + image_path.split('/')[-1].split('.')[0]
                features = get_features(image_path, show)
                np.save(f_path, features)
                count += 1
                print('Done:', count, end="\r")
                if show or (limit > 0 and i >= limit):
                    break

def main(argv):
    if len(argv)==1:
        extract(argv[0])
    else:
        extract(argv[0], int(argv[1]), int(argv[2]), int(argv[3]))

if __name__ == "__main__":
    main(sys.argv[1:])
