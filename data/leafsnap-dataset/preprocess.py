import glob
import os
from math import ceil, floor, sqrt

import cv2
import numpy as np

MAX_P = 0.95
MIN_P = 0.7
B_Y_THRESH = 0.3
B_X_THRESH = 0.1
W_THRESH = 0.5
K_SIZE = 10
OUT_SIZE = 512

def cut_edges(env, show=False):
    for species_path in sorted(glob.glob('dataset/images/'+env+'/*')):
        if not os.path.exists(species_path):
            os.makedirs(species_path)
        for image_path in sorted(glob.glob(species_path + '/*')):
            print(image_path)
            new_path = 'dataset/images/'+env+'_p/' + '/'.join(image_path.split('/')[3:])
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


def resize(env, show=False):
    count = 0
    for species_path in sorted(glob.glob('dataset/images/'+env+'/*')):
        new_path = 'dataset/images/'+env+'_r/' + '/'.join(species_path.split('/')[3:])
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        for image_path in sorted(glob.glob(species_path + '/*')):
            count += 1
            print('Processed: ', count, end="\r")
            new_img_path = new_path + '/' + image_path.split('/')[-1]
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            sqs = max(h, w)
            hd = (sqs - h) / 2
            wd = (sqs - w) / 2
            if wd != 0:
                a = np.mean(image[:,0],axis=0)
                b = np.mean(image[:,w-1],axis=0)
            else:
                a = np.mean(image[0,:],axis=0)
                b = np.mean(image[h-1,:],axis=0)
            edge_colour = (a + b) / 2
            squared = cv2.copyMakeBorder(image, top=floor(hd), bottom=ceil(hd), left=floor(wd),
                right=ceil(wd), borderType=cv2.BORDER_CONSTANT, value=edge_colour)
            resized = cv2.resize(squared, (OUT_SIZE, OUT_SIZE))
            if show:
                cv2.destroyAllWindows()
                cv2.imshow(image_path, resized)
                cv2.waitKey(0)
                break
            else:
                cv2.imwrite(new_img_path, resized)

def isolate_leaf(path):
    image = cv2.imread(path)
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
        return None, None
    segment = np.zeros((h,w), np.uint8)
    cv2.drawContours(segment, [curr[0]], -1, color=(255,255,255), thickness=-1)
    return segment, curr[0]

def get_curvature_map(line, segment, scale=25):
    curv = []
    r = scale
    ha = np.pi * pow(r,2) / 2
    for i in range(0, len(line), max(int(len(line)/100),1)):
        mask = np.zeros(segment.shape[:2], np.uint8)
        c = line[i][0]
        cv2.circle(mask, (int(c[0]),int(c[1])), r, (255,255,255), -1)
        res = cv2.bitwise_and(mask, segment)
        o = (np.sum(res/255) - ha) / ha
        curv.append(o)
    return curv

def get_curvature(env, show=False):
    count = 0
    for species_path in sorted(glob.glob('dataset/images/'+env+'/*')):
        new_path = 'dataset/curvatures/'+env+'/'+'/'.join(species_path.split('/')[3:])
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        for image_path in sorted(glob.glob(species_path + '/*')):
            count += 1
            print('Processed: ', count, end="\r")
            new_img_path = new_path + '/' + image_path.split('/')[-1]
            segmentation, contour = isolate_leaf(image_path)
            if segmentation is None or contour is None:
                continue
            maps = []
            for h in range(1,150,25):
                maps.append(get_curvature_map(contour, segmentation, h))
            out = (np.array(maps)+1)/2
            if show:
                cv2.destroyAllWindows()
                cv2.imshow(image_path, out)
                cv2.waitKey(0)
                break
            cv2.imwrite(new_img_path, out)

def segment(env, show=False):
    count = 0
    for species_path in sorted(glob.glob('dataset/images/'+env+'/*')):
        new_path = 'dataset/segmentations/'+env+'/'+'/'.join(species_path.split('/')[3:])
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        for image_path in sorted(glob.glob(species_path + '/*')):
            count += 1
            print('Processed: ', count, end="\r")
            new_img_path = new_path + '/' + image_path.split('/')[-1]
            segmentation, _ = isolate_leaf(image_path)
            if segmentation is None or np.mean(segmentation) > 130:
                continue
            if show:
                cv2.destroyAllWindows()
                cv2.imshow(image_path, segmentation)
                cv2.waitKey(0)
                break
            cv2.imwrite(new_img_path, segmentation)
