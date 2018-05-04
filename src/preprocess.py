#! /usr/bin/env python

import glob
import os
import sys
from math import ceil, floor
from shutil import copyfile
import random
import cv2
import numpy as np

MAX_P = 0.95
MIN_P = 0.7
B_Y_THRESH = 0.3
B_X_THRESH = 0.1
W_THRESH = 0.5
K_SIZE = 10
OUT_SIZE = 512


def cut_edges(dataset, env, show=False):
    for species_path in sorted(glob.glob(dataset + '/images/' +env +'/*')):
        if not os.path.exists(species_path):
            os.makedirs(species_path)
        count = 0
        for image_path in sorted(glob.glob(species_path + '/*')):
            count += 1
            print('Processed: ', count, end="\r")
            new_path = dataset + '/images/' +env +'_p/' + '/'.join(image_path.split('/')[3:])
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
        print('Processed: ', count)


def print_count(dataset, t):
    print('\n', t.upper())
    print('-' * 45)
    count = 0
    min = 999
    max = 0
    for species_path in sorted(glob.glob(dataset + '/images/' + t + '/*')):
        species = species_path.split('/')[-1]
        n = len(glob.glob(species_path + '/*'))
        count += n
        if n < min:
            min = n
        if n > max:
            max = n
        print((species + ': ').ljust(40), str(n).rjust(4))
    print('-' * 45)
    print('Total: '.ljust(35), str(count).rjust(9))
    print('Min: '.ljust(35), str(min).rjust(9))
    print('Max: '.ljust(35), str(max).rjust(9))


def resize(dataset, env, limit=-1, step=1, base=0, show=False):
    count = 0
    for species_path in sorted(glob.glob(dataset + '/images/' +env +'/*')):
        new_path = dataset + '/images/' +env +'_r/' + '/'.join(species_path.split('/')[3:])
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        for i, image_path in enumerate(sorted(glob.glob(species_path + '/*'))):
            if step == 1 or (step > 1 and (i - base) % step == 0):
                count += 1
                print('Processed: ', count, end="\r")
                new_img_path = new_path + '/' + image_path.split('/')[-1]
                image = cv2.imread(image_path)
                h, w = image.shape[:2]
                sqs = max(h, w)
                hd = (sqs - h) / 2
                wd = (sqs - w) / 2
                if wd != 0:
                    a = np.mean(image[:, 0], axis=0)
                    b = np.mean(image[:, w -1], axis=0)
                else:
                    a = np.mean(image[0, :], axis=0)
                    b = np.mean(image[h -1, :], axis=0)
                edge_colour = (a + b) / 2
                squared = cv2.copyMakeBorder(image, top=floor(hd), bottom=ceil(hd), left=floor(wd),
                                             right=ceil(wd), borderType=cv2.BORDER_CONSTANT, value=edge_colour)
                resized = cv2.resize(squared, (OUT_SIZE, OUT_SIZE))
                if show:
                    cv2.destroyAllWindows()
                    cv2.imshow(image_path, resized)
                    cv2.waitKey(0)
                else:
                    cv2.imwrite(new_img_path, resized)
                if show or (limit > 0 and i >= limit):
                    break
    print('Processed: ', count)


def split(dataset, n, shuffle=False, m='images'):
    train_paths = []
    test_paths = []
    test_counts = {}
    if dataset == 'leafsnap':
        envs = ['field', 'lab']
    else:
        envs = ['all']
    for env in envs:
        print('\n', env)
        for species_path in sorted(glob.glob(dataset +'/' + m + '/' + env + '/*')):
            species = species_path.split('/')[-1]
            test_path = species_path.replace('/' + env + '/', '/test/')
            if not os.path.exists(test_path):
                os.makedirs(test_path)
            train_path = species_path.replace('/' + env + '/', '/train/')
            if not os.path.exists(train_path):
                os.makedirs(train_path)
            paths = sorted(glob.glob(species_path + '/*'))
            print(species.ljust(30), str(len(paths)).rjust(4))
            if species not in test_counts:
                test_counts[species] = 0
            if shuffle:
                random.shuffle(paths)
            for i, path in enumerate(paths):
                if test_counts[species] < n:
                    test_paths.append((path, path.replace('/' + env + '/', '/test/')))
                    test_counts[species] += 1
                else:
                    train_paths.append((path, path.replace('/' + env + '/', '/train/')))
    print('Train'.ljust(10), len(train_paths))
    print('Test'.ljust(10), len(test_paths))
    for pin, pout in train_paths:
        copyfile(pin, pout)
    for pin, pout in test_paths:
        copyfile(pin, pout)


def help():
    print('Usage:')
    print('./preprocess.py -split dataset n_test [-s] \n\t\
            Split a dataset contating one "all" subset into "train" and "test" with n_test per class for testing\n\t\
            -s also shuffles the data randomly')
    print('./preprocess.py -resize dataset subset [-d]\n\t\
            Resizes all images to be sqaure, -d to debug (i.e. show images)')
    print('./preprocess.py -cut dataset subset [-d]\n\t\
            Crops the rulers from the borders of the leafsnap lab dataset\n\t\
            -d to debug (i.e. show images)')
    print('./preprocess.py -count dataset subset\n\t\
            Prints class counts for the subset of the dataset')
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '-count':
            print_count(sys.argv[2], sys.arg[3])
        elif sys.argv[1] == '-cut':
            cut_edges(sys.argv[2], sys.argv[3], show='-d' in sys.argv)
        elif sys.argv[1] == '-resize':
            resize(sys.argv[2], sys.argv[3], show='-d' in sys.argv)
        elif sys.argv[1] == '-split':
            split(sys.argv[2], int(sys.argv[3]), '-s' in sys.argv)
        else:
            help()
    else:
        help()
