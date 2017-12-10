import glob
from math import sqrt, floor

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def get_cm(line, segment, scale=25, method=0):
    curv = []
    if method == 0:
        l = 75
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
                curv.append(0)
        else:
            curv = curv[:75]
    else:
        for i in range(len(line)):
            if i - h < - len(line) + 1:
                m = line[i - h + len(line)][0]
            else:
                m = line[i - h][0]
            c = line[i][0]
            if i > len(line) - h - 1:
                p = line[i + h - len(line)][0]
            else:
                p = line[i + h][0]
            f = (p - c) / pow(h, 2)
            ff = (p - 2 * c + m) / pow(h, 2)
            k = (f[0] * ff[1] - f[1] * ff[0]) / pow(pow(f[0], 2) + pow(f[1], 2), 1.5)
            curv.append(k)
        curv = np.nan_to_num(curv)
        curv = np.clip(curv, -200, 200)
    return curv


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
        return None, None, None, None
    return grey, curr[0], curr[1], curr[2]


def get_cm_from_path(path):
    image = cv2.imread(path)
    h, w = image.shape[:2]
    grey, contour, area, length = isolate_leaf(image)
    s_path = 'dataset/segmentations/' + '/'.join(path.split('/')[2:])
    segmentation = cv2.imread(s_path, 0)
    if contour is None or segmentation is None:
        return []
    return get_cm(contour, segmentation)


def do(env, limit=-1):
    count = 0
    maps = []
    labels = []
    for species_path in sorted(glob.glob('dataset/images/' + env + '/*')):
        species = species_path.split('/')[-1]
        for i, image_path in enumerate(sorted(glob.glob(species_path + '/*'))):
            cm = get_cm_from_path(image_path)
            if len(cm) != 0:
                maps.append(cm)
                labels.append(species)
            count += 1
            print('Done:', count, end="\r")
            if limit > 0 and i >= limit - 1:
                break
    clf = KNeighborsClassifier()
    scores = cross_validate(clf, maps, labels, cv=5, scoring=['precision_macro', 'recall_macro', 'f1_macro'], return_train_score=False)
    print('Precision:', np.mean(scores['test_precision_macro']), 'Recall', np.mean(scores['test_recall_macro']), 'F1', np.mean(scores['test_f1_macro']))
    # k = 10
    # fold_a = []
    # for i in range(k):
    #     train_f = [item for index, item in enumerate(maps) if (index - i) % k != 0]
    #     train_l = [item for index, item in enumerate(labels) if (index - i) % k != 0]
    #     clf.fit(train_f, train_l)
    #     test_f = maps[i::10]
    #     test_l = labels[i::10]
    #     predicted = clf.predict(test_f)
    #     correct = predicted == test_l
    #     acc = np.mean(correct)
    #     print('Fold', i + 1, 'complete with recall:', acc)
    #     fold_a.append(acc)
    # print('Completed with average recall: ', np.mean(fold_a))

do('field', limit=15)
