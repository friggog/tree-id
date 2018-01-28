import glob
from math import sqrt, floor

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

MAP_LENGTH = 128
MAP_SCALE = 20

def get_curvature_map(line, segment):
    curv = []
    l = MAP_LENGTH
    r = MAP_SCALE
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
    return curv


def isolate_leaf(image):
    h, w = image.shape[:2]
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, np.median(grey) * 0.5, 255, cv2.THRESH_BINARY_INV)
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


def get_cm_from_path(path):
    image = cv2.imread(path)
    h, w = image.shape[:2]
    grey, contour, area, length, segmentation = isolate_leaf(image)
    if contour is None or segmentation is None:
        return []
    # cv2.imshow('2',segmentation)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return get_curvature_map(contour, segmentation)

def get_cm_offset(average_map, new_map):
    max_c = (0, 0, 0)
    for i in range(len(average_map)):
        r = np.roll(new_map, i)
        c = np.correlate(average_map, r)
        mse = ((average_map - r) ** 2).mean()
        if c > max_c[0] and mse < 0.1:
            max_c = (c, i, mse)
    # plt.plot(average_map)
    # plt.plot(np.roll(new_map, max_c[1]))
    # plt.show()
    # print(max_c[0], max_c[2])
    if max_c[2] != 0:
        return np.roll(new_map, max_c[1])
    else:
        return None


# def get_ave_maps(env, limit=-1):
#     count = 0
#     maps = []
#     labels = []
#     for species_path in sorted(glob.glob('dataset/images/' + env + '/*')):
#         species = species_path.split('/')[-1]
#         average_map = None
#         for i, image_path in enumerate(sorted(glob.glob(species_path + '/*'))):
#             if i < 1:
#                 continue # TEMP for testing
#             cm = get_cm_from_path(image_path)
#             if len(cm) != 0:
#                 if average_map is None:
#                     average_map = cm
#                 else:
#                     nm = get_cm_offset(average_map, cm)
#                     if nm is not None:
#                         average_map = (average_map + nm)/2
#             count += 1
#             print('Done:', count, end="\r")
#             if limit > 0 and i >= limit - 1:
#                 break
#         maps.append(average_map)
#         labels.append(species)
#     return maps, labels
#
# def predict_from_cm(curvature_map, maps, labels):
#     max_c = (0, 0, 0, 0)
#     for i, ex_map in enumerate(maps):
#         if ex_map is None:
#             continue
#         for j in range(len(ex_map)):
#             r = np.roll(curvature_map, j)
#             c = np.correlate(ex_map, r)
#             mse = ((ex_map - r) ** 2).mean()
#             if c > max_c[0] and mse < 0.1:
#                 max_c = (c, i, j, mse)
#     return labels[max_c[1]]
#
#
# c_maps, c_labels = get_ave_maps('field', 3)
# correct = 0
# total = 0
# for species_path in sorted(glob.glob('dataset/images/' + 'field' + '/*')):
#     species = species_path.split('/')[-1]
#     for i, image_path in enumerate(sorted(glob.glob(species_path + '/*'))):
#         if i < 1:
#             cm = get_cm_from_path(image_path)
#             predicted = predict_from_cm(cm, c_maps, c_labels)
#             correct += predicted == species
#             total += 1
#             print(species, predicted)
# print(correct/total)



# def get_shifted_maps(env, limit=-1):
#     count = 0
#     maps = []
#     labels = []
#     for species_path in sorted(glob.glob('dataset/images/' + env + '/*')):
#         species = species_path.split('/')[-1]
#         average_map = None
#         for i, image_path in enumerate(sorted(glob.glob(species_path + '/*'))):
#             cm = get_cm_from_path(image_path)
#             if len(cm) != 0:
#                 if average_map is None:
#                     average_map = cm
#                     maps.append(cm)
#                     labels.append(species)
#                 else:
#                     nm = get_cm_offset(average_map, cm)
#                     if nm is not None:
#                         maps.append(nm)
#                         labels.append(species)
#                         average_map = (average_map + nm)/2
#                     else:
#                         print('UHOH')
#             count += 1
#             print('Done:', count, end="\r")
#             if limit > 0 and i >= limit - 1:
#                 break
#     return maps, labels
#
# maps, labels = get_shifted_maps('field', limit=15)
# clf = KNeighborsClassifier()
# scores = cross_validate(clf, maps, labels, cv=5, scoring=['precision_macro', 'recall_macro', 'f1_macro'], return_train_score=False)
# print('Precision:', np.mean(scores['test_precision_macro']), 'Recall', np.mean(scores['test_recall_macro']), 'F1', np.mean(scores['test_f1_macro']))
