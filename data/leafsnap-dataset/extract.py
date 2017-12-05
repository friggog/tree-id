import glob
import os
from math import ceil, floor, sqrt

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib

import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')

def calculate_curvature(line):
    # TODO tune h
    out = []
    for h in range(10, 16, 2): # 10, 16, 2
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
        bins = 64 # 48
        hist = np.histogram(curv, bins=bins, range=[-64, 64], density=True)
        hist = np.nan_to_num(hist[0])
        out.extend(hist)
    return out


def get_features(path, show=False, env='lab'):
    image = cv2.imread(path)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, np.median(grey) * 0.6, 255, cv2.THRESH_BINARY_INV)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, sat, vib = cv2.split(hsv)
    _, sat = cv2.threshold(sat, 0.6*(np.min(sat)+np.max(sat))/2, 255, cv2.THRESH_BINARY)
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

def load_features(env, limit=-1, start=0):
    featureset = []
    labelset = []
    for species_path in sorted(glob.glob('dataset/features/' + env + '/*')):
        species = species_path.split('/')[-1]
        for i, f_path in enumerate(sorted(glob.glob(species_path + '/*'))):
            if i < start:
                continue
            features = np.load(f_path)
            if features is not None and len(features) != 0:
                featureset.append(features)
                labelset.append(species)
            if (limit > 0 and i > limit):
                break
    return featureset, labelset


def classify(mode=0):
    print('CLASSIFYING')
    # X_train, X_test, y_train, y_test = train_test_split(featureset, labelset, test_size=0.2, random_state=0)
    #
    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
    #                      'C': [1, 10, 100, 1000]},
    #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    #
    # scores = ['precision', 'recall']
    #
    # for score in scores:
    #     print("# Tuning hyper-parameters for %s" % score)
    #     print()
    #
    #     clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score)
    #     clf.fit(X_train, y_train)
    #
    #     print("Best parameters set found on development set:")
    #     print()
    #     print(clf.best_params_)
    #     print()
    #     print("Grid scores on development set:")
    #     print()
    #     means = clf.cv_results_['mean_test_score']
    #     stds = clf.cv_results_['std_test_score']
    #     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #         print("%0.3f (+/-%0.03f) for %r"
    #               % (mean, std * 2, params))
    #     print()
    #
    #     print("Detailed classification report:")
    #     print()
    #     print("The model is trained on the full development set.")
    #     print("The scores are computed on the full evaluation set.")
    #     print()
    #     y_true, y_pred = y_test, clf.predict(X_test)
    #     print(classification_report(y_true, y_pred))
    #     print()

    if mode == 0:
        featureset, labelset = load_features('lab_r')
        clf = SVC(kernel='linear', C=1000, class_weight='balanced')
        k = 10
        fold_a = []
        for i in range(k):
            train_f = [item for index, item in enumerate(featureset) if index - i % k != 0]
            train_l = [item for index, item in enumerate(labelset) if index - i % k != 0]
            clf.fit(train_f, train_l)
            test_f = featureset[i::10]
            test_l = labelset[i::10]
            predicted = clf.predict(test_f)
            correct = predicted == test_l
            acc = np.mean(correct)
            print('Fold', i + 1, 'complete with recall:', acc)
            fold_a.append(acc)
        # TO SAVE
        print('Completed with average recall: ', np.mean(fold_a))
    elif mode == 1:
        clf = SVC(kernel='linear', C=1000, class_weight='balanced')
        train_fs, train_ls = load_features('lab_r')
        train_fs2, train_ls2 = load_features('field_r', start=3)
        train_fs.extend(train_fs2)
        train_ls.extend(train_ls2)
        clf.fit(train_fs, train_ls)
        joblib.dump(clf, 'SVM.pkl')
        test_fs, test_ls = load_features('field_r', limit=3, start=0)
        predicted = clf.predict(test_fs)
        correct = predicted == test_ls
        acc = np.mean(correct)
        print('Completed recall: ', acc)

extract('field_r', limit=15)
# classify(mode=1)
