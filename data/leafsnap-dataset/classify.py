#! /usr/local/bin/python3

import glob
import subprocess
import time
import warnings

import numpy as np
from sklearn import decomposition
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from extract import extract
from preprocess import resize

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def load_features(env, limit=-1, start=0):
    featureset = []
    labelset = []
    for species_path in sorted(glob.glob('dataset/features/' + env + '/*')):
        species = species_path.split('/')[-1]
        for i, f_path in enumerate(sorted(glob.glob(species_path + '/*'))):
            if i < start:
                continue
            features = np.load(f_path)
            if features is not None:
                if len(features) != 0:
                    featureset.append(features)
                    labelset.append(species)
            if (limit > 0 and i >= limit):
                break
    return featureset, labelset


def cv_eval(model, featureset, labelset, folds=5):
    scores = cross_validate(model, featureset, labelset, cv=folds, scoring=['precision_macro', 'recall_macro', 'f1_macro'], return_train_score=False)
    print('Precision:'.ljust(20), np.mean(scores['test_precision_macro']))
    print('Recall'.ljust(20), np.mean(scores['test_recall_macro']))
    print('F1'.ljust(20), np.mean(scores['test_f1_macro']))


def top_k_scores(model, predicted, labels, k):
    order = np.argsort(predicted, axis=1)
    n = model.classes_[order[:, -k:]]
    predited_z = zip(labels, n)
    u_labels = np.unique(labels)
    TP = 0.
    FN = 0.
    # FP = 0.
    # TN = 0.
    r = 0.
    for L in u_labels:
        for x, y in predited_z:
            if x == L:
                if L in y:
                    TP += 1
                else:
                    FN += 1
            # else:
            #     if L in y:
            #         FP += 1
            #     else:
            #         TN += 1
        if TP == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)
        r += recall
    r /= len(u_labels)
    return r #p, r, 2*p*r/(p+r)


def classify(env, mode=0, limit=-1):
    featureset, labelset = load_features(env, limit=limit)
    clf = SVC(kernel='rbf', C=1000, gamma=1.5, class_weight='balanced', probability=True)
    cv_eval(clf, featureset, labelset)
    return
    print('')
    k = 5
    r1 = 0.
    r3 = 0.
    r5 = 0.
    for i in range(k):
        train_f = [item for index, item in enumerate(featureset) if (index - i) % k != 0]
        train_l = [item for index, item in enumerate(labelset) if (index - i) % k != 0]
        test_f = featureset[i::k]
        test_l = labelset[i::k]
        clf.fit(train_f, train_l)
        predicted = clf.predict_proba(test_f)
        r1f = top_k_scores(clf, predicted, test_l, 1)
        r3f = top_k_scores(clf, predicted, test_l, 3)
        r5f = top_k_scores(clf, predicted, test_l, 5)
        r1 += r1f
        r3 += r3f
        r5 += r5f
    r1 /= k
    r3 /= k
    r5 /= k
    print('Recall'.ljust(20), str(r1).ljust(20), str(r3).ljust(20), r5)

print('** EXTRACTING **')
p1 = subprocess.Popen(['python3', 'extract.py', 'lab', '40', '4', '0', '0'])
p2 = subprocess.Popen(['python3', 'extract.py', 'lab', '40', '4', '1', '0'])
p3 = subprocess.Popen(['python3', 'extract.py', 'lab', '40', '4', '2', '0'])
p4 = subprocess.Popen(['python3', 'extract.py', 'lab', '40', '4', '3', '0'])

p1.wait()
p2.wait()
p3.wait()
p4.wait()

print('** CLASSIFYING **')
classify('lab')
