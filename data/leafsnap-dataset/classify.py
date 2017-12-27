#! /usr/local/bin/python3

import glob
import subprocess
import warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.externals import joblib

from extract import extract
from preprocess import resize

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def load_features(test, limit=-1):
    train_f = []
    train_l = []
    test_f = []
    test_l = []
    for species_path in sorted(glob.glob('dataset/features/train/*')):
        species = species_path.split('/')[-1]
        for i, f_path in enumerate(glob.glob(species_path + '/*')):
            features = np.load(f_path)
            if features is not None:
                if len(features) != 0:
                    train_f.append(features)
                    train_l.append(species)
            if (limit > 0 and i >= limit):
                break
    if test:
        for species_path in sorted(glob.glob('dataset/features/test/*')):
            species = species_path.split('/')[-1]
            for i, f_path in enumerate(glob.glob(species_path + '/*')):
                features = np.load(f_path)
                if features is not None:
                    if len(features) != 0:
                        test_f.append(features)
                        test_l.append(species)
    return train_f, train_l, test_f, test_l


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
    return r  # p, r, 2*p*r/(p+r)


def classify(test, limit=-1):
    train_f, train_l, test_f, test_l = load_features(test, limit)
    clf = SVC(kernel='rbf', C=1000, gamma=1, class_weight='balanced', probability=True)
    cv_eval(clf, train_f, train_l)
    if test:
        clf.fit(train_f, train_l)
        joblib.dump(clf, 'SVM.pkl')
        predicted = clf.predict_proba(test_f)
        r1 = top_k_scores(clf, predicted, test_l, 1)
        r3 = top_k_scores(clf, predicted, test_l, 3)
        r5 = top_k_scores(clf, predicted, test_l, 5)
        print('Recall'.ljust(20), str(r1).ljust(20), str(r3).ljust(20), r5)


test = True
lim = -1

print('** EXTRACTING **')
p1 = subprocess.Popen(['python3', 'extract.py', str(test), str(lim), '4', '0', '0'])
p2 = subprocess.Popen(['python3', 'extract.py', str(test), str(lim), '4', '1', '0'])
p3 = subprocess.Popen(['python3', 'extract.py', str(test), str(lim), '4', '2', '0'])
p4 = subprocess.Popen(['python3', 'extract.py', str(test), str(lim), '4', '3', '0'])

p1.wait()
p2.wait()
p3.wait()
p4.wait()

print('** CLASSIFYING **')
classify(test, lim)
