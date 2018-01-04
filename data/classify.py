#! /usr/local/bin/python3

import glob
import subprocess
import warnings
import time
import sys

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score

# from extract import extract
# from preprocess import resize

np.set_printoptions(threshold=np.nan)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def load_features(dataset, test, limit=-1):
    train_f = []
    train_l = []
    test_f = []
    test_l = []
    count = 0
    for species_path in sorted(glob.glob(dataset + '/features/train/*')):
        species = species_path.split('/')[-1]
        for i, f_path in enumerate(glob.glob(species_path + '/*')):
            features = np.load(f_path)
            if features is not None:
                if len(features) != 0:
                    train_f.append(features)
                    train_l.append(species)
                    count += 1
                    print('Loaded:', count, '(train)', end='\r')
            if (limit > 0 and i >= limit):
                break
    print('Loaded:', count, '(train)')
    if test:
        count = 0
        for species_path in sorted(glob.glob(dataset + '/features/test/*')):
            species = species_path.split('/')[-1]
            for i, f_path in enumerate(glob.glob(species_path + '/*')):
                features = np.load(f_path)
                if features is not None:
                    if len(features) != 0:
                        test_f.append(features)
                        test_l.append(species)
                        count += 1
                        print('Loaded:', count, '(test)', end='\r')
        print('Loaded:', count, '(test)')
    return train_f, train_l, test_f, test_l


def cv_eval(model, featureset, labelset, folds=5):
    scores = cross_validate(model, featureset, labelset, cv=folds, scoring=['precision_macro', 'recall_macro', 'f1_macro'], return_train_score=False)
    print('Precision:'.ljust(20), np.mean(scores['test_precision_macro']))
    print('Recall'.ljust(20), np.mean(scores['test_recall_macro']))
    print('F1'.ljust(20), np.mean(scores['test_f1_macro']))


def top_k_scores(model, predicted, labels, k):
    order = np.argsort(predicted, axis=1)
    n = model.classes_[order[:, -k:]]
    u_labels = np.unique(labels)
    GTP = 0
    GFN = 0
    for L in u_labels:
        TP = 0.
        FN = 0.
        for i, x in enumerate(labels):
            y = n[i]
            if x == L:
                if x in y:
                    TP += 1
                else:
                    FN += 1
        GTP += TP
        GFN += FN
    if GTP == 0:
        return 0
    return GTP / (GTP + GFN)


def classify(dataset, test, limit=-1, reduce=0, gamma=1, save=False, cv=True):
    print('** CLASSIFYING **')
    print('-> loading data')
    train_f, train_l, test_f, test_l = load_features(dataset, test, limit)
    if reduce > 0:
        print('-> reducing dimensionality')
        reduction = PCA(n_components=reduce)
        if save:
            joblib.dump(reduction, 'PCA.lzma', compress=9)
        train_f = reduction.fit_transform(train_f)
        if test:
            test_f = reduction.transform(test_f)
        print('Reduced to', reduce)
    if cv:
        print('-> cross-validating')
        clf = SVC(kernel='rbf', C=1000, gamma=gamma, class_weight='balanced')
        t = time.time()
        cv_eval(clf, train_f, train_l)
        print('Fitted in', (time.time() - t))
    if test:
        print('-> testing')
        clf = SVC(kernel='rbf', C=1000, gamma=gamma, class_weight='balanced')
        t = time.time()
        clf.fit(train_f, train_l)
        print('Fitted in', (time.time() - t))
        if save:
            joblib.dump(clf, 'SVM.lzma', compress=9)
        predicted_p = clf.decision_function(test_f)
        predicted = clf.predict(test_f)
        r1 = top_k_scores(clf, predicted_p, test_l, 1)
        r3 = top_k_scores(clf, predicted_p, test_l, 3)
        r5 = top_k_scores(clf, predicted_p, test_l, 5)
        print('ACC', accuracy_score(predicted, test_l))
        # print(confusion_matrix(predicted, test_l))
        print('Recall'.ljust(20), str(r1).ljust(20), str(r3).ljust(20), r5)


def extract(dataset, test=False, limit=-1):
    print('** EXTRACTING **')
    p1 = subprocess.Popen(['python3', 'extract.py', dataset, str(test), str(limit), '4', '0', '0'])
    p2 = subprocess.Popen(['python3', 'extract.py', dataset, str(test), str(limit), '4', '1', '0'])
    p3 = subprocess.Popen(['python3', 'extract.py', dataset, str(test), str(limit), '4', '2', '0'])
    p4 = subprocess.Popen(['python3', 'extract.py', dataset, str(test), str(limit), '4', '3', '0'])
    p1.wait()
    p2.wait()
    p3.wait()
    p4.wait()


if __name__ == '__main__':
    if sys.argv[3].lower() == 'true':
        extract(sys.argv[1], test=(sys.argv[2].lower() == 'true'))
    classify(sys.argv[1], test=(sys.argv[2].lower() == 'true'), limit=-1, reduce=256, gamma=4.0, cv=True)
