import glob
import subprocess

import numpy as np
import time

from sklearn.model_selection import cross_validate
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn import decomposition

from extract import extract
from preprocess import resize

import warnings
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



def classify_svm(featureset,labelset,C,gamma):
    clf = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced')
    scores = cross_validate(clf, featureset, labelset, cv=5, scoring=['precision_macro', 'recall_macro', 'f1_macro'], return_train_score=False)
    print('Precision:'.ljust(20), np.mean(scores['test_precision_macro']))
    print('Recall'.ljust(20), np.mean(scores['test_recall_macro']))
    print('F1'.ljust(20), np.mean(scores['test_f1_macro']))


def classify(env, mode=0, limit=-1):
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
        featureset, labelset = load_features(env, limit=limit)
        # print('** PCA **')
        # print('Reducing from ', np.array(featureset).shape)
        # pca = decomposition.PCA(n_components=512, whiten=True)
        # featureset = pca.fit_transform(np.array(featureset))
        # print('Reduced to ', np.array(featureset).shape)

        classify_svm(featureset, labelset, 1000, 1) # TODO tweak

        # k = 10
        # fold_a = []
        # for i in range(k):
        #     train_f = [item for index, item in enumerate(featureset) if (index - i) % k != 0]
        #     train_l = [item for index, item in enumerate(labelset) if (index - i) % k != 0]
        #     clf.fit(train_f, train_l)
        #     test_f = featureset[i::10]
        #     test_l = labelset[i::10]
        #     predicted = clf.predict(test_f)
        #     correct = predicted == test_l
        #     acc = np.mean(correct)
        #     print('Fold', i + 1, 'complete with recall:', acc)
        #     fold_a.append(acc)
        # # TO SAVE
        # print('Completed with average recall: ', np.mean(fold_a))
    elif mode == 1:
        clf = SVC(kernel='rbf', C=1000, class_weight='balanced')
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


print('** EXTRACTING **')
p1 = subprocess.Popen(['python3', 'extract.py', 'lab', '15', '3', '0'])
p2 = subprocess.Popen(['python3', 'extract.py', 'lab', '15', '3', '1'])
p3 = subprocess.Popen(['python3', 'extract.py', 'lab', '15', '3', '2'])

p1.wait()
p2.wait()
p3.wait()

print('** CLASSIFYING **')
classify('lab')
