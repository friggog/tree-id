#! /usr/local/bin/python3

import functools
import glob
import os
import subprocess

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import (Activation, BatchNormalization, Dense, Dropout, LeakyReLU, GaussianNoise)
from keras.metrics import top_k_categorical_accuracy
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils import class_weight

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BATCH_SIZE = 1024
EPOCHS = 2048


def make_model(isize, d, w, act, norm=False, dropout=0.5, noise=0.2):
    model = Sequential()
    model.add(Dense(w, input_dim=isize))
    for i in range(d):
        if i > 0:
            model.add(Dense(w))
        model.add(GaussianNoise(noise))
        if norm:
            model.add(BatchNormalization())
        if act == 'lrelu':
            model.add(LeakyReLU())
        else:
            model.add(Activation(act))
        model.add(Dropout(dropout))
    model.add(Dense(184, activation='softmax'))
    return model


def load_data(env):
    train_l = []
    train_f = []
    print('** LOADING **')
    count = 0
    for j, species_path in enumerate(sorted(glob.glob('leafsnap/features/' + env + '/*'))):
        for i, feature_path in enumerate(sorted(glob.glob(species_path + '/*'))):
            f = np.load(feature_path)
            train_l.append(j)
            # o = []
            # for c in f:
                # hist = np.histogram(c, bins=64, range=(0, 1))[0]
                # fft = np.abs(np.fft.rfft(c, len(c)))
                # o.extend(hist)
                # o.extend(fft)
            train_f.append(f)
            count += 1
            print('Loaded:', count, end='\r')
    print('Training Data:', len(train_l))
    train_f = np.array(train_f).reshape((len(train_l), len(f)))
    # train_f = (train_f - train_f.mean(axis=0)) / train_f.std(axis=0)
    weights = class_weight.compute_class_weight('balanced', np.unique(train_l), train_l)
    weights = dict(enumerate(weights))
    train_l = to_categorical(train_l, num_classes=184)
    return (train_f, train_l, weights)


def extract_fs(env, lim):
    print('** EXTRACTING **')
    p1 = subprocess.Popen(['python3', 'extract.py', env, str(lim), '4', '0', '2'])
    p2 = subprocess.Popen(['python3', 'extract.py', env, str(lim), '4', '1', '2'])
    p3 = subprocess.Popen(['python3', 'extract.py', env, str(lim), '4', '2', '2'])
    p4 = subprocess.Popen(['python3', 'extract.py', env, str(lim), '4', '3', '2'])

    p1.wait()
    p2.wait()
    p3.wait()
    p4.wait()


def do(data, name, model):
    train_f, train_l, weights = data

    print('** TRAINING **')

    # top2_acc = functools.partial(top_k_categorical_accuracy, k=2)
    # top2_acc.__name__ = 'top2_acc'
    # top3_acc = functools.partial(top_k_categorical_accuracy, k=3)
    # top3_acc.__name__ = 'top3_acc'

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])  # , top2_acc, top3_acc])

    history = model.fit(x=train_f,
                        y=train_l,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_split=0.2,
                        class_weight=weights,
                        callbacks=[ModelCheckpoint(name + '.h5', monitor='val_acc', save_best_only=True)])
    print(name, 'DONE')
    np.save(name + 'training', history.history)
    # print('** EVALUATING **')
    # score = model.evaluate(val_f, val_l, batch_size=BATCH_SIZE)
    # print(score)


if __name__ == "__main__":
    # extract_fs('lab', -1)
    data = load_data('train')
    depth = 3
    width = 32
    drop = 0.3
    noise = 0.0
    model = make_model(data[0].shape[1], depth, width, 'relu', norm=True, dropout=drop, noise=noise)
    do(data, 'nets/relu_' +str(depth) +'_' +str(width) +'_' +str(drop) + '_' + str(noise), model)
