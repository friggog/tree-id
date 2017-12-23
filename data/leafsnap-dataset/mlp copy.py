#! /usr/local/bin/python3

import functools
import glob
import subprocess

import keras
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import (Activation, BatchNormalization, Conv1D, Dense, Dropout, Flatten, LeakyReLU, MaxPooling1D, PReLU)
from keras.metrics import top_k_categorical_accuracy
from keras.models import Sequential
from keras.utils import to_categorical

BATCH_SIZE = 128
EPOCHS = 512
F_LEN = 65 * 3


def make_model(d, w, act, norm=False, dropout=0.5):
    model = Sequential()
    model.add(Dense(w, input_dim=F_LEN))
    for i in range(d - 1):
        if i > 0:
            model.add(Dense(w))
        if norm:
            model.add(BatchNormalization())
        if act == 'lrelu':
            model.add(LeakyReLU())
        elif act == 'prelu':
            model.add(LeakyReLU())
        else:
            model.add(Activation(act))
        model.add(Dropout(dropout))
    model.add(Dense(184, activation='softmax'))
    return model


def load_data(env):
    train_l = []
    train_f = []
    val_l = []
    val_f = []
    print('** LOADING **')
    count = 0
    for j, species_path in enumerate(sorted(glob.glob('dataset/mlp_features/' + env + '/*'))):
        for i, feature_path in enumerate(sorted(glob.glob(species_path + '/*'))):
            f = np.load(feature_path)
            if i < 20:
                val_l.append(j)
                val_f.append(f)
            else:
                train_l.append(j)
                train_f.append(f)
            count += 1
            print('Loaded:', count, end='\r')
    print('Training Data:', len(train_l))
    print('Validation Data:', len(val_l))
    train_f = np.array(train_f).reshape((len(train_l), F_LEN))
    train_l = to_categorical(train_l, num_classes=184)
    val_f = np.array(val_f).reshape((len(val_l), F_LEN))
    val_l = to_categorical(val_l, num_classes=184)
    return (train_f, val_f, train_l, val_l)


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
    train_f, val_f, train_l, val_l = data

    print('** TRAINING **')

    top2_acc = functools.partial(top_k_categorical_accuracy, k=2)
    top2_acc.__name__ = 'top2_acc'
    top3_acc = functools.partial(top_k_categorical_accuracy, k=3)
    top3_acc.__name__ = 'top3_acc'

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', top2_acc, top3_acc])

    history = model.fit(train_f,
                        train_l,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(val_f, val_l),
                        callbacks=[ModelCheckpoint(name + '.h5', monitor='val_acc', save_best_only=True)])
    print(name, 'DONE')
    np.save(name + 'training', history.history)
    # print('** EVALUATING **')
    # score = model.evaluate(val_f, val_l, batch_size=BATCH_SIZE)
    # print(score)

if __name__ == "__main__":
    extract_fs('lab', -1)
    data = load_data('lab')
    model = make_model(4, 512, 'lrelu', True, 0.2)
    do(data, 'nets/lrelu_4_512_T', model)
