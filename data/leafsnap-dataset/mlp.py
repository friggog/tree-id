import functools
import glob

import keras
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, Dropout, BatchNormalization, Conv1D, Flatten, MaxPooling1D
from keras.models import Sequential
from keras.metrics import top_k_categorical_accuracy
from keras.utils import to_categorical

BATCH_SIZE = 192
EPOCHS = 512

def do(env):
    train_l = []
    train_f = []
    test_l = []
    test_f = []

    print('** LOADING **')
    count = 0
    for j, species_path in enumerate(sorted(glob.glob('dataset/mlp_features/' + env + '/*'))):
        for i, feature_path in enumerate(sorted(glob.glob(species_path + '/*'))):
            f = np.load(feature_path) / 255
            if i < 10:
                test_l.append(j)
                test_f.append(f)
            else:
                train_l.append(j)
                train_f.append(f)
            count += 1
            print('Loaded:', count, end='\r')
    print('Train Set:', len(train_l))
    print('Test Set:', len(test_l))
    train_f = np.array(train_f).reshape((len(train_l), 128))
    test_f = np.array(test_f).reshape((len(test_l), 128))
    train_l = to_categorical(train_l, num_classes=184)
    test_l = to_categorical(test_l, num_classes=184)

    print('** TRAINING **')
    model = Sequential()
    model.add(Dense(128, use_bias=False, input_dim=128))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(512, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(512, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(184, activation='softmax'))

    top2_acc = functools.partial(top_k_categorical_accuracy, k=2)
    top2_acc.__name__ = 'top2_acc'
    top3_acc = functools.partial(top_k_categorical_accuracy, k=3)
    top3_acc.__name__ = 'top3_acc'

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', top2_acc, top3_acc])
                  
    history = model.fit(train_f, 
                        train_l, 
                        batch_size=BATCH_SIZE, 
                        epochs=EPOCHS, 
                        validation_data=(test_f, test_l),
                        callbacks=[ModelCheckpoint('WIP.h5', monitor='val_acc', save_best_only=True)])

    print(history.history)
    print('** EVALUATING **')
    score = model.evaluate(test_f, test_l, batch_size=BATCH_SIZE)
    print(score)

if __name__ == "__main__":
    do('lab')
