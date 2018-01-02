#! /usr/local/bin/python3

import functools
import os

from keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization, Activation, GaussianDropout)
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import ModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


BATCH_SIZE = 48
TRAIN_STEP = 27895
VAL_STEP = 2760
EPOCHS = 768
WIDTHS = [64, 128, 256, 512, 768, 1024]
KERNELS = [3, 3, 3, 3, 3, 3]
DROP = [0, 0.5]

datagen = ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0,
    zoom_range=0.2,
    fill_mode='nearest')


generator = datagen.flow_from_directory(
    'leafsnap/images/train_r',
    target_size=(256, 256),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical')

vdatagen = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    rescale=1. / 255,
    horizontal_flip=False,
    vertical_flip=False,
    shear_range=0,
    zoom_range=0,
    fill_mode='nearest')

validator = vdatagen.flow_from_directory(
    'leafsnap/images/test_r',
    target_size=(256, 256),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical')

model = Sequential()
model.add(Conv2D(WIDTHS[0], (KERNELS[0], KERNELS[0]), input_shape=(256, 256, 1), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(WIDTHS[0], (KERNELS[0], KERNELS[0]), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GaussianDropout(DROP[0]))

model.add(Conv2D(WIDTHS[1], (KERNELS[1], KERNELS[1]), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(WIDTHS[1], (KERNELS[1], KERNELS[1]), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GaussianDropout(DROP[0]))

model.add(Conv2D(WIDTHS[2], (KERNELS[2], KERNELS[2]), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(WIDTHS[2], (KERNELS[2], KERNELS[2]), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GaussianDropout(DROP[0]))

model.add(Conv2D(WIDTHS[3], (KERNELS[3], KERNELS[3]), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(WIDTHS[3], (KERNELS[3], KERNELS[3]), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GaussianDropout(DROP[0]))

model.add(Conv2D(WIDTHS[4], (KERNELS[4], KERNELS[4]), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(WIDTHS[4], (KERNELS[4], KERNELS[4]), padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GaussianDropout(DROP[0]))

model.add(Flatten())
model.add(Dense(WIDTHS[-1]))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(DROP[1]))

model.add(Dense(WIDTHS[-1]))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(DROP[1]))

model.add(Dense(184, activation='softmax'))

top3_acc = functools.partial(top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'
top5_acc = functools.partial(top_k_categorical_accuracy, k=5)
top5_acc.__name__ = 'top5_acc'

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', top3_acc, top5_acc])

model.fit_generator(
    generator,
    steps_per_epoch=TRAIN_STEP // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validator,
    validation_steps=VAL_STEP // BATCH_SIZE,
    callbacks=[ModelCheckpoint('leaf_net_t.h5', save_best_only=True)])

model.save_weights('leaf_net.h5')
