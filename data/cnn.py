#! /usr/local/bin/python3

import functools
import os
import numpy as np

from keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization, Activation, GaussianDropout)
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import ModelCheckpoint
from keras import applications

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

top3_acc = functools.partial(top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'
top5_acc = functools.partial(top_k_categorical_accuracy, k=5)
top5_acc.__name__ = 'top5_acc'


def preprocess_input_vgg(x):
    """Wrapper around keras.applications.vgg16.preprocess_input()
    to make it compatible for use with keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument.

    Parameters
    ----------
    x : a numpy 3darray (a single image to be preprocessed)

    Note we cannot pass keras.applications.vgg16.preprocess_input()
    directly to to keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument because the former expects a
    4D tensor whereas the latter expects a 3D tensor. Hence the
    existence of this wrapper.

    Returns a numpy 3darray (the preprocessed image).

    """
    from keras.applications.vgg16 import preprocess_input
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    return X[0]


BATCH_SIZE = 48
TRAIN_STEP = 27895
VAL_STEP = 2760
EPOCHS = 768
WIDTHS = [64, 128, 256, 512, 768, 1024]
KERNELS = [3, 3, 3, 3, 3, 3]
DROP = [0, 0.5]

datagen = ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.1,
    height_shift_range=0.1,
    preprocessing_function=preprocess_input_vgg,
    # rescale=1. / 255,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0,
    zoom_range=0.2,
    fill_mode='nearest',)


generator = datagen.flow_from_directory(
    'leafsnap/images/train_r',
    target_size=(244, 244),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    save_to_dir='prev',
    save_format='jpeg')

vdatagen = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    preprocessing_function=preprocess_input_vgg,
    # rescale=1. / 255,
    horizontal_flip=False,
    vertical_flip=False,
    shear_range=0,
    zoom_range=0,
    fill_mode='nearest')

validator = vdatagen.flow_from_directory(
    'leafsnap/images/test_r',
    target_size=(244, 244),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical')


model = applications.VGG16(weights='imagenet', include_top=False)
model.add(Flatten(input_shape=model.output_shape[1:]))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(184, activation='softmax'))

for layer in model.layers[:25]:
    layer.trainable = False

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
