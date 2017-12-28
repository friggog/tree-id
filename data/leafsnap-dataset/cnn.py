import functools

from keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization, Activation)
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy


BATCH_SIZE = 256  # TODO maximise
TRAIN_STEP = 27794
VAL_STEP = 2760
EPOCHS = 128

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
    'dataset/images/train_r',
    target_size=(256, 256),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical')

validator = datagen.flow_from_directory(
    'dataset/images/test_r',
    target_size=(256, 256),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical')

model = Sequential()
model.add(Conv2D(32, (9, 9), input_shape=(256, 256, 1), padding='same'))
model.add(Conv2D(32, (9, 9), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (5, 5), padding='same'))
model.add(Conv2D(64, (5, 5), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(768, (3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
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
    validation_steps=VAL_STEP // BATCH_SIZE)

model.save_weights('leaf_net.h5')
