from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, GlobalMaxPool1D
from keras.utils import plot_model
from keras import optimizers
import numpy
import pydot

datagen = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

batch_size = 16

datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

generator = datagen.flow_from_directory(
        'dataset/images/lab_r',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical')

validator = datagen.flow_from_directory(
        'dataset/images/field_r',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical')

model = Sequential()
model.add(Conv2D(32, (9, 9), input_shape=(256, 256, 3), strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(32, (9, 9), activation='relu', strides=(1,1), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu', strides=(1,1), padding='same'))
model.add(Conv2D(64, (5, 5), activation='relu', strides=(1,1), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1), padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', strides=(1,1), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu', strides=(1,1), padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', strides=(1,1), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu', strides=(1,1), padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', strides=(1,1), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(768, (3, 3), activation='relu', strides=(1,1), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(184, activation='softmax'))

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit_generator(
        generator,
        steps_per_epoch=2048 // batch_size,
        epochs=64,
        validation_data=validator,
        validation_steps=512 // batch_size)

model.save_weights('leaf_net.h5')
