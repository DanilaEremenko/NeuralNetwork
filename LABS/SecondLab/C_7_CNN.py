from __future__ import print_function
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten
from keras.utils.vis_utils import plot_model

if __name__ == '__main__':
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=(1,2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1, activation='softmax'))

    plot_model(model, to_file='CNN.png')
