from __future__ import print_function
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten
from keras.utils.vis_utils import plot_model

if __name__ == '__main__':
    model = Sequential()

    input_layer = Dense(1, name='input')

    hidden = Dense(1, name='hidden')

    output_layer=Dense(1,name='output')(input_layer)


    model.add(input_layer)

    model.add(hidden)

    model.add(Dense(1)(input_layer))

    plot_model(model, to_file="CNN.png")
