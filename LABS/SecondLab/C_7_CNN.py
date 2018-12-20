from __future__ import print_function
from keras.models import Model
from keras.layers import Dense, Input
from keras.utils.vis_utils import plot_model

if __name__ == '__main__':
    input_layer = Input(shape=(100, 1))

    hidden = Dense(1, input_dim=1, name='custom')(input_layer)

    output = Dense(1, input_dim=2, activation='relu', name='out')([hidden, input_layer])

    model = Model(inputs=input_layer, outputs=[output])

    plot_model(model, to_file="CNN.png")
