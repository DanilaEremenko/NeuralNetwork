# dataset8

# dataset8

import keras
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from ADDITIONAL.IMAGE_CHANGER import deform_image, noise


def load_data(mode, show=False, show_indexes=[0]):
    if not [0, 1, 2].__contains__(mode):
        Exception("Unexpected mode value")

    # loading data from keras datasets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    if mode == 2:
        for i in range(0, 60000):
            x_train[i] = deform_image(x_train[i], (28, 28), 0.3, 0, 28)
        for i in range(0, 10000):
            x_test[i] = deform_image(x_test[i], (28, 28), 0.3, 0, 28)


    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    if mode == 1:
        for i in range(0, 60000):
            x_train[i] = noise(x_train[i], 2000)
        for i in range(0, 10000):
            x_test[i] = noise(x_test[i], 2000)

    # normalizations
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    if show:
        for i in show_indexes:
            plt.imshow(x_train[i].reshape(28, 28))
            plt.show()
            plt.close()

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    for mode in (0,1,2):
        (x_train, y_train), (x_test, y_test) = load_data(mode=mode, show=True, show_indexes=[0, 1])
