#dataset8

# dataset8

import keras
import numpy as np
import matplotlib.image as mpimg


def load_data():
    # loading data from keras datasets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # reshaping
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    # normalizations
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test)=load_data()


#i can be changed
i=2

print(y_train[i])
imgplt=plt.imshow(x_train[i].reshape(28,28))

