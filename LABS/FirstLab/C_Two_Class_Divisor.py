from keras.layers import Dense, Activation
from keras.models import Sequential

from ADDITIONAL.CUSTOM_KERAS import hard_lim
import numpy as np

import LABS.ZeroLab.C_DivIntoTwoClasses as dataset3

if __name__ == '__main__':
    train_size = 4000

    (x_train, y_train), (x_test, y_test) = dataset3.load_data(train_size=train_size, show=False)

    model = Sequential()

    model.add(Dense(7, input_dim=x_train.shape[1], activation=Activation(hard_lim), name='1',
                    weights=list([np.array([[0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 1.0],
                                            [1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0]], dtype=float),
                                  np.array([-0.1, 0.5, -0.2, 0.6, 0.9, 0.9, -1.4], dtype=float)])))

    model.add(Dense(2, activation=Activation(hard_lim), name='2',
                    weights=list([np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
                                            [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=float),
                                  np.array([-3.5, -2.5], dtype=float)])))

    model.add(Dense(1, activation=Activation(hard_lim), name='3',
                    weights=list([np.array([[1.0], [1.0]], dtype=float),
                                  np.array([-0.5], dtype=float)])))

    i = 0
    right = 0
    for pr in model.predict(x_train):
        if y_train[i] == pr:
            right += 1
        i += 1

    print(right/float(y_train.size)*100, "%")
