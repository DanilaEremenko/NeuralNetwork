from keras.layers import Dense, Activation
from keras.models import Sequential

from ADDITIONAL.CUSTOM_KERAS import hard_lim
import numpy as np

import LABS.ZeroLab.D_DivIntoNClasses as dataset4

if __name__ == '__main__':
    train_size = 100

    (x_train, y_train), (x_test, y_test) = dataset4.load_data(train_size=train_size, show=True)

    model = Sequential()

    model.add(Dense(17, input_dim=x_train.shape[1], activation=Activation(hard_lim), name='1',
                    weights=list([np.array(
                        [[1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0]],
                        dtype=float),
                        np.array(
                            [-0.1, 0.3, -0.1, 0.3, -0.5, 0.7, -0.1, 0.3, 0.4, 0.5, -1.7, -0.4, -0.7, 1.4, -0.7, -0.8,
                             0.2], dtype=float)])))

    model.add(Dense(7, activation=Activation(hard_lim), name='2',
                    weights=list([np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=float),
                                  np.array([-3.5, -3.5, -1.5, -0.5, -2.5, -0.5, -1.5], dtype=float)])))

    model.add(Dense(6, activation=Activation(hard_lim), name='3',
                    weights=list([np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], ], dtype=float),
                                  np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5], dtype=float)])))

    i = 0
    right = 0
    for pr in model.predict(x_train):

        if np.array_equal(pr, y_train[i]):
            right += 1
        else:
            print(pr)
            print(x_train[i])
            print(y_train[i])
        i += 1

    print(right / float(train_size) * 100, "%")
