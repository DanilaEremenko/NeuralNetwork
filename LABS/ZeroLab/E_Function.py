# dataset5

import matplotlib.pyplot as plt
import math as m
import numpy as np


def function(x):
    x = x * 100
    return 0.25 + x / 200.0 + 0.25 * m.sin(2.0 / 3.0 * x * m.sin(x / 50.0 + 3.0 * m.pi / 2.0))


def load_data(train_size=200, show=False):
    test_size = int(train_size * 0.2)

    h = 1.0 / float(train_size + test_size)

    x_train = np.empty(0)
    y_train = np.empty(0)

    x_test = np.empty(0)
    y_test = np.empty(0)

    i = 0
    for x in np.arange(0.0, 1.0, h, dtype=float):
        if i % 5 == 0:
            x_test = np.append(x_test, x)
            y_test = np.append(y_test, function(x))
        else:
            x_train = np.append(x_train, x)
            y_train = np.append(y_train, function(x))

        i += 1

    plt.plot(x_train, y_train, '.')
    plt.plot(x_test, y_test, '.')
    plt.legend(('train_data', 'test_data'), loc='upper left', shadow=True)
    if show:
        plt.show()
    plt.close()

    x_train = np.transpose(np.append(x_train, np.ones(x_train.size)).reshape(2, x_train.size))
    x_test = np.transpose(np.append(x_test, np.ones(x_test.size)).reshape(2, x_test.size))

    return (x_train, y_train), (x_test, y_test)
