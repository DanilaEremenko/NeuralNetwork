import numpy as np


def func(x1, x2, x3, x4, x5):
    return x1 & x2 | x3 | x4 & x5


def func_by_arr(arr):
    return arr[0] & arr[1] | arr[2] | arr[3] & arr[4]


def load_data():
    test_size = int(32 * 0.2)
    x_train = np.empty(0)
    y_train = np.empty(0)
    x_test = np.empty(0)
    y_test = np.empty(0)
    i = 0
    for x1 in range(0, 2):
        for x2 in range(0, 2):
            for x3 in range(0, 2):
                for x4 in range(0, 2):
                    for x5 in range(0, 2):
                        if i % test_size != 0:
                            x_train = np.append(x_train, np.array([x1, x2, x3, x4, x5]))
                            y_train = np.append(y_train, func(x1, x2, x3, x4, x5))
                        else:
                            x_test = np.append(x_test, np.array([x1, x2, x3, x4, x5]))
                            y_test = np.append(y_test, func(x1, x2, x3, x4, x5))
                        i += 1

    return (x_train.reshape(32 - test_size, 5), y_train), (x_test.reshape(test_size, 5), y_test)
