import numpy as np


#dataset2


import numpy as np


def func(x1, x2, x3, x4, x5):
    """
    ~x1 & ~x2 & x3 & ~x4 & x5  |
    x1 & ~x2 & ~x3 & ~x4 & x5  |
    x1 & ~x2 & ~x3 & x4 & x5   |
    x1 & ~x2 & x3 & x4 & x5    |
    x1 & x2 & ~x3 & ~x4 & ~x5
    
    """
    return (not x1) & (not x2) & x3 & (not x4) & x5 | x1 & (not x2) & (not x3) & (not x4) & x5 \
           | x1 & (not x2) & (not x3) & x4 & x5 \
           | x1 & (not x2) & x3 & x4 & x5 | x1 & x2 & (not x3) & (not x4) & (not x5)



def func_by_arr(arr):
    return func(bool(arr[0]),bool(arr[1]),bool(arr[2]),bool(arr[3]),bool(arr[4]))


def load_data():
    x_train = np.empty(0)
    y_train = np.empty(0)
    for x1 in range(0, 2):
        for x2 in range(0, 2):
            for x3 in range(0, 2):
                for x4 in range(0, 2):
                    for x5 in range(0, 2):
                        x_train = np.append(x_train, np.array([x1, x2, x3, x4, x5]))
                        y_train = np.append(y_train, func(x1, x2, x3, x4, x5))

    return (x_train.reshape(32, 5), y_train)

