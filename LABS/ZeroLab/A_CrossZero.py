#dataset1

import matplotlib.pyplot as plt
import matplotlib.patches as patch
import numpy as np


def load_data(y_train, show=False):
    if y_train.shape != (4, 4):
        raise TypeError("Shape can be only (4,4)")

    train_size = int(y_train.shape[0] * y_train.shape[1])
    x_train = np.zeros(train_size * 2)

    graph_pts_zero_x = np.zeros(8)
    graph_pts_zero_y = np.zeros(8)
    graph_pts_one_x = np.zeros(8)
    graph_pts_one_y = np.zeros(8)

    i = 0
    one_i = 0
    zero_i = 0
    for y in np.arange(0,y_train.shape[0], 1, dtype=int):
        for x in np.arange(0,y_train.shape[1], 1, dtype=int):
            if y_train[y_train.shape[0] - y - 1][x] == 1:
                graph_pts_one_x[one_i] = x/ 4.0 + 0.125
                graph_pts_one_y[one_i] =  y / 4.0+0.125
                one_i += 1
            elif y_train[y_train.shape[0] - y - 1][x] == 0:
                graph_pts_zero_x[zero_i] =  x / 4.0 + 0.125
                graph_pts_zero_y[zero_i] = y / 4.0+0.125
                zero_i += 1

            x_train[i] = x / 4.0+0.125
            x_train[i + 1] = y / 4.0+0.125
            i += 2

    plt.plot(graph_pts_zero_x, graph_pts_zero_y, '.')
    plt.plot(graph_pts_one_x, graph_pts_one_y, '.')
    plt.xlim(0, 1.2)
    plt.ylim(0, 1)
    
    plt.legend(('0 class', '1 class'),loc='upper right', shadow=True)
    

    if show:
        plt.show()
    plt.close()

    return (x_train.reshape(train_size, 2), y_train.reshape(train_size, 1))
