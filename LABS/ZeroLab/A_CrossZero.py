import matplotlib.pyplot as plt
import matplotlib.patches as patch
import numpy as np


def add_el_on_graph(x, y, el):
    if el == 1:
        plt.gca().add_patch(
            patch.Rectangle((x / 4.0, y / 4.0), 0.25, 0.25, color='#000000',
                            fill=True))
    else:
        plt.gca().add_patch(
            patch.Rectangle((x / 4.0, y / 4.0), 0.25, 0.25, color='#FFFFFF',
                            fill=True))


def load_data(y_train, show=False):
    if y_train.shape != (4, 4):
        raise TypeError("Shape can be only (4,4)")

    train_size = int(y_train.shape[0] * y_train.shape[1])
    x_train = np.zeros(train_size * 2)

    i = 0
    for y in np.arange(y_train.shape[0] - 1, -0.1, -1, dtype=int):
        for x in np.arange(y_train.shape[1] - 1, -0.1, -1, dtype=int):
            add_el_on_graph(x, y_train.shape[0] - y - 1, y_train[y][x])
            x_train[i] = x / 4.0
            x_train[i + 1] = y / 4.0
            i += 2

    if show:
        plt.show()

    return (x_train.reshape(train_size, 2), y_train.reshape(train_size, 1))
