import matplotlib.pyplot as plt
import numpy as np


import LABS.ZeroLab.D_DivIntoNClasses as dataset4


def load_data(train_size=2000, show=False):
    test_size = int(train_size * 0.2)

    x_train = np.empty(0)
    y_train = np.empty(0)

    x_test = np.empty(0)
    y_test = np.empty(0)

    x_train_for_plt = np.empty(0)
    x_train_missed_for_plt = np.empty(0)

    x_test_for_plt = np.empty(0)
    x_test_missed_for_plt = np.empty(0)

    for i in range(train_size + test_size):

        x = np.random.random()
        y = np.random.random()

        if i < train_size:
            x_train = np.append(x_train, (x, y))
        else:
            x_test = np.append(x_test, (x, y))

        if dataset4.isRect(x, y, xMin=0.2, xMax=0.6, yMin=0.1, yMax=0.5) or \
                dataset4.isTriangle(x, y, x1=0.5, x2=0.9, x3=0.9, y1=0.9, y2=0.5, y3=0.9):
            if i < train_size:
                x_train_for_plt = np.append(x_train_for_plt, (x, y))
                y_train = np.append(y_train, 1)
            else:
                x_test_for_plt = np.append(x_test_for_plt, (x, y))
                y_test = np.append(y_test, 1)
        else:
            if i < train_size:
                x_train_missed_for_plt = np.append(x_train_missed_for_plt, (x, y))
                y_train = np.append(y_train, 0)
            else:
                x_test_missed_for_plt = np.append(x_test_missed_for_plt, (x, y))
                y_test = np.append(y_test, 0)

    # Reshaping
    x_train.shape = (train_size, 2)
    x_test.shape = (test_size, 2)

    x_train_for_plt.shape = (int(x_train_for_plt.size / 2), 2)
    x_train_missed_for_plt.shape = (int(x_train_missed_for_plt.size / 2), 2)

    x_test_for_plt.shape = (int(x_test_for_plt.size / 2), 2)
    x_test_missed_for_plt.shape = (int(x_test_missed_for_plt.size / 2), 2)

    # Plotting train
    plt.xlim(0, 1.3)
    plt.ylim(0, 1)
    plt.title("train data")
    plt.plot(x_train_for_plt.transpose()[0], x_train_for_plt.transpose()[1], '.')
    plt.plot(x_train_missed_for_plt.transpose()[0], x_train_missed_for_plt.transpose()[1], '.')
    
    plt.legend(('0 class','1 class'),loc='upper right', shadow=True)
    
    
    if show:
        plt.show()

    plt.close()

    # Plotting test
    plt.xlim(0, 1.3)
    plt.ylim(0, 1)
    plt.title("test data")
    plt.plot(x_test_for_plt.transpose()[0], x_test_for_plt.transpose()[1], '.')
    plt.plot(x_test_missed_for_plt.transpose()[0], x_test_missed_for_plt.transpose()[1], '.')

    plt.legend(('0 class','1 class'),loc='upper right', shadow=True)
    
    if show:
        plt.show()
    plt.close()

    return (x_train, y_train), (x_test, y_test)
