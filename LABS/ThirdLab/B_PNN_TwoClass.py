import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from neupy import algorithms
from mpl_toolkits.mplot3d import Axes3D

import LABS.ZeroLab.C_DivIntoTwoClasses as dataset3


def diff_std():
    (x_train, y_train), (x_test, y_test) = dataset3.load_data(train_size=12000, show=True)

    for std in [0.1, 0.001, 0.0001]:
        pnn = algorithms.PNN(std=std, verbose=True)

        pnn.train(x_train, y_train)

        y_predicted = pnn.predict(x_test)

        mae = (np.abs(y_test - y_predicted)).mean()

        plt_x_zero = np.empty(0)
        plt_y_zero = np.empty(0)

        plt_x_one = np.empty(0)
        plt_y_one = np.empty(0)

        acc = 0.0
        i = 0
        for coord in x_test:
            if y_predicted[i] < 0.5:
                plt_x_zero = np.append(plt_x_zero, coord[0])
                plt_y_zero = np.append(plt_y_zero, coord[1])
            elif y_predicted[i] >= 0.5:
                plt_x_one = np.append(plt_x_one, coord[0])
                plt_y_one = np.append(plt_y_one, coord[1])
            i += 1

        plt.plot(plt_x_zero, plt_y_zero, '.')
        plt.plot(plt_x_one, plt_y_one, '.')

        plt.title('2 class classification\nstd = %.4f\nmae =%.4f' % (std, mae))

        plt.xlim(0, 1.3)
        plt.ylim(0, 1)

        plt.legend(('0 class', '1 class'), loc='upper right', shadow=True)

        plt.show()
        plt.close()


def diff_train():
    maes = [0, 0, 0, 0]
    train_size = [24000, 12000, 5000, 2000]
    std = [0.00035, 0.001, 0.0035, 0.01]

    for j in range(0, 4):
        (x_train, y_train), (x_test, y_test) = dataset3.load_data(train_size=train_size[j], show=False)

        pnn = algorithms.PNN(std=std[j], verbose=True)

        pnn.train(x_train, y_train)

        y_predicted = pnn.predict(x_test)

        mae = (np.abs(y_test - y_predicted)).mean()

        plt_x_zero = np.empty(0)
        plt_y_zero = np.empty(0)

        plt_x_one = np.empty(0)
        plt_y_one = np.empty(0)

        acc = 0.0
        i = 0
        for coord in x_test:
            if y_predicted[i] < 0.5:
                plt_x_zero = np.append(plt_x_zero, coord[0])
                plt_y_zero = np.append(plt_y_zero, coord[1])
            elif y_predicted[i] >= 0.5:
                plt_x_one = np.append(plt_x_one, coord[0])
                plt_y_one = np.append(plt_y_one, coord[1])
            i += 1

        plt.plot(plt_x_zero, plt_y_zero, '.')
        plt.plot(plt_x_one, plt_y_one, '.')

        plt.title('2 class classification\ntrain size = %d\nstd = %.4f\nmae =%.4f' % (train_size[j], std[j], mae))

        maes[j] = mae

        plt.xlim(0, 1.3)
        plt.ylim(0, 1)

        plt.legend(('0 class', '1 class'), loc='upper right', shadow=True)

        plt.show()
        plt.close()
    return train_size, std, maes


if __name__ == '__main__':
    # diff_std()
    train_size, std, maes = diff_train()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('train size')
    ax.set_ylabel('std')
    ax.set_zlabel('error rate')
    df = pd.DataFrame({'x': train_size, 'y': std, 'z': maes})
    surf = ax.plot_trisurf(df.x, df.y, df.z, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
