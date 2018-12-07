import matplotlib.pyplot as plt
import numpy as np


def linear(x, y, k, b):
    if k * x + b - y > 0:
        return 1
    else:
        return 0


def non_linear(x, y, k, b):
    if k * x * x + b - y > 0:
        return 1
    else:
        return 0


def func(x, k, b, func_type):
    if func_type == 'lin':
        return k * x + b
    else:
        return k * x * x + b


def load_data(train_size, k, b, show=False, func_type='lin'):
    test_size = int(train_size * 0.2)

    x_train = np.empty(0)
    y_train = np.empty(0)

    x_test = np.empty(0)
    y_test = np.empty(0)

    train_plt_points0 = np.empty(0)
    train_plt_points1 = np.empty(0)

    test_plt_points0 = np.empty(0)
    test_plt_points1 = np.empty(0)

    for i in range(0, train_size + test_size):
        x = np.random.random()
        y = np.random.random()

        if func_type == 'lin':
            y_type = linear(x=x, y=y, k=k, b=b)
        elif func_type == 'n_lin':
            y_type = non_linear(x=x, y=y, k=k, b=b)
        else:
            raise TypeError('func_type can be lin or n_lin')

        if i < train_size:
            x_train = np.append(x_train, (x, y))
            y_train = np.append(y_train, y_type)

            if y_type == 1:
                train_plt_points1 = np.append(train_plt_points1, (x, y))
            elif y_type == 0:
                train_plt_points0 = np.append(train_plt_points0, (x, y))

        else:
            x_test = np.append(x_test, (x, y))
            y_test = np.append(y_test, y_type)

            if y_type == 1:
                test_plt_points1 = np.append(test_plt_points1, (x, y))
            elif y_type == 0:
                test_plt_points0 = np.append(test_plt_points0, (x, y))

    # reshaping
    x_train.shape = (int(x_train.size / 2), 2)
    train_plt_points0.shape = (int(train_plt_points0.size / 2), 2)
    train_plt_points1.shape = (int(train_plt_points1.size / 2), 2)

    x_test.shape = (int(x_test.size / 2), 2)
    test_plt_points0.shape = (int(test_plt_points0.size / 2), 2)
    test_plt_points1.shape = (int(test_plt_points1.size / 2), 2)

    # plotting
    plt.xlim(0, 1.3)
    plt.ylim(0, 1)
    plt.title("train data")
    plt.plot(train_plt_points0.transpose()[0], train_plt_points0.transpose()[1], '.')
    plt.plot(train_plt_points1.transpose()[0], train_plt_points1.transpose()[1], '.')

    plt.legend(('0 class', '1 class'), loc='upper right', shadow=True)

    if show:
        plt.show()

    plt.close()

    # plotting
    plt.xlim(0, 1.3)
    plt.ylim(0, 1)
    plt.title("test data")
    plt.plot(test_plt_points0.transpose()[0], test_plt_points0.transpose()[1], '.')
    plt.plot(test_plt_points1.transpose()[0], test_plt_points1.transpose()[1], '.')

    plt.legend(('0 class', '1 class'), loc='upper right', shadow=True)

    if show:
        plt.show()

    return (x_train, y_train), (x_test, y_test)


def load_data_func(train_size, k, b, show=False, func_type='lin'):
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
            y_test = np.append(y_test, func(x=x, k=k, b=b, func_type=func_type))

        else:
            x_train = np.append(x_train, x)
            y_train = np.append(y_train, func(x=x, k=k, b=b, func_type=func_type))

        i += 1

    plt.plot(x_train, y_train, '.')
    plt.plot(x_test, y_test, '.')
    plt.legend(('train_data', 'test_data'), loc='upper left', shadow=True)
    if show:
        plt.show()
    plt.close()

    return (x_train, y_train), (x_test, y_test)
