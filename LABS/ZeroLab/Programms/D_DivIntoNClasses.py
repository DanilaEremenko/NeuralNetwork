import matplotlib.pyplot as plt
import math as m
import numpy as np


def isElipse(x, y, xCenter, yCenter, radVert, radHor):
    if not m.pow((x - xCenter) / radHor, 2) + m.pow((y - yCenter) / radVert, 2) <= 1:
        return False
    return True


def isRect(x, y, xMin, yMin, xMax, yMax):
    if (xMin < x and x < xMax) and (yMin < y and y < yMax):
        return True
    return False


def isTriangle(x, y, x1, y1, x2, y2, x3, y3):
    sign1 = (x1 - x) * (y2 - y1) - (x2 - x1) * (y1 - y)
    sign2 = (x2 - x) * (y3 - y2) - (x3 - x2) * (y2 - y)
    sign3 = (x3 - x) * (y1 - y3) - (x1 - x3) * (y3 - y)

    # Normalization
    try:
        sign1 /= m.fabs(sign1)
        sign2 /= m.fabs(sign2)
        sign3 /= m.fabs(sign3)
    except ZeroDivisionError:
        return False

    if int(sign1) == int(sign2) == int(sign3):
        return True

    return False


def load_data(train_size=4000, show=False):
    test_size = int(train_size * 0.2)

    x_train = np.empty(0)
    y_train = np.empty(0)

    x_test = np.empty(0)
    y_test = np.empty(0)

    train_plt_points0 = np.empty(0)
    train_plt_points1 = np.empty(0)
    train_plt_points2 = np.empty(0)
    train_plt_points3 = np.empty(0)
    train_plt_points4 = np.empty(0)

    test_plt_points0 = np.empty(0)
    test_plt_points1 = np.empty(0)
    test_plt_points2 = np.empty(0)
    test_plt_points3 = np.empty(0)
    test_plt_points4 = np.empty(0)

    for i in range(0, train_size + test_size):
        x = np.random.random()
        y = np.random.random()

        # FirstFigure
        if isRect(x, y, xMin=0.15, yMin=0.67, xMax=0.35, yMax=0.82):

            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, 1)
                train_plt_points1 = np.append(train_plt_points1, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, 1)
                test_plt_points1 = np.append(test_plt_points1, (x, y))

        # SecondFigure
        elif isElipse(x, y, xCenter=0.75, yCenter=0.75, radHor=0.15, radVert=0.15):

            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, 2)
                train_plt_points2 = np.append(train_plt_points2, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, 2)
                test_plt_points2 = np.append(test_plt_points2, (x, y))

        # ThirdFigure
        elif isTriangle(x, y, x1=0.15, y1=0.15, x2=0.30, y2=0.30, x3=0.50, y3=0.15):

            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, 3)
                train_plt_points3 = np.append(train_plt_points3, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, 3)
                test_plt_points3 = np.append(test_plt_points3, (x, y))

        # FourthFigure
        elif isRect(x, y, xMin=0.62, yMin=0.15, xMax=0.85, yMax=0.30):

            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, 4)
                train_plt_points4 = np.append(train_plt_points4, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, 4)
                test_plt_points4 = np.append(test_plt_points4, (x, y))

        else:

            if i < train_size:
                x_train = np.append(x_train, (x, y))
                y_train = np.append(y_train, 0)
                train_plt_points0 = np.append(train_plt_points0, (x, y))
            else:
                x_test = np.append(x_test, (x, y))
                y_test = np.append(y_test, 0)
                test_plt_points0 = np.append(test_plt_points0, (x, y))

    # reshaping
    x_train.shape = (int(x_train.size / 2), 2)
    train_plt_points0.shape = (int(train_plt_points0.size / 2), 2)
    train_plt_points1.shape = (int(train_plt_points1.size / 2), 2)
    train_plt_points2.shape = (int(train_plt_points2.size / 2), 2)
    train_plt_points3.shape = (int(train_plt_points3.size / 2), 2)
    train_plt_points4.shape = (int(train_plt_points4.size / 2), 2)

    x_test.shape = (int(x_test.size / 2), 2)
    test_plt_points0.shape = (int(test_plt_points0.size / 2), 2)
    test_plt_points1.shape = (int(test_plt_points1.size / 2), 2)
    test_plt_points2.shape = (int(test_plt_points2.size / 2), 2)
    test_plt_points3.shape = (int(test_plt_points3.size / 2), 2)
    test_plt_points4.shape = (int(test_plt_points4.size / 2), 2)

    # plotting
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("train data")
    plt.plot(train_plt_points0.transpose()[0], train_plt_points0.transpose()[1], '.')
    plt.plot(train_plt_points1.transpose()[0], train_plt_points1.transpose()[1], '.')
    plt.plot(train_plt_points2.transpose()[0], train_plt_points2.transpose()[1], '.')
    plt.plot(train_plt_points3.transpose()[0], train_plt_points3.transpose()[1], '.')
    plt.plot(train_plt_points4.transpose()[0], train_plt_points4.transpose()[1], '.')

    if show:
        plt.show()

    plt.close()

    # plotting
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("test data")
    plt.plot(test_plt_points0.transpose()[0], test_plt_points0.transpose()[1], '.')
    plt.plot(test_plt_points1.transpose()[0], test_plt_points1.transpose()[1], '.')
    plt.plot(test_plt_points2.transpose()[0], test_plt_points2.transpose()[1], '.')
    plt.plot(test_plt_points3.transpose()[0], test_plt_points3.transpose()[1], '.')
    plt.plot(test_plt_points4.transpose()[0], test_plt_points4.transpose()[1], '.')

    if show:
        plt.show()

    return (x_train, y_train), (x_test, y_test)

