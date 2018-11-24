import matplotlib.pyplot as plt
import numpy as np
import random as rand


def isAcceptableCoordinatesE(x, y, xBottom=5, xTop=30, yBottom=10, yTop=80):
    high = yTop - yBottom
    wide = int((yTop - yBottom) / 8)
    # Rect
    if (not range(xBottom, xTop).__contains__(x) and range(yBottom, yTop).__contains__(y)):
        return False

    # Vert
    if (range(xBottom, xBottom + wide * 2).__contains__(x) and range(yBottom, yBottom + high).__contains__(y)):
        return True
    # Bottom
    elif (range(xBottom, xBottom + high).__contains__(x) and range(yBottom, yBottom + wide * 2).__contains__(y)):
        return True
    # Top
    elif (range(xBottom, xBottom + high).__contains__(x) and range(yTop - wide * 2, yTop).__contains__(y)):
        return True
    # Middle
    elif (range(xBottom, xBottom + wide * 4).__contains__(x) and range(yBottom + 3 * wide,
                                                                       yBottom + 5 * wide).__contains__(y)):
        return True

    return False


def isAcceptableCoordinatesP(x, y, xBottom=50, xTop=75, yBottom=80, yTop=80):
    high = yTop - yBottom
    wide = int((yTop - yBottom) / 8)

    # Bigest rect
    if (not range(xBottom, xTop).__contains__(x) and range(yBottom, yTop).__contains__(y)):
        return False

    # Vertic
    if (range(xBottom, xBottom + wide).__contains__(x) and
            range(yBottom, yBottom + high * 2).__contains__(y)):
        return True

    # Bottom
    elif (range(xBottom, xBottom + high).__contains__(x) and
          range(yBottom + wide * 3, yBottom + wide * 4).__contains__(y)):
        return True

    # Top
    elif (range(xBottom, xBottom + high).__contains__(x) and
          range(yTop - wide, yTop).__contains__(y)):
        return True

    # Right side
    elif (range(xTop - wide, xTop).__contains__(x) and
          range(yBottom + 3 * wide, yTop).__contains__(y)):
        return True


def isAcceptableCoordinatesC(x, y, xBottom=5, xTop=30, yBottom=10, yTop=80):
    high = yTop - yBottom
    wide = int((yTop - yBottom) / 8)

    # Rect
    if (not range(xBottom, xTop).__contains__(x) and range(yBottom, yTop).__contains__(y)):
        return False

    # Vert
    if (range(xBottom, xBottom + wide).__contains__(x) and range(yBottom, yBottom + high).__contains__(y)):
        return True

    # Bottom
    elif (range(xBottom, xBottom + high).__contains__(x) and range(yBottom, yBottom + wide).__contains__(y)):
        return True
    # Top
    elif (range(xBottom, xBottom + high).__contains__(x) and range(yTop - wide, yTop).__contains__(y)):
        return True

    return False


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

        x = rand.randint(0, 100)
        y = rand.randint(0, 100)

        if i < train_size:
            x_train = np.append(x_train, (x, y))
        else:
            x_test = np.append(x_test, (x, y))

        if isAcceptableCoordinatesE(x, y, xBottom=20, xTop=80, yBottom=10, yTop=100):
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

    # Normalizing
    x_train /= float(x_train.max())
    y_train /= float(y_train.max())

    x_test /= float(x_test.max())
    y_test /= float(y_test.max())

    x_train_for_plt /= float(x_train_for_plt.max())
    x_test_for_plt /= float(x_test_for_plt.max())

    x_train_missed_for_plt /= float(x_train_missed_for_plt.max())
    x_test_missed_for_plt /= float(x_test_missed_for_plt.max())

    # Reshaping
    x_train.shape = (train_size, 2)
    x_test.shape = (test_size, 2)

    x_train_for_plt.shape = (int(x_train_for_plt.size / 2), 2)
    x_train_missed_for_plt.shape = (int(x_train_missed_for_plt.size / 2), 2)

    x_test_for_plt.shape = (int(x_test_for_plt.size / 2), 2)
    x_test_missed_for_plt.shape = (int(x_test_missed_for_plt.size / 2), 2)

    # Plotting train
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("train data")
    plt.plot(x_train_for_plt.transpose()[0], x_train_for_plt.transpose()[1], '.')
    plt.plot(x_train_missed_for_plt.transpose()[0], x_train_missed_for_plt.transpose()[1], '.')

    if show:
        plt.show()

    plt.close()

    # Plotting test
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("test data")
    plt.plot(x_test_for_plt.transpose()[0], x_test_for_plt.transpose()[1], '.')
    plt.plot(x_test_missed_for_plt.transpose()[0], x_test_missed_for_plt.transpose()[1], '.')

    if show:
        plt.show()

    return (x_train, y_train), (x_test, y_test)

