import matplotlib.pyplot as plt
import math as m

import numpy as np


def function(x):
    x = x * 100
    return 0.25 + x / 200 + 0.25 * m.sin(2 / 3 * x * m.sin(x / 50 + 3 * m.pi / 2))


if __name__ == '__main__':
    xArray = np.arange(0.0, 1.0, 0.001, dtype=float)
    yArray = []

    for x in xArray:
        yArray.append(function(x))
    plt.plot(xArray, yArray)
    plt.plot(xArray, yArray, color='mediumvioletred')
    plt.ylim(0, 1)

    SHOW = input('SHOW? [Y/n]')
    SAVE = input('SAVE?[Y/n]')
    if SAVE == 'Y':
        plt.savefig('../Pictures/5_Function.png', dpi=200)
    if SHOW == 'Y':
        plt.show()
