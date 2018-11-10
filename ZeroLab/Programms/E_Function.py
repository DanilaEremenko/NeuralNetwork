import matplotlib.pyplot as plt
import math as m

import numpy as np


def function(x):
    x = x * 100
    return 0.25 + x / 200 + 0.25 * m.sin(2 / 3 * x * m.sin(x / 50 + 3 * m.pi / 2))


if __name__ == '__main__':
    xArray = np.arange(0.0, 1.0, 0.001, dtype=float)
    yArray = np.array([function(y) for y in xArray])

    plt.plot(xArray, yArray)
    plt.plot(xArray, yArray, color='mediumvioletred')
    plt.ylim(0, 1)


    if input('SAVE?[Y/n]') == 'Y':
        plt.savefig('../Pictures/5_Function.png', dpi=200)
    if input('SHOW? [Y/n]') == 'Y':
        plt.show()
