import matplotlib.pyplot as plt
import math as m


def function(x):
    return x / 10 % 4 + m.sin(x) * m.fabs(maxX - x) / maxX


if __name__ == '__main__':
    maxX = 80
    xArray = []
    yArray = []

    for x in range(0, maxX):
        xArray.append(x)
        yArray.append(function(x))
        x += 1

    plt.plot(xArray, yArray)
    plt.plot(xArray, yArray, color='mediumvioletred')
    plt.ylim(-2, 10)

    # SHOW = 'Y'
    # SAVE = 'N'
    SHOW = input('SHOW? [Y/n]')
    SAVE = input('SAVE?[Y/n]')
    if SAVE == 'Y':
        plt.savefig('../Pictures/5_Function.png', dpi=200)
    if SHOW == 'Y':
        plt.show()
