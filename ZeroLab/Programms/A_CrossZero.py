import matplotlib.pyplot as plt
import matplotlib.patches as patch
from math import pi

import numpy as np


def matrixIsAcceptable(elements, maxX, maxY):
    print(elements)
    zeroAm = 0
    oneAm = 0
    # Check horizontal
    for y in range(0, maxY):
        checkedEl = elements[y][0]
        isDivised = True
        for x in range(0, maxX):
            if elements[y][x] == 0:
                zeroAm += 1
            elif elements[y][x] == 1:
                oneAm += 1

            if (elements[y][x] != checkedEl):
                isDivised = False

        if isDivised:
            print('is divesed')
            return False

    print('zeroAm = ', zeroAm)
    print('oneAm = ', oneAm)

    if not zeroAm == oneAm == 8:
        print('is not equal')
        return False

    # check vertical
    for x in range(0, maxX):
        checkedEl = elements[0][x]
        isDivised = True
        for y in range(1, maxY):
            if (elements[y][x] != checkedEl):
                isDivised = False
        if isDivised:
            print('is divised')
            return False

    return not isDivised


def addCircle(x, y, color):
    r=0.1
    plt.gca().add_patch(patch.Circle((x + (0.25-r), y + (0.25-r)), radius=r, color=color))
    plt.gca().add_patch(patch.Circle((x + (0.25-r), y + (0.25-r)), radius=r * 0.9, color='#FFFFFF'))


def addCross(x, y, color):
    # L
    plt.gca().add_patch(patch.Rectangle((x + 0.25 * 0.7, y+0.25*0.1), color=color, width=0.03, height=0.175, angle=45))
    # R
    plt.gca().add_patch(patch.Rectangle((x + 0.25 * 0.3, y+0.25*0.1), color=color, width=0.175, height=0.03, angle=45))


if __name__ == '__main__':
    minEl = 0
    maxEl = 1

    xSize = 4
    ySize = 4

    elements = np.random.randint(minEl, maxEl + 1, size=(ySize, xSize))
    while not matrixIsAcceptable(elements, xSize, ySize):
        elements = np.random.randint(minEl, maxEl + 1, size=(ySize, xSize))
        print('----------------------------------')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    for x in range(0, xSize):
        for y in range(0, ySize):
            if elements[y][x] == 1:
                addCross(x=x / 4.0, y=(ySize - y - 1) / 4.0, color="#000000")
            else:
                addCircle(x=x / 4.0, y=(ySize - y - 1) / 4.0, color="#000000")

    plt.show()
    plt.close()

    # if input('SAVE? [Y/n]') == 'Y':
    #     plt.savefig("../Pictures/1_CrossZero.png", dpi=200)
    # if input('SHOW? [Y/n]') == 'Y':
    #     plt.show()
