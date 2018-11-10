import matplotlib.pyplot as plt
import matplotlib.patches as patch

import numpy as np


def matrixIsAcceptable(elements, maxX, maxY):
    print(elements)
    zeroAm = 0
    oneAm = 0
    # Проверка по горизонтали
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

    # Проверка по вертикали
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
                plt.gca().add_patch(
                    patch.Rectangle((x / 4, (ySize - y - 1) / 4), 0.25, 0.25, color='#000000', fill=True))
            else:
                plt.gca().add_patch(
                    patch.Rectangle((x / 4, (ySize - y - 1) / 4), 0.25, 0.25, color='#FFFFFF', fill=True))

    if input('SAVE? [Y/n]') == 'Y':
        plt.savefig("../Pictures/1_CrossZero.png", dpi=200)
    if input('SHOW? [Y/n]') == 'Y':
        plt.show()
    plt.close()
