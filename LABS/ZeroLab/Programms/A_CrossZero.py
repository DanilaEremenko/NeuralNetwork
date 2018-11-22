import matplotlib.pyplot as plt
import matplotlib.patches as patch

import numpy as np

from ADDITIONAL.IMAGE_CHANGER import plot_examples_field


def matrixIsAcceptable(elements, maxX, maxY, verbose=False):
    if verbose:
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
            if verbose:
                print('is divesed')
            return False

    if verbose:
        print('zeroAm = ', zeroAm)
        print('oneAm = ', oneAm)

    if not zeroAm == oneAm == 8:
        if verbose:
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
            if verbose:
                print('is divised')
            return False

    return not isDivised


def load_data_to_dir(ex_num, dir_address, images_size, x_pictures, y_types, x_path="x_train.txt", y_path="y_train.txt"):
    minEl = 0
    maxEl = 1
    xSize = 4
    ySize = 4

    x_train = np.empty(0)
    y_train = np.empty(0)

    for i in range(0, ex_num):
        elements = np.random.randint(minEl, maxEl + 1, size=(ySize, xSize))
        while not matrixIsAcceptable(elements, xSize, ySize):
            elements = np.random.randint(minEl, maxEl + 1, size=(ySize, xSize))

        path = str(dir_address) + "/cz_" + str(i) + ".png"

        pxs_field = plot_examples_field(data=elements, data_size=elements.shape, x_pictures=x_pictures, y_types=y_types,
                                        images_sizes=images_size, filed_path=path,
                                        save=True, show=False)

        pxs_field.shape = (1, pxs_field.shape[0] * pxs_field.shape[1])

        elements.shape = 16

        y_train = np.append(y_train, elements)
        x_train = np.append(x_train, pxs_field)

    np.savetxt(dir_address + "/" + x_path,
               x_train.reshape(ex_num * xSize * ySize, images_size[0] * images_size[1]), fmt='%d')
    np.savetxt(dir_address + "/" + y_path, y_train, fmt='%d')
    pass


def load_data_from_dir(dir, x_path="x_train.txt", y_path="y_train.txt"):
    return np.loadtxt(dir + "/" + x_path, dtype=int), np.loadtxt(dir + "/" + y_path, dtype=int)
