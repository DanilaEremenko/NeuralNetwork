import matplotlib.pyplot as plt
import matplotlib.patches as patch

import numpy as np
import TfWithKeras.IMAGE_CHANGER as ich
from keras.preprocessing import image


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


def addCircle(x, y, color):
    r = 0.1
    plt.gca().add_patch(patch.Circle((x + (0.25 - r * 1.3), y + (0.25 - r * 1.3)), radius=r, color=color))
    plt.gca().add_patch(patch.Circle((x + (0.25 - r * 1.3), y + (0.25 - r * 1.3)), radius=r * 0.9, color='#FFFFFF'))


def addCross(x, y, color):
    # L
    plt.gca().add_patch(
        patch.Rectangle((x + 0.25 * 0.7, y + 0.25 * 0.2), color=color, width=0.03, height=0.175, angle=45))
    # R
    plt.gca().add_patch(
        patch.Rectangle((x + 0.25 * 0.3, y + 0.25 * 0.2), color=color, width=0.175, height=0.03, angle=45))


def plot_field(elements, path, save=False, show=True):
    xSize = 4
    ySize = 4
    for x in range(0, xSize):
        for y in range(0, ySize):
            if elements[y][x] == 1:
                addCross(x=x / 4.0, y=(ySize - y - 1) / 4.0, color="#000000")
            else:
                addCircle(x=x / 4.0, y=(ySize - y - 1) / 4.0, color="#000000")

    for i in np.arange(0.25, 0.76, 0.25, dtype=float):
        plt.gca().add_patch(patch.Rectangle((i, 0), color="#000000", width=0.0001, height=1))
        plt.gca().add_patch(patch.Rectangle((0, i), color="#000000", width=1, height=0.0001))

    if save:
        plt.savefig(path, dpi=200)
    if show:
        plt.show()
    plt.close()
    return


def load_data(ex_num, path_for_images, images_size=(64, 64)):
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

        path = str(path_for_images) + "/cz_" + str(i) + ".png"

        # cut main part of image and change resolution
        plot_field(elements, path, save=True, show=False)
        ich.cut_image(path, path, area=(158, 112, 1154, 857))
        ich.resize_image(path, path, size=images_size)

        # convert to numpy array
        img = image.load_img(path, grayscale=True)
        pxs = image.img_to_array(img)

        # normalization
        # pxs = 255 - pxs
        # pxs /= 255.0
        # pxs = np.expand_dims(pxs, axis=0)
        pxs.shape = (1, images_size[0] * images_size[1])

        elements.shape = (16)

        y_train = np.append(y_train, elements)
        x_train = np.append(x_train, pxs)
    x_train.shape = (ex_num, 4096)
    y_train.shape = (ex_num,16)
    return (x_train, y_train)
    # return (x_train, y_train), (x_test, y_test)
    # return elements


# if __name__ == '__main__':
#     (x_train, y_train) = load_data(10, "A_CZ", images_size=(64, 64))
