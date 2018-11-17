from PIL import Image
from keras.preprocessing import image as kimage
import numpy as np
from math import sqrt


def resize_image(input_path, output_path, size):
    original_image = Image.open(input_path)

    resized_image = original_image.resize(size)

    resized_image.save(output_path)


def cut_image(input_path, output_path, area):
    original_image = Image.open(input_path)

    original_image = original_image.crop(area)

    original_image.save(output_path)


def plot_examples_field(els, els_size, cr_path, cir_path, path, save=False, show=True, ):
    pxs_cross = kimage.img_to_array(kimage.load_img(cr_path, grayscale=True))
    pxs_circle = kimage.img_to_array(kimage.load_img(cir_path, grayscale=True))

    if (pxs_cross.shape[0] != pxs_cross.shape[1]) or (pxs_circle.shape[0] != pxs_circle.shape[1]):
        raise ValueError('Input images is not square')

    pxs_cross.shape = (pxs_cross.shape[0] * pxs_cross.shape[1])
    pxs_circle.shape = (pxs_circle.shape[0] * pxs_circle.shape[1])

    if pxs_circle.shape != pxs_cross.shape:
        raise ValueError('Unequal size of images')

    pxs_filed = np.empty(0)
    for x in range(0, els_size[0]):
        for y in range(0, els_size[1]):
            if els[y][x] == 1:
                pxs_filed = np.append(pxs_filed, pxs_cross)
            else:
                pxs_filed = np.append(pxs_filed, pxs_circle)

    f_size = int(sqrt(pxs_circle.shape[0]) * els_size[0])

    img_field = Image.fromarray(pxs_filed.reshape(f_size, f_size)).convert('L')
    if save:
        img_field.save(path)
    if show:
        img_field.show()
