from PIL import Image
from keras.preprocessing import image as kimage
import numpy as np


def resize_image(input_path, output_path, size):
    original_image = Image.open(input_path)

    resized_image = original_image.resize(size)

    resized_image.save(output_path)


def cut_image(input_path, output_path, area):
    original_image = Image.open(input_path)

    original_image = original_image.crop(area)

    original_image.save(output_path)


def plot_examples_field(elements, size, cr_path, cir_path, path, save=False, show=True,
                        img_size=(128, 128)):
    pxs_cross = kimage.img_to_array(kimage.load_img(cr_path, grayscale=True)).reshape(32 * 32)
    pxs_circle = kimage.img_to_array(kimage.load_img(cir_path, grayscale=True)).reshape(32 * 32)

    pxs_filed = np.empty(0)
    for x in range(0, size[0]):
        for y in range(0, size[1]):
            if elements[y][x] == 1:
                pxs_filed = np.append(pxs_filed, pxs_cross)
            else:
                pxs_filed = np.append(pxs_filed, pxs_circle)
    img_field = Image.fromarray(pxs_filed.reshape(img_size)).convert('L')
    if save:
        img_field.save(path)
    if show:
        img_field.show()
