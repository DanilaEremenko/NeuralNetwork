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


def plot_examples_field(data, data_size, x_pictures, y_types, images_sizes, filed_path, save=False, show=True, ):
    x_pxs = np.empty(0)

    for x_path in x_pictures:
        x_pxs = np.append(x_pxs, kimage.img_to_array(kimage.load_img(x_path, color_mode="grayscale")))
    x_pxs.shape = (x_pictures.__len__(), images_sizes[0] * images_sizes[1])

    pxs_fieled = np.empty(0)
    for x in range(0, data_size[0]):
        for y in range(0, data_size[1]):
            type_number = 0
            for y_type in y_types:
                if data[x][y] == y_type:
                    pxs_fieled = np.append(pxs_fieled, x_pxs[type_number])
                type_number += 1

    pxs_fieled.shape = (images_sizes[0] * data_size[0] * data_size[1], images_sizes[0])

    img_field = Image.fromarray(pxs_fieled.transpose()).convert('L')

    if save:
        img_field.save(filed_path)
    if show:
        img_field.show()

    return pxs_fieled


def get_pxs(path):
    return kimage.img_to_array(kimage.load_img(path, color_mode="grayscale"))
