from PIL import Image
from keras.preprocessing import image as kimage
import numpy as np
import numpy.random as r


def resize_image(input_path, output_path, size):
    original_image = Image.open(input_path)

    resized_image = original_image.resize(size)

    resized_image.save(output_path)


def cut_image(input_path, output_path, area):
    original_image = Image.open(input_path)

    original_image = original_image.crop(area)

    original_image.save(output_path)


def plot_examples_field(data, data_size, x_pictures, y_types, images_sizes, filed_path, save=False,
                        show=True, ):
    x_pics = np.empty(0)

    for path in x_pictures.reshape(x_pictures.size, 1):
        x_pics = np.append(x_pics, get_pxs(path[0]))

    # TOOD
    x_pics.shape = (x_pictures.shape[0], x_pictures.shape[1], images_sizes[0] * images_sizes[1])

    pxs_fieled = np.empty(0)
    for x in range(0, data_size[0]):
        for y in range(0, data_size[1]):
            type_number = 0
            for y_type in y_types:
                if data[x][y] == y_type:
                    pxs_fieled = \
                        np.append(pxs_fieled, deform_image(
                            x_pics[type_number][r.randint(0, x_pictures.size / y_types.size)].reshape(images_sizes),
                            shape=1024,
                            k=r.uniform(-0.3, 0.3), n=r.randint(0, 32),
                            m=r.randint(0, 32)))
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


def deform_image(arr, shape, k, n, m):
    if n > m:
        c = n
        n = m
        m = c
    A = arr.shape[0] / 3.0
    w = 2.0 / arr.shape[1]
    shift = lambda x: A * np.sin(2.0 * np.pi * x * w)
    for i in range(n, m):
        arr[:, i] = np.roll(arr[:, i], int(shift(i) * k))
    return arr.reshape(shape)

# if __name__ == '__main__':
# arr1 = deform_image(get_pxs("Circle.png"), shape=(32,32),k=0.25, n =0, m = 32)
# arr2 = deform_image(get_pxs("Circle.png"), shape=(32,32),k=0.25, n =0, m = 16)
# arr3 =deform_image(get_pxs("Cross.png"), shape=(32,32),k=0.25, n =10, m = 29)
# img = Image.fromarray(arr1.transpose()).convert('L')
# img.show()
# img = Image.fromarray(arr2.transpose()).convert('L')
# img.show()
# img = Image.fromarray(arr3.transpose()).convert('L')
# img.show()
