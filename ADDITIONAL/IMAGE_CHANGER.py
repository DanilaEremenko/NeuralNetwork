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


def plot_examples_field(data, data_size, x_pictures, y_types, images_sizes, img_path, save=False,
                        show=True, ):
    x_pics = np.empty(0)

    for path in x_pictures.reshape(x_pictures.size, 1):
        x_pics = np.append(x_pics, get_pxs(path[0]))

    # TOOD
    x_pics.shape = (x_pictures.shape[0], x_pictures.shape[1], images_sizes[0] * images_sizes[1])

    x_test_pxs = np.empty(0)
    test_size = int(data_size[0] * data_size[1] * 0.25)

    x_train_pxs = np.empty(0)
    train_size = int(data_size[0] * data_size[1] - test_size)

    y_train = np.zeros(train_size)

    y_test = np.zeros(test_size,dtype=int)

    i = 0
    tr_i = 0
    te_i = 0
    for x in range(0, data_size[0]):
        for y in range(0, data_size[1]):
            type_number = 0
            for y_type in y_types:
                if data[x][y] == y_type:
                    if i % 4 != 0:
                        x_train_pxs = \
                            np.append(x_train_pxs, deform_image(
                                x_pics[type_number][r.randint(0, x_pictures.size / y_types.size)].reshape(images_sizes),
                                shape=1024,
                                k=r.uniform(-0.3, 0.3), n=r.randint(0, 32),
                                m=r.randint(0, 32)))
                        y_train[tr_i] = y_type
                        tr_i += 1
                    else:
                        x_test_pxs = \
                            np.append(x_test_pxs, deform_image(
                                x_pics[type_number][r.randint(0, x_pictures.size / y_types.size)].reshape(images_sizes),
                                shape=1024,
                                k=r.uniform(-0.3, 0.3), n=r.randint(0, 32),
                                m=r.randint(0, 32)))
                        y_test[te_i] = y_type
                        te_i += 1
                type_number += 1
            i += 1

    x_test_pxs.shape = (images_sizes[0] * test_size, images_sizes[0])
    img_test = Image.fromarray(x_test_pxs.transpose()).convert('L')

    x_train_pxs.shape = (images_sizes[0] * train_size, images_sizes[0])
    img_train = Image.fromarray(x_train_pxs.transpose()).convert('L')

    if save:
        img_train.save(str(img_path) + "_train.png")
        img_test.save(str(img_path) + "_test.png")
    if show:
        img_train.show()
        img_test.show()

    return (x_train_pxs, y_train), (x_test_pxs, y_test)


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

def show_image(input_path):
    image = Image.open(input_path)
    image.show()
    
    
def show_image_by_pxs(pxs):
    image = Image.fromarray(pxs).convert('L')
    image.show()

def save_image_by_pxs(pxs,output_path):
    image = Image.fromarray(pxs).convert('L')
    image.save(output_path)
    
    