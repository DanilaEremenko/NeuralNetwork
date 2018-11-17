import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

import ADDITIONAL.GUI_REPORTER as gr

from LABS.ZeroLab.Programms import A_CrossZero as dataset

if __name__ == '__main__':
    np.random.seed(42)
    in_image_size = (32, 32)
    out_image_size = (512, 32)
    ex_num = 10
    neur_number = 800
    dir_address = "A_CZ"

    # CREATE FORMATTED DIRECTORY
    # x_pictures = np.array(["Circle.png", "Cross.png"])
    # y_types = np.array([0, 1])
    # dataset.load_data_to_dir(ex_num, dir_address, images_size=in_image_size, x_pictures=x_pictures, y_types=y_types)

    # LOAD DATA FROM FORMATTED DIRECTORY
    (x_train, y_train) = dataset.load_data_from_dir(dir_address, "x_train.txt", "y_train.txt")

    x_train = x_train / 255.0

    model = Sequential()

    # 800 neurons with 784 input,initialize - normal distribution
    model.add(Dense(neur_number, input_dim=in_image_size[0] * in_image_size[1], init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='hard_sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.0008), metrics=['accuracy'])

    # batch_size define speed of studying
    history = model.fit(x_train, y_train, batch_size=1, nb_epoch=5, verbose=1)

    # score = model.evaluate(x_test, y_test, verbose=1)
    # print "accuracy on testin data %.f%%" % (score[1] * 100)

    gr.plot_history_separte(history, save_path_acc="ACC.png", save_path_loss="LOSS.png", save=True, show=False)

    # model.save('MNIST_MODEL_200.h5')
