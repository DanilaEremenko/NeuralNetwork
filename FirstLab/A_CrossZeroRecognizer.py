import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD

import TfWithKeras.GUI_REPORTER as gr

from keras.preprocessing import image
import ZeroLab.Programms.A_CrossZero as dataset

if __name__ == '__main__':
    np.random.seed(42)
    image_size = (64, 64)
    ex_num = 10
    neur_number = 800
    dir_address = "A_CZ"

    # CREATE FORMATTER DURECTORY
    #dataset.load_data_to_dir(ex_num, dir_address, images_size=image_size)

    # LOAD DATA FROM FORMATTED DIRECTORY
    (x_train, y_train) = dataset.load_data_from_dir(dir_address, "x_train.txt", "y_train.txt")

    x_train.shape = (ex_num, image_size[0] * image_size[1])
    y_train.shape = (ex_num, 16)

    x_train = x_train / 255.0

    model = Sequential()

    # 800 neurons with 784 input,initialize - normal distribution
    model.add(Dense(700, input_dim=image_size[0] * image_size[1], init='normal', activation='relu'))
    model.add(Dense(16, init='normal', activation='hard_sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.0008), metrics=['accuracy'])

    # batch_size define speed of studying
    history = model.fit(x_train, y_train, batch_size=1, nb_epoch=100, verbose=1)

    # score = model.evaluate(x_test, y_test, verbose=1)
    # print "accuracy on testin data %.f%%" % (score[1] * 100)

    gr.plot_history_separte(history, save_path_acc="ACC.png", save_path_loss="LOSS.png", save=True, show=False)

    # model.save('MNIST_MODEL_200.h5')
