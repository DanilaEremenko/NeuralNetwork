import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

import TfWithKeras.GUI_REPORTER as gr

from keras.preprocessing import image
import ZeroLab.Programms.A_CrossZero as dataset

if __name__ == '__main__':
    #np.random.seed(42)
    # x_train-image of numbers
    # y_train-answer of NN
    (x_train, y_train) = dataset.load_data(10, "A_CZ", images_size=(64, 64))

    x_train = x_train / 255.0

    model = Sequential()

    # 800 neurons with 784 input,initialize - normal distribution
    model.add(Dense(800, input_dim=4096, init='normal', activation='relu'))
    model.add(Dense(16, init='normal', activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

    # batch_size define speed of studying
    history = model.fit(x_train, y_train, batch_size=200, nb_epoch=20, verbose=1)

    # score = model.evaluate(x_test, y_test, verbose=1)
    # print "accuracy on testin data %.f%%" % (score[1] * 100)

    gr.plot_history_separte(history,save_path_acc="ACC.png",save_path_loss="LOSS.png",save=True,show=False)

    # model.save('MNIST_MODEL_200.h5')
