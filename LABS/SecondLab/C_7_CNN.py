from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Conv1D
from keras.optimizers import SGD, Adam, Adadelta
import matplotlib.pyplot as plt
import numpy as np

import ADDITIONAL.GUI_REPORTER as gr

from LABS.ZeroLab import E_Function as dataset5

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=2000, show=False)

    model = Sequential()
    model.add(Conv1D(2, 2, input_shape=(x_train.shape[1], 1)))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='relu'))

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(x_train, y_train, epochs=5, batch_size=10)

    # gr.plot_graphic(x=history.epoch, y=np.array(history.history["val_loss"]), x_label='epochs', y_label='val_loss',
    #                 title="val_loss" + ' history', save=False, show=True)

    # plt.plot(np.transpose(x_test)[0], y_test, '.-')
    # plt.plot(np.transpose(x_test)[0], model.predict(x_test), '.-')
    # plt.legend(('function', 'approximation'), loc='lower left', shadow=True)
    # plt.title('aproximation comparison\nlr = %.3f\nval_loss = %.4f\n neurons = %.d %.d' % (
    #     lr, history.history["val_loss"][history.epoch.__len__() - 1], first_layer, second_layer))

    # plt.savefig(dir_name + "/" + "compare.png", dpi=200)
    plt.show()
    plt.close()
