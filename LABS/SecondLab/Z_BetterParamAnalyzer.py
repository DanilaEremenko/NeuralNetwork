from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, Adadelta
import matplotlib.pyplot as plt
import numpy as np

from LABS.ZeroLab import E_Function as dataset5
import ADDITIONAL.GUI_REPORTER as gr
from ADDITIONAL.CUSTOM_KERAS import EarlyStoppingByLossVal

import os

if __name__ == '__main__':
    np.random.seed(42)
    # 1,2 initializing
    train_size = 2000
    batch_size = 10
    epochs = 3000
    verbose = 0

    lrs = [0.001, 0.0001, 0.00001, 0.000001]
    first_layers = np.array([40, 45, 50, 55, 60, 65, 70, 75, 80])

    dir_name = "Adams"
    os.mkdir(dir_name)

    i = 1
    for first_layer in first_layers:
        for lr in lrs:
            for amsgard in np.array([True, False]):
                optimizer = Adam(lr=lr, amsgrad=amsgard)

                goal_loss = 0.015

                (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=train_size, show=False)

                model = Sequential()

                model.add(
                    Dense(first_layer, input_dim=2, kernel_initializer='he_uniform', bias_initializer='he_uniform',
                          activation='sigmoid'))

                model.add(Dense(1, kernel_initializer='he_uniform', bias_initializer='he_uniform', activation='linear'))

                # 3 setting stopper
                callbacks = [EarlyStoppingByLossVal(monitor='val_loss', value=goal_loss, verbose=1)]

                # 4 model fitting
                model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

                history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                                    verbose=verbose, callbacks=callbacks, validation_data=(x_test, y_test))

                val_loss = history.history["val_loss"][history.epoch.__len__() - 1]

                gr.plot_graphic(x=history.epoch, y=np.array(history.history["val_loss"]), x_label='epochs',
                                y_label='val_loss',
                                title="val_loss" + ' history',
                                save_path=dir_name + "/" + "%.4f_val_loss_" % val_loss + "_" + str(
                                    first_layer) + "_" + str(
                                    lr) + "_" + str(amsgard) + ".png"
                                , save=True, show=False)

                plt.plot(np.transpose(x_test)[0], y_test, '.-')
                plt.plot(np.transpose(x_test)[0], model.predict(x_test), '.-')
                plt.legend(('function', 'approximation'), loc='lower left', shadow=True)
                plt.title('aproximation comparison\nlr = %.3f\nval_loss = %.4f\n neurons = %.d' % (
                    lr, history.history["val_loss"][history.epoch.__len__() - 1], first_layer))

                plt.savefig(dir_name + "/" + "%.4f_compare_" % val_loss + str(first_layer) + "_" + str(lr) + "_" + str(
                    amsgard) + ".jpg", dpi=200)
                plt.close()

                print("i = ", i)
                i += 1
