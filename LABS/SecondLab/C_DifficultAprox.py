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
    batch_size = 20
    epochs = 10000
    lr = 0.001
    verbose = 1
    first_layer = 50

    opt_type = 2  # 0-SGD, 1 - SGD + Nesterov, 2 - Adam,3-Adam+Amsgard, 4 - Adadelta
    opt_name = "None"
    optimizer = SGD(lr=lr)

    goal_loss = 0.0005

    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=train_size, show=False)

    k_init='he_uniform'

    model = Sequential()

    model.add(
        Dense(first_layer, input_dim=2, kernel_initializer=k_init, bias_initializer=k_init,
              activation='sigmoid'))

    model.add(Dense(1, kernel_initializer='he_uniform', bias_initializer='he_uniform', activation='linear'))

    # 3 setting stopper
    # callbacks.EarlyStopping(monitor='acc', min_delta=0, patience=5, mode='max')
    callbacks = [EarlyStoppingByLossVal(monitor='val_loss', value=goal_loss, verbose=1)]

    if opt_type == 0:
        optimizer = SGD(lr=lr)
        opt_name = "SGD"
    elif opt_type == 1:
        optimizer = SGD(lr=lr, nesterov=True)
        opt_name = "SGD+Nesterov"
    elif opt_type == 2:
        optimizer = Adam(lr=lr)
        opt_name = "Adam"
    elif opt_type == 3:
        optimizer = Adam(lr=lr, amsgrad=True)
        opt_name = "Adam+Amsgard"
    elif opt_type == 4:
        optimizer = Adadelta()
        opt_name = "Adadelta"
    else:
        Exception("Unexpected opt_type value")

    # 4 model fitting
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                        verbose=verbose, callbacks=callbacks, validation_data=(x_test, y_test))

    # Save information about learning and save NN
    dir_name = "C_" + opt_name + "_" + str(history.epoch.__len__()) + "_" + str(lr)+"_"+str(k_init)

    os.mkdir(dir_name)

    gr.plot_graphic(x=history.epoch, y=np.array(history.history["val_loss"]), x_label='epochs', y_label='val_loss',
                    title="val_loss" + ' history', save_path=dir_name + "/" + "val_loss.png", save=True, show=True)

    plt.plot(np.transpose(x_test)[0], y_test, '.-')
    plt.plot(np.transpose(x_test)[0], model.predict(x_test), '.-')
    plt.legend(('function', 'approximation'), loc='lower left', shadow=True)
    plt.title('aproximation comparison\nlr = %.3f\nval_loss = %.4f\n neurons = %.d' % (
        lr, history.history["val_loss"][history.epoch.__len__() - 1], first_layer))

    plt.savefig(dir_name + "/" + "compare.png", dpi=200)
    plt.show()
    plt.close()

    model.save(dir_name + "/" + dir_name + '.h5')
