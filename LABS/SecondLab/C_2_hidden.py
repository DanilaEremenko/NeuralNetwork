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
    lr = 0.01
    verbose = 1
    first_layer = 60
    second_layer = 30
    third_layer = 10

    opt_type = 3  # 0-SGD, 1 - SGD + Nesterov, 2 - Adam, 3 - Adadelta
    opt_name = "None"
    optimizer = SGD(lr=lr)

    goal_loss = 0.005

    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=train_size, show=False)

    model = Sequential()

    model.add(
        Dense(first_layer, input_dim=2, kernel_initializer='he_uniform', bias_initializer='he_uniform',
              activation='sigmoid'))

    model.add(
        Dense(second_layer, kernel_initializer='he_uniform', bias_initializer='he_uniform', activation='sigmoid'))

    model.add(
        Dense(third_layer, kernel_initializer='he_uniform', bias_initializer='he_uniform', activation='sigmoid'))

    model.add(Dense(1, kernel_initializer='he_uniform', bias_initializer='he_uniform', activation='linear'))

    # plot_model(model, to_file="C_Model.png", show_shapes=True, show_layer_names=True)

    # 3 setting stopper
    # callbacks.EarlyStopping(monitor='acc', min_delta=0, patience=5, mode='max')
    callbacks = [EarlyStoppingByLossVal(monitor='val_loss', value=goal_loss, verbose=1)]

    if opt_type == 0:
        optimizer = SGD(lr)
        opt_name = "SGD"
    elif opt_type == 1:
        optimizer = SGD(lr, nesterov=True)
        opt_name = "SGD+Nesterov"
    elif opt_type == 2:
        optimizer = Adam(lr)
        opt_name = "Adam"
    elif opt_type == 3:
        optimizer = Adadelta()
        opt_name = "Adadelta"
    else:
        Exception("Unexpected opt_type value")

    # 4 model fitting
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                        verbose=verbose, callbacks=callbacks, validation_data=(x_test, y_test))

    # Save information about learning and save NN
    dir_name = "C_2" + opt_name + "_" + str(history.epoch.__len__()) + "_" + str(lr)

    os.mkdir(dir_name)

    gr.plot_graphic(x=history.epoch, y=np.array(history.history["val_loss"]), x_label='epochs', y_label='val_loss',
                    title="val_loss" + ' history', save_path=dir_name + "/" + "val_loss.png", save=True, show=True)

    plt.plot(np.transpose(x_test)[0], y_test, '.-')
    plt.plot(np.transpose(x_test)[0], model.predict(x_test), '.-')
    plt.legend(('function', 'approximation'), loc='lower left', shadow=True)
    plt.title('aproximation comparison\nlr = %.3f\nval_loss = %.4f\n neurons = %.d %.d %.d' % (
        lr, history.history["val_loss"][history.epoch.__len__() - 1], first_layer, second_layer, third_layer))

    plt.savefig(dir_name + "/" + "compare.png", dpi=200)
    plt.show()
    plt.close()

    model.save(dir_name + "/" + dir_name + '.h5')
