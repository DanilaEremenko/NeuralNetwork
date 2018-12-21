import keras.initializers as wi
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam, SGD, Adadelta, Nadam

from LABS.ZeroLab import C_DivIntoTwoClasses as dataset3
import ADDITIONAL.GUI_REPORTER as gr
from ADDITIONAL.CUSTOM_KERAS import EarlyStoppingByLossVal

from ADDITIONAL.CUSTOM_KERAS import custom_fit

import os

if __name__ == '__main__':
    np.random.seed(42)
    # 1,2 initializing
    train_size = 16000
    batch_size = 160
    epochs = 500
    lr = 0.005
    verbose = 1
    first_layer = 10
    second_layer = 5

    opt_name = "Nadam"
    optimizer = Nadam(lr=lr)

    goal_loss = 0.05

    (x_train, y_train), (x_test, y_test) = dataset3.load_data(train_size=train_size, show=True)

    model = Sequential()

    model.add(
        Dense(first_layer, input_dim=2, kernel_initializer='he_uniform', bias_initializer='he_uniform',
              activation='relu'))

    model.add(
        Dense(second_layer, kernel_initializer='he_uniform', bias_initializer='he_uniform', activation='linear'))

    model.add(Dense(1, kernel_initializer='he_uniform', bias_initializer='he_uniform', activation='sigmoid'))

    # plot_model(model, to_file="C_Model.png", show_shapes=True, show_layer_names=True)

    # 3 setting stopper
    # callbacks.EarlyStopping(monitor='acc', min_delta=0, patience=5, mode='max')
    callbacks = [EarlyStoppingByLossVal(monitor='val_loss', value=goal_loss, verbose=1)]

    # 4 model fitting
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_crossentropy'])

    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                        verbose=verbose, callbacks=callbacks, validation_data=(x_test, y_test))

    # Save information about learning and save NN
    dir_name = "D_" + opt_name + "_" + str(history.epoch.__len__()) + "_" + str(lr)

    os.mkdir(dir_name)

    gr.plot_graphic(x=history.epoch, y=np.array(history.history["val_loss"]), x_label='epochs', y_label='val_loss',
                    title="val_loss" + ' history', save_path=dir_name + "/" + "val_loss.png", save=True, show=True)

    plt_x_zero = np.empty(0)
    plt_y_zero = np.empty(0)

    plt_x_one = np.empty(0)
    plt_y_one = np.empty(0)

    for i in x_test:
            if model.predict(np.array([i[0], i[1]]).reshape(1, 2)) < 0.5:
                plt_x_zero = np.append(plt_x_zero, i[0])
                plt_y_zero = np.append(plt_y_zero, i[1])
            elif model.predict(np.array([i[0], i[1]]).reshape(1, 2)) >= 0.5:
                plt_x_one = np.append(plt_x_one, i[0])
                plt_y_one = np.append(plt_y_one, i[1])

    plt.plot(plt_x_zero, plt_y_zero, '.')
    plt.plot(plt_x_one, plt_y_one, '.')
    plt.xlim(0, 1.3)
    plt.ylim(0, 1)

    plt.title('aproximation\nlr = %.3f\nval_loss = %.4f\n neurons = %.d %.d' % (
        lr, history.history["val_loss"][history.epoch.__len__() - 1], first_layer, second_layer))

    plt.savefig(dir_name + "/" + "compare.png", dpi=200)
    plt.show()
    plt.close()

    model.save(dir_name + "/" + dir_name + '.h5')
