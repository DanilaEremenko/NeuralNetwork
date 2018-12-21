import LABS.ZeroLab.D_DivIntoNClasses as dataset4

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam, SGD, Adadelta

import ADDITIONAL.GUI_REPORTER as gr
from ADDITIONAL.CUSTOM_KERAS import EarlyStoppingByLossVal

if __name__ == '__main__':
    np.random.seed(42)
    # 1,2 initializing
    train_size = 16000
    batch_size = 100
    epochs = 1000
    lr = 0.007
    verbose = 1
    neurons_number = [25, 7]

    opt_name = "Adam"
    optimizer = Adam(lr=lr)

    goal_loss = 0.013

    (x_train, y_train), (x_test, y_test) = dataset4.load_data(train_size=train_size, show=True)

    model = Sequential()

    model.add(Dense(neurons_number[0], input_dim=2, activation='relu'))

    model.add(Dense(neurons_number[1], activation='sigmoid'))

    # 3 setting stopper
    callbacks = [EarlyStoppingByLossVal(monitor='val_loss', value=goal_loss, verbose=1)]

    # 4 model fitting
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                        verbose=verbose, callbacks=callbacks, validation_data=(x_test, y_test))

    gr.plot_graphic(x=history.epoch, y=np.array(history.history["val_loss"]), x_label='epochs', y_label='val_loss',
                    title="val_loss" + ' history', save=False, show=True)

    plt_x_zero = np.empty(0)
    plt_y_zero = np.empty(0)

    plt_x_one = np.empty(0)
    plt_y_one = np.empty(0)

    plt_x_two = np.empty(0)
    plt_y_two = np.empty(0)

    plt_x_three = np.empty(0)
    plt_y_three = np.empty(0)

    plt_x_four = np.empty(0)
    plt_y_four = np.empty(0)

    plt_x_five = np.empty(0)
    plt_y_five = np.empty(0)

    plt_x_six = np.empty(0)
    plt_y_six = np.empty(0)

    for i in x_test:
        max = model.predict(np.array([i[0], i[1]]).reshape(1, 2))[0][0]
        max_index = 0
        for j in np.arange(1, 7, step=1, dtype=int):
            if model.predict(np.array([i[0], i[1]]).reshape(1, 2))[0][j] > max:
                max = model.predict(np.array([i[0], i[1]]).reshape(1, 2))[0][j]
                max_index = j
        if max_index == 6:
            plt_x_zero = np.append(plt_x_zero, i[0])
            plt_y_zero = np.append(plt_y_zero, i[1])
        elif max_index == 0:
            plt_x_one = np.append(plt_x_one, i[0])
            plt_y_one = np.append(plt_y_one, i[1])
        elif max_index == 1:
            plt_x_two = np.append(plt_x_two, i[0])
            plt_y_two = np.append(plt_y_two, i[1])
        elif max_index == 2:
            plt_x_three = np.append(plt_x_three, i[0])
            plt_y_three = np.append(plt_y_three, i[1])
        elif max_index == 3:
            plt_x_four = np.append(plt_x_four, i[0])
            plt_y_four = np.append(plt_y_four, i[1])
        elif max_index == 4:
            plt_x_five = np.append(plt_x_five, i[0])
            plt_y_five = np.append(plt_y_five, i[1])
        elif max_index == 5:
            plt_x_six = np.append(plt_x_six, i[0])
            plt_y_six = np.append(plt_y_six, i[1])

    plt.plot(plt_x_zero, plt_y_zero, '.')
    plt.plot(plt_x_one, plt_y_one, '.')
    plt.plot(plt_x_two, plt_y_two, '.')
    plt.plot(plt_x_three, plt_y_three, '.')
    plt.plot(plt_x_four, plt_y_four, '.')
    plt.plot(plt_x_five, plt_y_five, '.')
    plt.plot(plt_x_six, plt_y_six, '.')

    plt.xlim(0, 1.5)
    plt.ylim(0, 1)

    plt.legend(('0000001 class', '1000000 class', '0100000 class', '0010000 class', '0001000 class', '0000100 class',
                '0000010 class'), loc='upper right', shadow=True)

    plt.title('classification\nlr = %.3f\nval_loss = %.4f\n neurons = %.d %.d' % (
        lr, history.history["val_loss"][history.epoch.__len__() - 1], neurons_number[0], neurons_number[1]))

    plt.show()
    plt.close()

