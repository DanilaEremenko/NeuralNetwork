from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, Adadelta, RMSprop

import numpy as np

from LABS.ZeroLab import E_Function as dataset5
from ADDITIONAL.CUSTOM_KERAS import EarlyStoppingByLossVal, custom_fit

if __name__ == '__main__':
    # 1 parameters initializing---------------------------------------------------------
    np.random.seed(42)

    train_size = 6000
    batch_size = 64
    epochs = 1000
    lr = 0.1
    goal_loss = 0.01
    optimizer = Adam(lr=lr)
    opt_name = "Adam"

    draw_part = 10
    verbose = 0

    # 2 model and data initializing---------------------------------------------------------

    for lr in np.array([0.1]):
        for neurons_number in np.array([48]):
            (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=train_size, show=False)

            x_train = np.transpose(np.append(x_train, np.ones(x_train.size)))
            x_train = np.transpose(np.append(x_train, np.ones(x_train.size)))
            x_train = np.transpose(np.append(x_train, np.ones(x_train.size)).reshape(8, x_train.size/4))

            x_test = np.transpose(np.append(x_test, np.ones(x_test.size)))
            x_test = np.transpose(np.append(x_test, np.ones(x_test.size)))
            x_test = np.transpose(np.append(x_test, np.ones(x_test.size)).reshape(8, x_test.size/4))

            model = Sequential()

            model.add(Dense(neurons_number, input_dim=8, activation='sigmoid'))

            model.add(Dense(1, activation='linear'))

            # 3 setting stopper---------------------------------------------------------
            callbacks = [EarlyStoppingByLossVal(monitor='val_loss', value=goal_loss, verbose=1)]

            model.compile(optimizer=optimizer, loss='mae')

            # 4 model fitting---------------------------------------------------------

            dir_name = None

            compare_title = 'aproximation comparison\nlr = %.3f\n neurons = %.d' % (lr, neurons_number)

            model = custom_fit(model=model, callbacks=callbacks, x_train=x_train, y_train=y_train, x_test=x_test,
                               y_test=y_test,
                               epochs=epochs, batch_size=batch_size,
                               dir_name=dir_name, compare_title=compare_title,
                               draw_step=draw_part, verbose=verbose)
