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

    train_size = 2000
    batch_size = 20
    epochs = 100
    lr = 0.1
    goal_loss = 0.0005

    neurons_number = [300]

    opt_type = 2
    opt_name = "None"
    optimizer = SGD(lr=lr)

    draw_step = 10
    verbose = 1

    # 2 model and data initializing---------------------------------------------------------
    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=train_size, show=False)

    model = Sequential()

    model.add(
        Dense(neurons_number[0], input_dim=2, kernel_initializer='he_uniform', bias_initializer='he_uniform',
              activation='sigmoid'))

    model.add(Dense(1, kernel_initializer='he_uniform', bias_initializer='he_uniform', activation='linear'))

    # 3 setting stopper---------------------------------------------------------
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
        optimizer = RMSprop(lr=lr)
        opt_name = "RMSprop"
    elif opt_type == 5:
        optimizer = Adadelta()
        opt_name = "Adadelta"
    else:
        Exception("Unexpected opt_type value")

    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

    # 4 model fitting---------------------------------------------------------

    dir_name = "C_1_History" + opt_name + "_%.d_%.d" % (lr, neurons_number[0])

    compare_title = 'aproximation comparison\nlr = %.3f\n neurons = %.d' % (lr, neurons_number[0])

    model = custom_fit(model=model, callbacks=callbacks, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                       epochs=epochs, batch_size=batch_size,
                       dir_name=dir_name, compare_title=compare_title,
                       draw_step=draw_step, verbose=verbose)

    # 5 model saving---------------------------------------------------------
    model.save(dir_name + "/" + dir_name + '.h5')
