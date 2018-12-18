from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
import numpy as np

from LABS.ZeroLab import E_Function as dataset5
from ADDITIONAL.CUSTOM_KERAS import EarlyStoppingByLossVal, custom_fit

if __name__ == '__main__':
    np.random.seed(42)
    # 1 parameters initializing
    train_size = 2000
    batch_size = 20
    epochs = 10000
    lr = 0.001
    goal_loss = 0.0005

    first_layer = 120
    second_layer = 60

    opt_type = 2  # 0-SGD, 1 - SGD + Nesterov, 2 - Adam,3 - Adam + Amsgard,4 - RMS prop,5 - Adadelta
    opt_name = "None"
    optimizer = SGD(lr=lr)

    draw_step = 10
    verbose = 1

    # 2 model and data initializing
    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=train_size, show=False, func_type='sin')

    model = Sequential()

    model.add(
        Dense(first_layer, input_dim=2, kernel_initializer='he_uniform', bias_initializer='he_uniform',
              activation='sigmoid'))

    model.add(
        Dense(second_layer, kernel_initializer='he_uniform', bias_initializer='he_uniform', activation='sigmoid'))

    model.add(Dense(1, kernel_initializer='he_uniform', bias_initializer='he_uniform', activation='linear'))

    # 3 setting stopper
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

    # 4 model fitting

    dir_name = "C_2_History" + opt_name + "_" + str(lr) + "_" + str(
        first_layer) + "_" + str(second_layer)

    compare_title = 'aproximation comparison\nlr = %.3f\n neurons = %.d %.d' % (
        lr, first_layer, second_layer)

    model = custom_fit(model=model, callbacks=callbacks, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                       epochs=epochs, batch_size=batch_size,
                       dir_name=dir_name, compare_title=compare_title,
                       draw_step=draw_step, verbose=1)

    # 5 model saving
    model.save(dir_name + "/" + dir_name + '.h5')
