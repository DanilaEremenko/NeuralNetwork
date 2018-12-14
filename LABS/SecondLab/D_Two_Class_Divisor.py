import keras.initializers as wi
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense,Activation
from keras.models import Sequential
from keras.optimizers import Adam, SGD,Adadelta

from LABS.ZeroLab import C_DivIntoTwoClasses as dataset3
import ADDITIONAL.GUI_REPORTER as gr
from ADDITIONAL.CUSTOM_KERAS import EarlyStoppingByLossVal

if __name__ == '__main__':
    np.random.seed(42)
    # 1,2 initializing
    train_size = 10000
    batch_size = 20
    epochs = 1000
    lr = 0.01
    verbose = 1
    first_layer = 200
    second_layer = 50

    opt_type = 3  # 0-SGD, 1 - SGD + Nesterov, 2 - Adam, 3 - Adadelta
    opt_name = "None"
    optimizer = SGD(lr=lr)

    goal_loss = 0.1

    (x_train, y_train), (x_test, y_test) = dataset3.load_data(train_size=train_size, show=False)

    model = Sequential()

    model.add(
        Dense(first_layer, input_dim=2, kernel_initializer='he_uniform', bias_initializer='he_uniform',
              activation='sigmoid'))

    model.add(
        Dense(second_layer, kernel_initializer='he_uniform', bias_initializer='he_uniform', activation='sigmoid'))

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
                        verbose=verbose)

    gr.plot_graphic(x=history.epoch, y=np.array(history.history['loss']),
                    x_label='epochs', y_label='loss', title='mean_squared_error history', show=True)

    gr.plot_graphic(x=history.epoch, y=np.array(history.history['binary_accuracy']),
                    x_label='epochs', y_label='binary_accuracy', title='binary_accuracy history', show=True)

    plt_x_zero = np.empty(0)
    plt_y_zero = np.empty(0)

    plt_x_one = np.empty(0)
    plt_y_one = np.empty(0)

    for x in np.arange(0.0, 1.0, step=0.01, dtype=float):
        for y in np.arange(0.0, 1.0, step=0.01, dtype=float):
            if model.predict(np.array([x, y]).reshape(1, 2)) == 0:
                plt_x_zero = np.append(plt_x_zero, x)
                plt_y_zero = np.append(plt_y_zero, y)
            elif model.predict(np.array([x, y]).reshape(1, 2)) == 1:
                plt_x_one = np.append(plt_x_one, x)
                plt_y_one = np.append(plt_y_one, y)

    plt.plot(plt_x_zero, plt_y_zero, '.')
    plt.plot(plt_x_one, plt_y_one, '.')

    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                        verbose=verbose, callbacks=callbacks, validation_data=(x_test, y_test))

    # Save information about learning and save NN
    dir_name = "D_" + opt_name + "_" + str(history.epoch.__len__()) + "_" + str(lr)

    os.mkdir(dir_name)

    gr.plot_graphic(x=history.epoch, y=np.array(history.history["val_loss"]), x_label='epochs', y_label='val_loss',
                    title="val_loss" + ' history', save_path=dir_name + "/" + "val_loss.png", save=True, show=True)

    lt.title('aproximation\nlr = %.3f\nval_loss = %.4f\n neurons = %.d %.d' % (
        lr, history.history["val_loss"][history.epoch.__len__() - 1], first_layer, second_layer))

    plt.savefig(dir_name + "/" + "compare.png", dpi=200)
    plt.show()
    plt.close()

    model.save(dir_name + "/" + dir_name + '.h5')
