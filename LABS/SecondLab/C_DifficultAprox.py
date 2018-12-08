from keras.models import Sequential
from keras import callbacks
from keras.layers import Dense
from keras.optimizers import SGD, Adam, Adadelta
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import numpy as np
import keras.initializers as wi

from LABS.ZeroLab import E_Function as dataset5
import ADDITIONAL.GUI_REPORTER as gr

if __name__ == '__main__':

    np.random.seed(42)
    # 1,2 initializing

    train_size = 2000
    batch_size = 20
    epochs = 50
    lr = 0.1
    verbose = 1

    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=train_size, show=False, func_type='sin')
    model = Sequential()

    # model.add(Dense(1, input_dim=2, activation='sigmoid', name='1_sigmoid'))
    model.add(
        Dense(1000, input_dim=2, kernel_initializer='he_uniform', bias_initializer='he_uniform', activation='sigmoid'))

    model.add(
        Dense(200, input_dim=2, kernel_initializer='he_uniform', bias_initializer='he_uniform', activation='sigmoid'))

    model.add(Dense(1, kernel_initializer='he_uniform', bias_initializer='he_uniform', activation='sigmoid'))

    # plot_model(model, to_file="C_Model.png", show_shapes=True, show_layer_names=True)

    # 3 setting stopper
    # stopper = callbacks.EarlyStopping(monitor='acc', min_delta=0, patience=5, mode='max')

    # 4 model fitting
    model.compile(optimizer=SGD(lr=lr), loss='mae', metrics=['mae'])

    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                        verbose=verbose)

    gr.plot_history_separte(history=history, save_path_acc="ACC.png", save_path_loss="LOSS.png",
                            save=False, show=True, acc='mean_absolute_error')

    plt.plot(np.transpose(x_test)[0], model.predict(x_test), '.-')
    plt.plot(np.transpose(x_test)[0], y_test, '.-')
    plt.legend(('approximation', 'function'), loc='lower left', shadow=True)
    plt.title('aproximation comparison')

    plt.show()
    plt.close()
