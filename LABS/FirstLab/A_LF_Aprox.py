from __future__ import print_function

import LABS.ZeroLab.Programms.A_CrossZero as dataset1
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import ADDITIONAL.GUI_REPORTER as gr
import numpy as np

if __name__ == '__main__':
    first_layer_nur = 10
    second_layer_nur = 9
    lr = 0.0675
    batch_size = 1
    epochs = 100
    verbose = 1

    y_train = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 0, 1]])

    (x_train, y_train) = dataset1.load_data(y_train, show=False)

    model = Sequential()

    model.add(Dense(first_layer_nur, input_dim=x_train.shape[1], init='he_normal', activation='relu'))
    model.add(Dense(second_layer_nur, init='glorot_normal', activation='linear'))
    model.add(Dense(1, init='he_normal', activation='hard_sigmoid'))

    model.compile(loss='mean_squared_error', optimizer=SGD(lr=lr), metrics=['accuracy'])

    # batch_size define speed of studying
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)

    print("\naccuracy on train data\t %.f%%" % (history.history['acc'][epochs - 1] * 100))
    gr.plot_history_separte(history, save_path_acc="ACC.png", save_path_loss="LOSS.png",
                            save=True, show=False)
