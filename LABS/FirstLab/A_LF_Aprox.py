from __future__ import print_function

import LABS.ZeroLab.Programms.A_CrossZero as dataset1
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import ADDITIONAL.GUI_REPORTER as gr
import numpy as np

if __name__ == '__main__':
    first_layer_nur = 10
    lr = 0.45
    batch_size = 1
    epochs = 5
    verbose = 1

    y_train = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 0, 1]])

    (x_train, y_train) = dataset1.load_data(y_train)

    model = Sequential()

    model.add(Dense(first_layer_nur, input_dim=x_train.shape[1], init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='hard_sigmoid'))

    model.compile(loss='mean_squared_error', optimizer=SGD(lr=lr), metrics=['accuracy'])

    # batch_size define speed of studying
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)

    gr.plot_history_separte(history, save_path_acc="ACC.png", save_path_loss="LOSS.png",
                            save=True, show=False)
