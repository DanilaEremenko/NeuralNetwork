from __future__ import print_function

import LABS.ZeroLab.Programms.B_LogicalFunc as dataset2
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import ADDITIONAL.GUI_REPORTER as gr

if __name__ == '__main__':
    first_layer_nur = 15
    second_layer_nur = 10
    lr = 0.45
    batch_size = 1
    epochs = 50
    verbose = 1
    (x_train, y_train), (x_test, y_test) = dataset2.load_data()

    model = Sequential()

    model.add(Dense(first_layer_nur, input_dim=x_train.shape[1], init='normal', activation='relu'))
    model.add(Dense(second_layer_nur, init='normal', activation='tanh'))
    model.add(Dense(1, init='normal', activation='hard_sigmoid'))

    model.compile(loss='mean_squared_error', optimizer=SGD(lr=lr), metrics=['accuracy'])

    # batch_size define speed of studying
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)

    score = model.evaluate(x_test, y_test, verbose=1)

    print("\naccuracy on train data\t %.f%%" % (history.history['acc'][epochs - 1] * 100))
    print("\naccuracy on testing data %.f%%" % (score[1] * 100))
    # print("loss on train data %.f%%" % (history.history['loss'][history.history] * 100))
    gr.plot_history_separte(history, save_path_acc="B_LF_Aprox/ACC.png", save_path_loss="B_LF_Aprox/LOSS.png",
                            save=True, show=False)
