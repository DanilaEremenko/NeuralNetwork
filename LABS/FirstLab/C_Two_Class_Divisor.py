from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import ADDITIONAL.GUI_REPORTER as gr

import LABS.ZeroLab.C_DivIntoTwoClasses as dataset3

if __name__ == '__main__':
    train_size = 4000

    first_layer_nur = 30
    second_layer_nur = 10
    third_layer_nur = 5
    lr = 0.99
    batch_size = 50
    epochs = 250
    verbose = 1

    (x_train, y_train), (x_test, y_test) = dataset3.load_data(train_size=train_size, show=False)

    model = Sequential()

    model.add(Dense(first_layer_nur, input_dim=x_train.shape[1], init='he_normal', activation='relu'))
    model.add(Dense(second_layer_nur, init='he_normal', activation='linear'))
    model.add(Dense(third_layer_nur, init='glorot_normal', activation='hard_sigmoid'))
    model.add(Dense(1, init='he_normal', activation='hard_sigmoid'))

    model.compile(loss='mean_squared_error', optimizer=SGD(lr=lr), metrics=['accuracy'])

    # batch_size define speed of studying
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)

    score = model.evaluate(x_test, y_test, verbose=1)

    print("\naccuracy on train data\t %.f%%" % (history.history['acc'][epochs - 1] * 100))
    print("\naccuracy on testing data %.f%%" % (score[1] * 100))
    # print("loss on train data %.f%%" % (history.history['loss'][history.history] * 100))
    gr.plot_history_separte(history, save_path_acc="ACC.png", save_path_loss="LOSS.png",
                            save=True, show=False)