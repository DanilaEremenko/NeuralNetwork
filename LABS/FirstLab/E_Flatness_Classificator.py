import E_Flatness_Classificator_Data as dataset5
from keras import Sequential, callbacks
from keras.layers import Dense
from keras.optimizers import SGD

import ADDITIONAL.GUI_REPORTER as gr

if __name__ == '__main__':
    train_size = 4000

    first_layer_nur = 1
    lr = 0.3
    batch_size = 20
    epochs = 50
    verbose = 1

    # func_type=[lin,n_lin]
    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=train_size, show=True, k=1, b=0,
                                                              func_type='lin')

    model = Sequential()

    model.add(Dense(1, kernel_initializer='glorot_normal', activation='hard_sigmoid'))

    stopper = callbacks.EarlyStopping(monitor='acc', min_delta=0, patience=5, mode='max')

    model.compile(loss='mean_squared_error', optimizer=SGD(lr=lr), metrics=['accuracy'])

    # batch_size define speed of studying
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[stopper], verbose=verbose)

    score = model.evaluate(x_test, y_test, verbose=1)

    print("\naccuracy on train data\t %.f%%" % (history.history['acc'][stopper.stopped_epoch] * 100))
    print("\naccuracy on testing data %.f%%" % (score[1] * 100))
    print("loss on train data %.f%%" % (history.history['loss'][stopper.stopped_epoch] * 100))
    gr.plot_history_separte(history, save_path_acc="ACC.png", save_path_loss="LOSS.png",
                            save=True, show=True)
