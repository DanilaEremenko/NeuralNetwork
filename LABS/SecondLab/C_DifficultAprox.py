from keras.models import Sequential
from keras import callbacks
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

from LABS.ZeroLab import E_Function as dataset5
import ADDITIONAL.GUI_REPORTER as gr

if __name__ == '__main__':
    # 1,2 initializing
    train_size = 300
    batch_size = 10
    epochs = 100
    lr=0.01
    verbose = 1

    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=train_size, show=False)

    model = Sequential()

    model.add(Dense(1, input_dim=1, activation='sigmoid', name='1_sigmoid'))
    model.add(Dense(20, activation='sigmoid', name='2_sigmoid'))
    model.add(Dense(1, activation='linear', name='3_linear'))

    plot_model(model, to_file="C_Model.png", show_shapes=True, show_layer_names=True)

    # 3 setting stopper
    stopper = callbacks.EarlyStopping(monitor='acc', min_delta=0, patience=5, mode='max')

    # 4 model fitting
    model.compile(optimizer=SGD(lr=lr), loss='mean_squared_error', metrics=['mean_absolute_error'])

    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)

    gr.plot_history_separte(history=history, save_path_acc="ACC.png", save_path_loss="LOSS.png",
                            save=False, show=True,acc='mean_absolute_error')

    plt.plot(x_test, model.predict(x_test),'.-')
    plt.plot(x_test,y_test,'.-')
    plt.legend(('approximation', 'function'), loc='upper left', shadow=True)
    plt.title('aproximation comparison')

    plt.show()
    plt.close()
