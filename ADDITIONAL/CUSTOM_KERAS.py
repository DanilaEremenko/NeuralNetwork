import keras.backend as K
from keras.callbacks import Callback
import numpy as np
import os
import matplotlib.pyplot as plt
import ADDITIONAL.GUI_REPORTER as gr


def hard_lim(x):
    return K.cast(K.greater_equal(x, 0), K.floatx())


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            print("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 10
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

def custom_fit(model, callbacks, x_train, y_train, x_test, y_test, epochs, batch_size,
               dir_name, compare_title, draw_step=10, verbose=1):
    np.random.seed(42)

    epochs_step = int(epochs / draw_step)

    if dir_name != None:
        os.mkdir(dir_name)

    full_loss_history = np.empty(0)

    for init_epoch in np.arange(0, epochs, step=epochs_step):
        history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=init_epoch + epochs_step,
                            verbose=verbose, callbacks=callbacks, validation_data=(x_test, y_test),
                            initial_epoch=init_epoch)

        full_loss_history = np.append(full_loss_history, history.history['val_loss'])

        plt.plot(np.transpose(x_test)[0], y_test, '.')
        plt.plot(np.transpose(x_test)[0], model.predict(x_test), '.')
        plt.legend(('function', 'approximation'), loc='upper left', shadow=True)
        plt.title(
            compare_title + "\nval_loss = %.4f\nepoch = %d" % (history.history["val_loss"][history.epoch.__len__() - 1],
                                                               init_epoch + epochs_step))

        if dir_name != None:
            plt.savefig(dir_name + "/" + "%.d_compare_%.4f.png" %
                        (init_epoch + epochs_step, history.history["val_loss"][history.epoch.__len__() - 1])
                        , dpi=200)

        save = False if dir_name == None else True

        if save:
            save_path = dir_name + "/" + "val_loss.png"
        else:
            save_path = None

        plt.show()
        plt.close()

    gr.plot_graphic(x=np.arange(1, epochs + 1), y=full_loss_history,
                    x_label='epochs', y_label='val_loss',
                    title="val_loss" + ' history', save_path=save_path, save=save, show=True)

    return model
