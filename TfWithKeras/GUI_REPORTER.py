import matplotlib.pyplot as plt
import numpy as np


def plot_history(history, save_path, save=False, show=True):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['loss']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['acc']),
             label='Accuracy')
    plt.legend()

    if show:
        plt.show()
    if save:
        plt.savefig(save_path, dpi=200)


def plot_history_separte(history, save_path_acc, save_path_loss, save=False, show=True):

    plot_graphic(x=history.epoch, y=np.array(history.history['loss']), label='Train Loss', save_path=save_path_loss,
                 save=save, show=show)
    plot_graphic(x=history.epoch, y=np.array(history.history['acc']), label='Train Acc', save_path=save_path_acc,
                 save=save, show=show)


def plot_graphic(x, y, label, save_path, save, show):
    plt.plot(x, y, label=label)

    plt.legend()

    if show:
        plt.show()
    if save:
        plt.savefig(save_path, dpi=200)
    plt.close()
