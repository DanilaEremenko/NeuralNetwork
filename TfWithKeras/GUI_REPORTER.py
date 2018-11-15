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
