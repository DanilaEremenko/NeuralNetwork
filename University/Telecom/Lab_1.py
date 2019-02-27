from __future__ import print_function
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


def get_sin_sig(t, freeq, ampl):
    return ampl * np.sin(t * freeq)


def get_rect_sig(t, freeq, ampl):
    return ampl * signal.square(freeq * t)


def plot_spectogram(t):
    """
    f   : Array of sample frequencies.
    t   : Array of segment times.
    Sxx : Spectrogram of x. By default, the last axis of Sxx corresponds
    to the segment times.

    """
    f, t, sxx = signal.spectrogram(x=t)
    # plt.pcolormesh(t, f, sxx)
    plt.plot(f, sxx)
    plt.show()


def plot_graphic(x, y, title, x_label="x", y_label="y", show=True):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(x, y)
    if show:
        plt.show()


if __name__ == '__main__':

    freeqs = [25]
    ampls = [1]
    intv = [0, 1]
    p_num = 500
    t = np.linspace(intv[0], intv[1], p_num, endpoint=False)

    legends = np.empty(0)

    for freeq, ampl in zip(freeqs, ampls):
        sig = get_sin_sig(t=t, freeq=freeq, ampl=ampl)
        plot_graphic(t, sig, title="sin_sig", show=True)
        plot_spectogram(t=sig)

        sig = get_rect_sig(t=t, freeq=freeq, ampl=ampl)
        plot_graphic(t, sig, title="rect_sig", show=True)
        plot_spectogram(t=sig)

    plt.show()
    # print("\tt\t\tts")
    # for x, y in zip(t, rec_sig):
    #     print("\t%.3f\t%.f" % (x, y))
