from __future__ import print_function
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


# x(t) = sign(sin(t))

def get_sin_sig(t, freeq, ampl):
    return ampl * np.sin(t * freeq * t * 10)


def get_rect_sig(t, freeq, ampl):
    return ampl * signal.square(t * freeq * t * 10)


def plot_spectogram(t):
    """
    f   : Array of sample frequencies.
    t   : Array of segment times.
    Sxx : Spectrogram of x. By default, the last axis of Sxx corresponds
    to the segment times.

    """
    f, t, sxx = signal.spectrogram(x=t)
    print(plt.pcolormesh.__doc__)
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

    # intv = [0, 1]
    # p_num = 500
    # t = np.linspace(intv[0], intv[1], p_num, endpoint=False)

    # legends = np.empty(0)

    # for freeq, ampl in zip(freeqs, ampls):
    #     sig = get_sin_sig(t=t, freeq=freeq, ampl=ampl)
    #     plot_graphic(t, sig, title="sin_sig", show=True)
    #     plot_spectogram(t=sig)
    #
    #     sig = get_rect_sig(t=t, freeq=freeq, ampl=ampl)
    #     plot_graphic(t, sig, title="rect_sig", show=True)
    #     plot_spectogram(t=sig)
    #
    # plt.show()

    freeq = 20
    ampl = 1
    fs = 1000  # sampling rate
    ts = 1.0 / fs  # sampling interval
    n = 8192  # number of fft points, pick power of 2
    t = np.arange(0, n * ts, ts)  # time vector
    pts_num = 100

    signs = [np.cos(2 * np.pi * freeq * t), np.sign(np.cos(2 * np.pi * freeq * t))]
    titles = ['sin', 'square_sign']
    for sig, title in zip(signs, titles):
        sig_fft = np.fft.fft(sig) / n * 2  # /N to scale due to python DFT equation,
        fft_freq = np.fft.fftfreq(n, ts)  # python function to get Hz frequency axis

        plot_graphic(x=t[:pts_num], y=sig[:pts_num], title='signal' + title, x_label='time(S)',
                     y_label='amplitude (V)',
                     show=True)
        plot_graphic(x=fft_freq[:fs], y=abs(sig_fft)[:fs], title='spectr' + title, x_label='frequency (Hz)',
                     y_label='amplitude (V)',
                     show=True)
