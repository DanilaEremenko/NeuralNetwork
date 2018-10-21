import matplotlib.pyplot as plt
import numpy as np


def getN(t, k, T: np.ndarray, q):
    if q == 1:
        return 1 if T[k] <= t and t < T[k + 1] else 0
    else:
        sum = 0
        recurs1 = getN(t=t, T=T, k=k, q=q - 1)
        recurs2 = getN(t=t, T=T, k=k + 1, q=q - 1)
        if T[k + q - 1] - T[k] != 0:
            sum += ((t - T[k]) / (T[k + q - 1] - T[k])) * recurs1
        if T[k + q] - T[k + 1] != 0:
            sum += ((T[k + q] - t) / (T[k + q] - T[k + 1])) * recurs2
        return sum


def getPt(t, p0, p1, p2, p3, w0, w1, w2, w3, T: np.ndarray, pNumber):
    return (w0 * p0 * getN(t=t, k=0, T=T, q=pNumber - 1) +
            w1 * p1 * getN(t=t, k=1, T=T, q=pNumber - 1) +
            w2 * p2 * getN(t=t, k=2, T=T, q=pNumber - 1) +
            w3 * p3 * getN(t=t, k=3, T=T, q=pNumber - 1)
            ) / \
           (w0 * getN(t=t, k=0, T=T, q=pNumber - 1) +
            w1 * getN(t=t, k=1, T=T, q=pNumber - 1) +
            w2 * getN(t=t, k=2, T=T, q=pNumber - 1) +
            w3 * getN(t=t, k=3, T=T, q=pNumber - 1))


if __name__ == '__main__':
    # labels
    X_SHIFT = 0.9
    Y_SHIFT = 1.1
    F_SIZE = 12

    # points
    px = np.array([0, 3, 7, 3])
    py = np.array([0, 3, 0, -4])
    w = np.array([1.0, 0.25, 2.0, 1.0])

    print("Default poligon\nx\ty")
    for x, y in zip(px, py):
        print(x, '\t', y)

    for i in range(0, px.__len__()):
        plt.text(px[i] * X_SHIFT, py[i] * Y_SHIFT, "p" + str(i), fontsize=F_SIZE)
    plt.plot(px, py, '--o')

    ptx = []
    pty = []
    print("\np(t) calculating-------------------------------------------")
    T = np.arange(2, 4.1, 0.25, dtype=float)
    for t in np.arange(2.25, 3.5, 0.25, dtype=float):
        ptx.append(getPt(t, px[0], px[1], px[2], px[3], w[0], w[1], w[2], w[3], T, 4))
        pty.append(getPt(t, py[0], py[1], py[2], py[3], w[0], w[1], w[2], w[3], T, 4))
        plt.text(ptx[ptx.__len__() - 1] * X_SHIFT, pty[pty.__len__() - 1] * Y_SHIFT, "pt_%.3f" % t, fontsize=F_SIZE)
        print("Px(%.3f) = %.3f" % (t, ptx[ptx.__len__() - 1]))
        print("Py(%.3f) = %.3f\n" % (t, pty[pty.__len__() - 1]))

    # plot-----------------------------------------------------------------------
    plt.plot(ptx, pty, '--o')

    if (input("SAVE[Y/n]") == "Y"):
        name = input("Name = ")
        plt.savefig("Pictures/"+name, dpi=200)
    if (input("SHOW[Y/n]") == "Y"):
        plt.show()

# ---------------------OLD getN-----------------------
# if m == 0:
#     return 1 if T[k] < T[k + 1] else 0
#
# elif m == 1:
#     sum = 0.0
#     if T[k + m - 1] - T[k] != 0:
#         sum = (t - T[k]) / (T[k + m - 1] - T[k]) * getN(t=t, k=k, T=T, m=0)
#     if T[k + m] - T[k + 1] != 0:
#         sum = sum + (T[k + m] - t) / (T[k + m] - T[k + 1]) * getN(t=t, k=k + 1, T=T, m=0)
#     return sum
#
# elif m == 2:
#     sum = 0.0
#     if T[k + m - 1] - T[k] != 0:
#         sum = (t - T[k]) / (T[k + m - 1] - T[k]) * getN(t=t, k=k, T=T, m=1)
#     if T[k + m] - T[k + 1] != 0:
#         sum = sum + (T[k + m] - t) / (T[k + m] - T[k + 1]) * getN(t=t, k=k + 1, T=T, m=1)
#     return sum
#
# elif m == 3:
#     sum = 0.0
#     if T[k + m - 1] - T[k] != 0:
#         sum = (t - T[k]) / (T[k + m - 1] - T[k]) * getN(t=t, k=k, T=T, m=2)
#     if T[k + m] - T[k + 1] != 0:
#         sum = sum + (T[k + m] - t) / (T[k + m] - T[k + 1]) * getN(t=t, k=k + 1, T=T, m=2)
#     return sum
