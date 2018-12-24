import LABS.ZeroLab.E_Function as dataset5

from neupy import algorithms
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=6000, show=False)
    for std in [0.1, 0.01, 0.001]:
        nw = algorithms.GRNN(std=std, verbose=True)

        nw.train(x_train, y_train)

        y_pred = nw.predict(x_test)

        mae = (np.abs(y_test - y_pred)).mean()

        plt.plot(x_test, y_test, 'b.', label='real')
        plt.plot(x_test, y_pred, 'r.', label='fit')
        plt.legend(loc='upper right')
        plt.title('GRNN aprox with neupy\nstd = %.4f' % (std))
        plt.show()
