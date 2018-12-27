from neupy import algorithms
import numpy as np
from LABS.ZeroLab import E_Function as dataset5

from matplotlib import pyplot as plt

if __name__ == '__main__':

    epochs = 100
    step = 0.5
    train_size=2000

    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=train_size)

    data = zip(x_train, y_train)


    for m in [5,10,15]:
        sofm = algorithms.SOFM(
            n_inputs=2,
            n_outputs=m * m,

            step=step,
            show_epoch=20,

            verbose=True,

            learning_radius=1,
            features_grid=(m, m, 1),
        )

        sofm.train(data, epochs=epochs)

        plt.plot(x_train, y_train, 'b.', label='real')
        plt.plot(sofm.weight[0], sofm.weight[1], 'r.', label='fit')

        plt.legend(loc='upper right')
        plt.title("SOFM\nm = %.d\n lr = %4.f" % (m, step))

        plt.show()
