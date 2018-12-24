from neupy import algorithms
import matplotlib.pyplot as plt

import LABS.ZeroLab.E_Function as dataset5
from ADDITIONAL.IMPLEMENTATIONS.RBF.RBFN import RBFN

import numpy as np

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=6000, show=False)

    # git realization
    model = RBFN(hidden_shape=40, sigma=39.)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    plt.plot(x_test, y_test, 'b.', label='real')
    plt.plot(x_test, y_pred, 'r.', label='fit')
    plt.legend(loc='upper right')
    plt.title('RBFN from git')
    plt.show()

    # Neupy
    # nw = algorithms.RBFKMeans(n_clusters=1, verbose=True)
    #
    #
    # rbf_data = np.transpose(np.append(x_train, y_train).reshape(2,x_train.size))
    #
    # nw.train(input_train=rbf_data,
    #          epochs=50)
    #
    # y_pred = nw.predict(x_test)
    #
    # plt.plot(np.transpose(rbf_data)[0], np.transpose(rbf_data)[1], 'b.', label='real')
    # plt.plot(x_test, y_pred, 'r.', label='fit')
    # plt.legend(loc='upper left')
    # plt.title('RBFN neupy')
    # plt.show()
