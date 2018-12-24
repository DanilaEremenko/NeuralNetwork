import numpy as np
import LABS.ZeroLab.F_ImagesGenerator as dataset8

from neupy import algorithms

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataset8.load_data(mode=0)

    pnn = algorithms.PNN(std=10, verbose=False)

    pnn.train(x_train, y_train)

    y_predicted = pnn.predict(x_test)
