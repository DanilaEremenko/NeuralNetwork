import numpy as np
import LABS.ZeroLab.C_DivIntoTwoClasses as dataset3

from neupy import algorithms

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataset3.load_data(train_size=12000, show=True)

    pnn = algorithms.PNN(std=10, verbose=False)

    pnn.train(x_train, y_train)

    y_predicted = pnn.predict(x_test)

