from neupy import algorithms
import LABS.ZeroLab.E_Function as dataset5
import matplotlib.pyplot as plt

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=6000, show=False)

    nw = algorithms.GRNN(std=0.001, verbose=True)


    nw.train(x_train, y_train)

    y_pred = nw.predict(x_test)

    plt.plot(x_test, y_test, 'b.', label='real')
    plt.plot(x_test, y_pred, 'r.', label='fit')
    plt.legend(loc='upper left')
    plt.title('GRNN neupy')
    plt.show()