from neupy import algorithms
import matplotlib.pyplot as plt
import numpy as np

from LABS.ZeroLab import C_DivIntoTwoClasses as dataset3

if __name__ == '__main__':
    train_size = 6000
    epochs = 100
    step = 0.5
    (x_train, y_train), (x_test, y_test) = dataset3.load_data(train_size=train_size)

    lvqnet = algorithms.LVQ2(

        n_inputs=2,

        n_classes=2,


        step=step,

        verbose=True

    )

    lvqnet.train(x_train, y_train, epochs=epochs)
    y_pred = lvqnet.predict(x_test)

    mae = (np.abs(y_test - y_pred)).mean()

    plt_x_zero = np.empty(0)
    plt_y_zero = np.empty(0)

    plt_x_one = np.empty(0)
    plt_y_one = np.empty(0)

    i = 0
    for coord in x_test:
        if y_pred[i] < 0.5:
            plt_x_zero = np.append(plt_x_zero, coord[0])
            plt_y_zero = np.append(plt_y_zero, coord[1])
        elif y_pred[i] >= 0.5:
            plt_x_one = np.append(plt_x_one, coord[0])
            plt_y_one = np.append(plt_y_one, coord[1])
        i += 1

    plt.plot(plt_x_zero, plt_y_zero, '.')
    plt.plot(plt_x_one, plt_y_one, '.')

    plt.title('\n2 class classification\nmae =%.4f' % (mae))

    plt.xlim(0, 1.3)
    plt.ylim(0, 1)

    plt.legend(('0 class', '1 class'), loc='upper right', shadow=True)

    plt.show()
    plt.close()
