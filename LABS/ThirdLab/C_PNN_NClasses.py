import numpy as np
import LABS.ZeroLab.D_DivIntoNClasses as dataset4

import matplotlib.pyplot as plt
from neupy import algorithms

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataset4.load_data_neupy(train_size=12000, show=True)

    for std in [0.1, 0.05, 0.01]:
        pnn = algorithms.PNN(std=std, verbose=True)

        pnn.train(x_train, y_train)

        y_predicted = pnn.predict(x_test)

        mae = (np.abs(y_test - y_predicted)).mean()

        plt_x_zero = np.empty(0)
        plt_y_zero = np.empty(0)

        plt_x_one = np.empty(0)
        plt_y_one = np.empty(0)

        plt_x_two = np.empty(0)
        plt_y_two = np.empty(0)

        plt_x_three = np.empty(0)
        plt_y_three = np.empty(0)

        plt_x_four = np.empty(0)
        plt_y_four = np.empty(0)

        plt_x_five = np.empty(0)
        plt_y_five = np.empty(0)

        plt_x_six = np.empty(0)
        plt_y_six = np.empty(0)

        i = 0
        for predict in y_predicted:

            if predict == 0.0:
                plt_x_zero = np.append(plt_x_zero, x_test[i][0])
                plt_y_zero = np.append(plt_y_zero, x_test[i][1])
            elif predict == 0.1:
                plt_x_one = np.append(plt_x_one, x_test[i][0])
                plt_y_one = np.append(plt_y_one, x_test[i][1])
            elif predict == 0.2:
                plt_x_two = np.append(plt_x_two, x_test[i][0])
                plt_y_two = np.append(plt_y_two, x_test[i][1])
            elif predict == 0.3:
                plt_x_three = np.append(plt_x_three, x_test[i][0])
                plt_y_three = np.append(plt_y_three, x_test[i][1])
            elif predict == 0.4:
                plt_x_four = np.append(plt_x_four, x_test[i][0])
                plt_y_four = np.append(plt_y_four, x_test[i][1])
            elif predict == 0.5:
                plt_x_five = np.append(plt_x_five, x_test[i][0])
                plt_y_five = np.append(plt_y_five, x_test[i][1])
            elif predict == 0.6:
                plt_x_six = np.append(plt_x_six, x_test[i][0])
                plt_y_six = np.append(plt_y_six, x_test[i][1])
            i += 1

        plt.plot(plt_x_zero, plt_y_zero, '.')
        plt.plot(plt_x_one, plt_y_one, '.')
        plt.plot(plt_x_two, plt_y_two, '.')
        plt.plot(plt_x_three, plt_y_three, '.')
        plt.plot(plt_x_four, plt_y_four, '.')
        plt.plot(plt_x_five, plt_y_five, '.')
        plt.plot(plt_x_six, plt_y_six, '.')

        plt.xlim(0, 1.5)
        plt.ylim(0, 1)

        plt.legend(('0.0 class', '0.1 class', '0.2 class', '0.3 class', '0.4 class', '0.5 class',
                    '0.6 class'), loc='upper right', shadow=True)

        plt.title('7 classification\nstd =%.4f\nmae = %.4f' % (std, mae))

        plt.show()
        plt.close()
