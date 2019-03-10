from neupy import algorithms
from matplotlib import pyplot as plt
from neupy.exceptions import StopTraining
import numpy as np

from LABS.ZeroLab import E_Function as dataset5


def on_epoch_end(model):
    if model.train_errors.last() < goal_loss:
        raise StopTraining("Training has been interrupted")


def get_index(poss_code):
    for i in range(poss_code.__len__()):
        if poss_code[i] == 1:
            return i


if __name__ == '__main__':

    epochs = 100
    step = 0.5
    train_size = 4000

    goal_loss = 0.01

    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=train_size, mode=1)

    data = zip(x_train, y_train)

    for m in [5]:
        sofm = algorithms.SOFM(
            n_inputs=2,
            n_outputs=m * m,

            step=step,
            show_epoch=20,

            verbose=True,

            learning_radius=1,
            features_grid=(m, m, 1),

            epoch_end_signal=on_epoch_end

        )

        sofm.train(data, epochs=10)

        plt.plot(x_train, y_train, 'b.', label='real')
        plt.plot(sofm.weight[0], sofm.weight[1], 'r.', label='sofm weights')

        plt.legend(loc='upper right')
        plt.title("SOFM\nm = %.d\n lr = %.4f" % (m, step))

        plt.show()
        plt.close()

        weights_plt_x = []
        weights_plt_y = []

        for i in range(0, m * m):
            weights_plt_x.append(np.empty(0))
            weights_plt_y.append(np.empty(0))

        for x in np.arange(0.0, 1.0, step=0.01):
            for y in np.arange(0.0, 1.0, step=0.01):
                i = get_index(sofm.predict(np.array([x, y]))[0])

                weights_plt_x[i] = np.append(weights_plt_x[i], x)
                weights_plt_y[i] = np.append(weights_plt_y[i], y)

        for i in range(0, m * m):
            plt.plot(weights_plt_x[i], weights_plt_y[i])
        plt.title("Weights distribution for \nSOFM\nm = %.d\n lr = %.4f" % (m, step))
        plt.show()
        plt.close()
