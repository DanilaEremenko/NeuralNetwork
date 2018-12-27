import numpy as np
from neupy.exceptions import StopTraining
from neupy import algorithms

import matplotlib.pyplot as plt


def on_epoch_end(model):
    if model.train_errors.last() < goal_loss:
        raise StopTraining("Training has been interrupted")


def ex_5_load_data(train_size=2000, show=False):
    x_train = np.random.rand(train_size, 2)
    plt.plot(x_train.reshape(2, train_size)[0], x_train.reshape(2, train_size)[1], '.')
    if show:
        plt.show()
    plt.close()

    return x_train


if __name__ == '__main__':
    step = 0.01
    goal_loss = 0.05

    train_size = 2000

    data = ex_5_load_data(train_size=2000)

    sofm = algorithms.SOFM(
        n_inputs=2,
        n_outputs=5 * 6,

        step=step,
        show_epoch=20,

        verbose=True,

        learning_radius=0,
        features_grid=(5, 6, 1),

        epoch_end_signal=on_epoch_end

    )

    plt.plot(sofm.weight[0], sofm.weight[1], '.')
    plt.title("sofm weigths after init")
    plt.show()
    plt.close()

    sofm.train(data, epochs=1)

    plt.plot(sofm.weight[0], sofm.weight[1], '.')
    plt.title("sofm weigths after 1 epoch")
    plt.show()
    plt.close()

    sofm.train(data, epochs=200)

    plt.plot(sofm.weight[0], sofm.weight[1], '.')
    plt.title("sofm weigths after 200 epoch")
    plt.show()
    plt.close()

    y_predict = sofm.predict(0.5, 0.3)
