import numpy as np
from neupy import algorithms
from neupy.exceptions import StopTraining
from sklearn import metrics

import LABS.ZeroLab.F_ImagesGenerator as dataset8


def on_epoch_end(model):
    if model.train_errors.last() < goal_loss:
        raise StopTraining("Training has been interrupted")


if __name__ == '__main__':
    train_size = 10000
    test_size = 1000
    epochs = 10

    neu_num = 5
    goal_loss = 0.01
    step = 0.1

    verbose = True
    show_step = 20
    learning_radius = 1
    sofm = algorithms.SOFM(
        n_inputs=784,
        n_outputs=neu_num * neu_num,
        features_grid=(neu_num, neu_num, 1),

        step=step,
        show_epoch=show_step,

        verbose=True,

        learning_radius=learning_radius,

        epoch_end_signal=on_epoch_end

    )

    (x_train, y_train), (x_test, y_test) = dataset8.load_data(mode=0)

    sofm.train(zip(x_train, y_train), epochs=epochs)

    y_predicted = sofm.predict(x_test)

    print("accuracy = %.2f" % (metrics.accuracy_score(y_predicted, y_test)))

    y_predicted = sofm.predict(x_test)

    print("accuracy on noise data = %.2f" % (metrics.accuracy_score(y_predicted, y_test)))
