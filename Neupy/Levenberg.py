from neupy.algorithms import LevenbergMarquardt
import LABS.ZeroLab.E_Function as dataset5
import matplotlib.pyplot as plt
from neupy import layers

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=6000, show=False)

    model = LevenbergMarquardt(
        [
            layers.Input(1),
            layers.Sigmoid(20),
            layers.Sigmoid(10),
            layers.Linear(1),

        ],

        error='mse',

        mu=0.1,

        verbose=True,

    )
    model.architecture()

    model.train(x_train, y_train, x_test, y_test, epochs=50)

    y_pred = model.predict(x_test)

    plt.plot(x_test, y_test, '.b')
    plt.plot(x_test, y_pred, '.r')

    plt.show()
