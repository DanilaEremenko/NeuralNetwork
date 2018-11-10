import tensorflow as tf
from tensorflow import keras

import numpy as np

from ZeroLab.Programms.E_Function import function as f
from TfWithKeras.GUI import plot_history
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mnist = keras.datasets.mnist

    # (x_train, y_train), (x_test, y_test) = mnist.load_data(500)
    # x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.arange(0.0, 1.0, 0.001, dtype=float)
    y_train = np.array([f(y) for y in x_train])
    model = tf.keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=5)
    # plot_history(history)

    plt.plot(x_train, model.predict_proba(x_train))
    plt.plot(x_train, y_train,color='mediumvioletred')
    plt.show()
# model.evaluate(x_test, y_test)
