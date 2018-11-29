from __future__ import print_function

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import ADDITIONAL.GUI_REPORTER as gr
import numpy as np

from keras.utils.vis_utils import plot_model

import LABS.ZeroLab.A_CrossZero as dataset1

if __name__ == '__main__':
    y_train = np.array([[1, 1, 1, 1],
                        [0, 1, 1, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 0]])

    (x_train, y_train) = dataset1.load_data(y_train, show=False)

    model = Sequential()

    model.add(Dense(4, input_dim=x_train.shape[1], activation='hard_sigmoid',
                    weights=list([np.array([[0.0, 0.0, 1.0, 1.0],
                                            [1.0, -1.0, 0.0, 0.0]], dtype=float),
                                  np.array([-0.25, 0.75, -0.25, -0.75], dtype=float)])))
    model.add(Dense(2, activation='hard_sigmoid',
                    weights=list([np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
                                  np.array([-2.5, -0.5], dtype=float)])))
    model.add(Dense(1, activation='hard_sigmoid',
                    weights=list([np.array([[1.0], [1.0]], dtype=float),
                                  np.array([-0.5], dtype=float)])))


    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
