from __future__ import print_function

from keras.layers import Dense,Activation
from keras.models import Sequential


from ADDITIONAL.CUSTOM_KERAS import hard_lim
import numpy as np

import LABS.ZeroLab.B_LogicalFunc as dataset2

if __name__ == '__main__':
    (x_train, y_train) = dataset2.load_data()


    model = Sequential()

    model.add(Dense(5, input_dim=x_train.shape[1], activation=Activation(hard_lim),name='1',
                    weights=list([np.array([[-1.0, 1.0, 1.0, 1.0,1.0],
                                            [-1.0, -1.0, -1.0, -1.0, 1.0],
                                            [1.0, -1.0, -1.0, 1.0, -1.0],
                                            [-1.0, -1.0, 1.0, 1.0, -1.0],
                                            [1.0, 1.0, 1.0, 1.0, -1.0]], dtype=float),
                                  np.array([-1.5, -1.5, -2.5, -3.5,-1.5], dtype=float)])))
    model.add(Dense(1, activation=Activation(hard_lim),name='2',
                    weights=list([np.array([[1.0], [1.0], [1.0], [1.0],[1.0]], dtype=float),
                                  np.array([-0.5], dtype=float)])))


    i=0
    for pr in model.predict(x_train):
        print(float(dataset2.func_by_arr(x_train[i])),pr)
        i+=1