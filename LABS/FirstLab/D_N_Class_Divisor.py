from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import ADDITIONAL.GUI_REPORTER as gr

import LABS.ZeroLab.Programms.D_DivIntoNClasses as dataset4

if __name__ == '__main__':
    train_size = 4000

    (x_train, y_train), (x_test, y_test) = dataset4.load_data(train_size=train_size, show=True)
