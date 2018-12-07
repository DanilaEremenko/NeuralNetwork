import keras
from keras.models import Sequential
from keras.layers import Dense

from LABS.ZeroLab import E_Function as dataset5

if __name__ == '__main__':
    train_size = 300

    (x_train, y_train), (x_test, y_test) = dataset5.load_data(train_size=train_size, show=True)

    model = Sequential()

    model.add(Dense(1, input_dim=1, activation='sigmoid'))
    model.add(Dense(20, activation='sigmoid'))
    model.add(Dense(1, activation='linear'))
