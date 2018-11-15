import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

from TfWithKeras.GUI_REPORTER import plot_history

if __name__ == '__main__':

    np.random.seed(42)

    # x_train-image of numbers
    # y_train-answer of NN
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784)

    x_train = x_train.astype('float32')
    x_train = x_train / 255

    y_train = np_utils.to_categorical(y_train, 10)

    model = Sequential()

    # 800 neurons with 784 input,initialize - normal distribution
    model.add(Dense(800, input_dim=784, init='normal', activation='relu'))
    model.add(Dense(10,init='normal',activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])


    #batch_size define speed of studying
    history=model.fit(x_train,y_train,batch_size=200,nb_epoch=200,verbose=0)

    plot_history(history,'HISTORY_200.png', save=False, show=True)

    #model.save('MNIST_MODEL_200.h5')