from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import numpy as np

from TfWithKeras.GUI_REPORTER import plot_history

if __name__ == '__main__':

    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([[0], [1], [1], [0]])

    model = Sequential()
    model.add(Dense(units=8, input_dim=2))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('hard_sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])

    history = model.fit(inputs, outputs, batch_size=1, nb_epoch=300)

    plot_history(history=history,save_path='history.png',save=True,show=False)

    print model.predict(inputs)

    model.save('XOR_MODEL.h5')
