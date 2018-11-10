from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import numpy as np

import sys

sys.path.append("../TfWithKeras/")
from REPORTS_GUI import plot_history

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(units=8, input_dim=2))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])

history = model.fit(inputs, outputs, batch_size=1, nb_epoch=1000)

plot_history(history)

print(model.predict(inputs))

# model.save('XOR/XOR_MODEL.h5')
