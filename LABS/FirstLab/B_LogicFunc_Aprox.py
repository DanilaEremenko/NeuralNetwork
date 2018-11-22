from LABS.ZeroLab.Programms.B_LogicalFunc import load_data,func,func_by_arr
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import ADDITIONAL.GUI_REPORTER as gr
import numpy as np

if __name__ == '__main__':
    neur_number = 5
    (x_train, y_train), (x_test, y_test) = load_data()

    model = Sequential()

    model.add(Dense(neur_number, input_dim=x_train.shape[1], init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='hard_sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.0007), metrics=['accuracy'])

    # batch_size define speed of studying
    history = model.fit(x_train, y_train, batch_size=13, epochs=400, verbose=1)

    score = model.evaluate(x_test, y_test, verbose=1)
    print "accuracy on testing data %.f%%" % (score[1] * 100)

    gr.plot_history_separte(history, save_path_acc="ACC.png", save_path_loss="LOSS.png", save=False, show=True)
